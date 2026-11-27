import Foundation
import Security
import StoreKit
import SwiftUI

protocol TrialUsageStorage: Sendable {
  func load() throws -> Double
  func save(_ used: Double) throws
}

@MainActor final class PurchaseStore: ObservableObject {
  static let monthlyID = "local.hearme.app.monthly"
  static let lifetimeID = "local.hearme.app.lifetime"
  @Published private(set) var products: [Product] = []
  @Published private(set) var ready = false
  @Published private(set) var paid = false
  @Published private(set) var lifetime = false
  @Published private(set) var remaining: TimeInterval = TrialAllowance.duration
  @Published private(set) var busy = false
  @Published var notice: String?
  @Published var showPaywall = false
  private var trial = TrialAllowance(used: 0)
  private let trialStorage: any TrialUsageStorage
  private var storageHealthy = true
  private var listening = false
  private var updates: Task<Void, Never>?
  private var startup: Task<Void, Never>?
  private var timer: Timer?
  private var ticks = 0
  private var entitlementRevision = 0

  var canListen: Bool { ready && (paid || (storageHealthy && remaining > 0)) }
  var status: String {
    if !ready { return "Checking access…" }
    if lifetime { return "Lifetime access" }
    if paid { return "Monthly subscription active" }
    if !storageHealthy { return "Trial storage unavailable" }
    if remaining <= 0 { return "Your free trial has ended" }
    let minutes = Int(ceil(remaining / 60))
    return "Free trial · \(minutes / 60)h \(minutes % 60)m listening left"
  }

  init(storage: any TrialUsageStorage = KeychainTrialStorage()) {
    trialStorage = storage
    #if HEARME_DIRECT
    ready = true
    paid = true
    lifetime = true
    #else
    updates = Task { [weak self] in
      for await result in Transaction.updates {
        guard let self else { return }
        if case .verified(let transaction) = result {
          await self.refreshEntitlements()
          await transaction.finish()
        }
      }
    }
    startup = Task { [weak self] in
      guard let self else { return }
      let stored = await Task.detached(priority: .userInitiated) {
        Result { () throws -> Double in
          let used = try storage.load()
          try storage.save(used)
          return used
        }
      }.value
      switch stored {
      case .success(let used):
        self.trial = TrialAllowance(used: used)
        self.remaining = self.trial.remaining
      case .failure:
        self.storageHealthy = false
        self.notice = "HearME couldn’t access your trial time. Check Keychain access or restore a purchase."
      }
      await self.refreshEntitlements()
      self.ready = true
      await self.loadProducts()
    }
    timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
      Task { @MainActor in
        guard let self else { return }
        self.accountTime()
        self.ticks += 1
        if self.ticks % 15 == 0 { await self.refreshEntitlements() }
      }
    }
    #endif
  }

  deinit { updates?.cancel(); startup?.cancel(); timer?.invalidate() }

  func setListening(_ value: Bool) {
    accountTime()
    listening = value
    accountTime()
  }

  private func accountTime() {
    guard ready else { return }
    let previous = trial.used
    trial.update(listening: listening, paid: paid, uptime: ProcessInfo.processInfo.systemUptime)
    if previous != trial.used {
      do { try trialStorage.save(trial.used) } catch {
        storageHealthy = false
        notice = "HearME couldn’t save your trial time. Check Keychain access or restore a purchase."
      }
    }
    remaining = trial.remaining
  }

  func refreshEntitlements() async {
    entitlementRevision += 1
    let revision = entitlementRevision
    var hasMonthly = false
    var hasLifetime = false
    for await result in Transaction.currentEntitlements {
      guard case .verified(let transaction) = result, transaction.revocationDate == nil,
        !transaction.isUpgraded else { continue }
      if transaction.productID == Self.lifetimeID { hasLifetime = true }
      if transaction.productID == Self.monthlyID { hasMonthly = true }
    }
    // Some StoreKit environments return an empty current-entitlements sequence
    // immediately after a verified purchase. Check the signed per-product record
    // too; never treat an expired, upgraded, revoked or unverified record as access.
    if !hasLifetime, let result = await Transaction.latest(for: Self.lifetimeID),
      case .verified(let transaction) = result,
      transaction.productType == .nonConsumable,
      transaction.revocationDate == nil, !transaction.isUpgraded {
      hasLifetime = true
    }
    if !hasMonthly, let result = await Transaction.latest(for: Self.monthlyID),
      case .verified(let transaction) = result,
      transaction.productType == .autoRenewable,
      transaction.revocationDate == nil, !transaction.isUpgraded,
      let expiration = transaction.expirationDate, expiration > Date() {
      hasMonthly = true
    }
    // Subscription status is authoritative for billing grace, expiry and retry,
    // including a test/server expiry whose older signed transaction is cached.
    if let subscription = products.first(where: { $0.id == Self.monthlyID })?.subscription,
      let statuses = try? await subscription.status, !statuses.isEmpty {
      hasMonthly = statuses.contains { status in
        guard status.state == .subscribed || status.state == .inGracePeriod,
          case .verified(let transaction) = status.transaction,
          case .verified = status.renewalInfo else { return false }
        return transaction.productID == Self.monthlyID && transaction.revocationDate == nil
          && !transaction.isUpgraded
      }
    }
    guard revision == entitlementRevision else { return }
    accountTime()
    lifetime = hasLifetime
    paid = hasMonthly || hasLifetime
    accountTime()
  }

  func loadProducts() async {
    do {
      products = try await Product.products(for: [Self.monthlyID, Self.lifetimeID])
      if products.count != 2 {
        notice = "Purchases are unavailable right now. Please try again later."
      }
      await refreshEntitlements()
    } catch { notice = error.localizedDescription }
  }

  func purchase(_ product: Product) async {
    guard !busy, !lifetime else { return }
    busy = true
    notice = nil
    defer { busy = false }
    do {
      switch try await product.purchase() {
      case .success(let result):
        guard case .verified(let transaction) = result else {
          notice = "The purchase couldn’t be verified. Please restore purchases or try again."
          return
        }
        await refreshEntitlements()
        await transaction.finish()
        if paid { showPaywall = false }
      case .pending: notice = "Your purchase is awaiting approval. Access will unlock when it’s approved."
      case .userCancelled: break
      @unknown default: notice = "The purchase hasn’t completed. Please try again."
      }
    } catch { notice = error.localizedDescription }
  }

  func restore() async {
    guard !busy else { return }
    busy = true
    notice = nil
    defer { busy = false }
    do {
      try await AppStore.sync()
      await refreshEntitlements()
      notice = paid ? "Your purchases have been restored." : "No active purchases were found."
    } catch { notice = error.localizedDescription }
  }
}

/// Separate from preferences so clearing preferences doesn't reset the trial.
struct KeychainTrialStorage: TrialUsageStorage {
  private var query: [String: Any] {
    [kSecClass as String: kSecClassGenericPassword,
     kSecAttrService as String: "local.hearme.app.trial",
     kSecAttrAccount as String: "listening-seconds-v1"]
  }
  func load() throws -> Double {
    var request = query
    request[kSecReturnData as String] = true
    request[kSecUseAuthenticationUI as String] = kSecUseAuthenticationUIFail
    var result: CFTypeRef?
    let status = SecItemCopyMatching(request as CFDictionary, &result)
    if status == errSecItemNotFound { return 0 }
    guard status == errSecSuccess, let data = result as? Data,
      let text = String(data: data, encoding: .utf8), let value = Double(text),
      value.isFinite, value >= 0 else { throw CocoaError(.fileReadCorruptFile) }
    return value
  }
  func save(_ used: Double) throws {
    let values = [kSecValueData as String: Data(String(used).utf8)]
    var request = query
    request[kSecUseAuthenticationUI as String] = kSecUseAuthenticationUIFail
    var status = SecItemUpdate(request as CFDictionary, values as CFDictionary)
    if status == errSecItemNotFound {
      var item = query.merging(values) { _, new in new }
      item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
      status = SecItemAdd(item as CFDictionary, nil)
    }
    guard status == errSecSuccess else { throw CocoaError(.fileWriteUnknown) }
  }
}

struct PurchaseView: View {
  @ObservedObject var store: PurchaseStore
  @State private var showPrivacy = false
  var body: some View {
    Group {
      if showPrivacy {
        privacy
      } else {
        purchaseControls
      }
    }
  }

  private var purchaseControls: some View {
    VStack(alignment: .leading, spacing: 18) {
      Text("Keep meetings easy to hear").font(.title2.weight(.semibold))
      Text(store.status).foregroundStyle(.secondary)
      Text("Unlock unlimited listening with either plan.")
      ForEach([PurchaseStore.monthlyID, PurchaseStore.lifetimeID], id: \.self) { id in
        if let product = store.products.first(where: { $0.id == id }) {
          Button {
            Task { await store.purchase(product) }
          } label: {
            HStack {
              Text(id == PurchaseStore.monthlyID ? "Monthly" : "Lifetime")
              Spacer()
              Text(product.displayPrice + (id == PurchaseStore.monthlyID ? " / month" : " one time"))
            }.frame(maxWidth: .infinity).padding(8)
          }
          .disabled(store.busy || store.lifetime || (store.paid && id == PurchaseStore.monthlyID))
        }
      }
      if store.products.isEmpty {
        Text("Plans are currently unavailable.").foregroundStyle(.secondary)
        Button("Retry loading plans") { Task { await store.loadProducts() } }
      }
      Text("Monthly renews automatically until canceled. Manage or cancel in your App Store account. Lifetime is a single purchase. The free trial does not charge you automatically.")
        .font(.caption).foregroundStyle(.secondary)
      if store.paid && !store.lifetime {
        Text("Buying lifetime does not cancel an existing monthly subscription. Cancel it in your App Store account to avoid further renewals.")
          .font(.caption).foregroundStyle(.secondary)
      }
      if let notice = store.notice { Text(notice).font(.callout).textSelection(.enabled) }
      HStack {
        Button("Restore purchases") { Task { await store.restore() } }.disabled(store.busy)
        Link("Manage subscription", destination: URL(string: "https://apps.apple.com/account/subscriptions")!)
      }
      Button("Privacy policy") { showPrivacy = true }.buttonStyle(.link)
      Link("Terms of Use", destination: URL(string: "https://www.apple.com/legal/internet-services/itunes/dev/stdeula/")!)
        .font(.caption)
      HStack {
        if store.busy { ProgressView().controlSize(.small) }
        Spacer()
        Button("Done") { store.showPaywall = false }.keyboardShortcut(.cancelAction)
      }
    }.padding(24).frame(width: 420)
  }

  private var privacy: some View {
      VStack(alignment: .leading, spacing: 16) {
        Text("Privacy policy").font(.title2.weight(.semibold))
        Text("HearME processes system audio on your Mac. Meeting audio is not uploaded by HearME. Your microphone is not captured.")
        Text("Trial listening time is saved in your Mac’s Keychain. Preferences are stored locally. Apple processes purchases; HearME reads verified purchase status to unlock access and does not receive your payment card details.")
        Link("Apple’s privacy policy", destination: URL(string: "https://www.apple.com/legal/privacy/")!)
        HStack { Spacer(); Button("Done") { showPrivacy = false } }
      }.padding(24).frame(width: 400)
  }
}
