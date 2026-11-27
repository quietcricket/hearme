import XCTest
import StoreKit
import StoreKitTest

@MainActor final class PurchaseStoreTests: XCTestCase {
  func testVerifiedPurchasesExpirationAndRefund() async throws {
    let config = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().appendingPathComponent("StoreKit/HearME.storekit")
    let session = try SKTestSession(contentsOf: config)
    session.disableDialogs = true
    session.timeRate = .oneRenewalEveryTwoSeconds
    session.clearTransactions()
    defer { session.clearTransactions() }
    let store = PurchaseStore(storage: TestTrialStorage())
    for _ in 0..<100 where !store.ready {
      try await Task.sleep(for: .milliseconds(50))
    }
    await store.loadProducts()
    XCTAssertEqual(Set(store.products.map(\.id)), [PurchaseStore.monthlyID, PurchaseStore.lifetimeID])
    XCTAssertFalse(store.paid)
    XCTAssertEqual(store.products.first { $0.id == PurchaseStore.monthlyID }?.price, Decimal(string: "2.99"))
    XCTAssertEqual(store.products.first { $0.id == PurchaseStore.lifetimeID }?.price, Decimal(string: "29.99"))

    let monthly = try XCTUnwrap(store.products.first { $0.id == PurchaseStore.monthlyID })
    await store.purchase(monthly)
    await settle(store) { $0.paid }
    XCTAssertTrue(store.paid, store.notice ?? "Verified current entitlements did not grant access")
    XCTAssertFalse(store.lifetime)
    let subscription = try XCTUnwrap(session.allTransactions().first { $0.productIdentifier == PurchaseStore.monthlyID })
    try session.disableAutoRenewForTransaction(identifier: subscription.identifier)
    // Let the signed subscription actually expire at the accelerated rate.
    // This also exercises the per-product fallback when status lists are empty.
    await settle(store) { !$0.paid }
    XCTAssertFalse(store.paid)

    let lifetime = try XCTUnwrap(store.products.first { $0.id == PurchaseStore.lifetimeID })
    await store.purchase(lifetime)
    await settle(store) { $0.lifetime }
    XCTAssertTrue(store.paid, store.notice ?? "Verified current entitlements did not grant access")
    XCTAssertTrue(store.lifetime)
    await store.restore()
    XCTAssertTrue(store.lifetime)
    let reopened = PurchaseStore(storage: TestTrialStorage())
    for _ in 0..<100 where !reopened.ready {
      try await Task.sleep(for: .milliseconds(50))
    }
    XCTAssertTrue(reopened.paid)
    XCTAssertTrue(reopened.lifetime)
    let purchased = try XCTUnwrap(session.allTransactions().first { $0.productIdentifier == PurchaseStore.lifetimeID })
    try session.refundTransaction(identifier: purchased.identifier)
    await settle(store) { !$0.paid }
    XCTAssertFalse(store.paid)
    XCTAssertFalse(store.lifetime)

    try await session.setSimulatedError(.generic(.userCancelled), forAPI: .purchase)
    await store.purchase(monthly)
    XCTAssertFalse(store.paid, "Cancellation must not unlock access")
    try await session.setSimulatedError(nil, forAPI: .purchase)

    session.timeRate = .realTime
    session.askToBuyEnabled = true
    await store.purchase(monthly)
    XCTAssertFalse(store.paid, "Pending approval must not unlock access")
    XCTAssertTrue(store.notice?.contains("awaiting approval") == true)
    let pending = try XCTUnwrap(session.allTransactions().last {
      $0.productIdentifier == PurchaseStore.monthlyID && $0.state == .deferred
    })
    try session.approveAskToBuyTransaction(identifier: pending.identifier)
    await settle(store) { $0.paid }
    XCTAssertTrue(store.paid, "Approved transaction updates must unlock access")
  }

  private func settle(_ store: PurchaseStore, until condition: (PurchaseStore) -> Bool) async {
    for _ in 0..<100 {
      await store.refreshEntitlements()
      if condition(store) { return }
      try? await Task.sleep(for: .milliseconds(50))
    }
  }
}

private struct TestTrialStorage: TrialUsageStorage {
  func load() throws -> Double { 0 }
  func save(_ used: Double) throws {}
}
