import AppKit
import Carbon
import CoreAudio
import Combine
import ServiceManagement
import SwiftUI

/// A global key combination: Carbon key code and modifier mask, plus its display label.
struct HotKey: Codable, Equatable {
  var keyCode: UInt32
  var modifiers: UInt32
  var label: String

  static let standard = HotKey(keyCode: UInt32(kVK_ANSI_H),
                               modifiers: UInt32(controlKey | optionKey), label: "⌃⌥H")

  /// Builds a hot key from a key press; nil unless ⌘, ⌃ or ⌥ is held.
  init?(event: NSEvent) {
    let flags = event.modifierFlags.intersection([.command, .control, .option, .shift])
    guard !flags.isDisjoint(with: [.command, .control, .option]) else { return nil }
    var mask = 0, symbols = ""
    if flags.contains(.control) { mask |= controlKey; symbols += "⌃" }
    if flags.contains(.option) { mask |= optionKey; symbols += "⌥" }
    if flags.contains(.shift) { mask |= shiftKey; symbols += "⇧" }
    if flags.contains(.command) { mask |= cmdKey; symbols += "⌘" }
    let named: [UInt16: String] = [
      49: "Space", 36: "↩", 48: "⇥", 51: "⌫", 117: "⌦", 123: "←", 124: "→", 125: "↓", 126: "↑",
      122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5", 97: "F6", 98: "F7", 100: "F8",
      101: "F9", 109: "F10", 103: "F11", 111: "F12",
    ]
    let key = named[event.keyCode] ?? (event.charactersIgnoringModifiers ?? "?").uppercased()
    self.init(keyCode: UInt32(event.keyCode), modifiers: UInt32(mask), label: symbols + key)
  }

  init(keyCode: UInt32, modifiers: UInt32, label: String) {
    self.keyCode = keyCode
    self.modifiers = modifiers
    self.label = label
  }
}

private final class ListeningShortcut {
  private var hotKey: EventHotKeyRef?
  private var handler: EventHandlerRef?
  private let action: () -> Void
  var isRegistered: Bool { hotKey != nil }

  init(_ key: HotKey, action: @escaping () -> Void) {
    self.action = action
    var event = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                              eventKind: UInt32(kEventHotKeyPressed))
    let status = InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
      guard let event, let context else { return OSStatus(eventNotHandledErr) }
      var identifier = EventHotKeyID()
      let result = GetEventParameter(event, EventParamName(kEventParamDirectObject),
                                     EventParamType(typeEventHotKeyID), nil,
                                     MemoryLayout<EventHotKeyID>.size, nil, &identifier)
      guard result == noErr, identifier.signature == 0x484D4554, identifier.id == 1 else {
        return OSStatus(eventNotHandledErr)
      }
      Unmanaged<ListeningShortcut>.fromOpaque(context).takeUnretainedValue().action()
      return noErr
    }, 1, &event, Unmanaged.passUnretained(self).toOpaque(), &handler)
    guard status == noErr else { return }
    var identifier = EventHotKeyID(signature: 0x484D4554, id: 1)
    let registration = RegisterEventHotKey(key.keyCode, key.modifiers,
                                           identifier, GetApplicationEventTarget(), 0, &hotKey)
    if registration != noErr {
      hotKey = nil
      if let handler { RemoveEventHandler(handler); self.handler = nil }
    }
  }

  deinit {
    if let hotKey { UnregisterEventHotKey(hotKey) }
    if let handler { RemoveEventHandler(handler) }
  }
}

@MainActor final class AppModel: ObservableObject {
  @Published var systemOutputName = "macOS audio output"
  @Published var running = false
  @Published var loading = false
  @Published var stopping = false
  @Published var recovering = false
  @Published var modelPreparing = false
  @Published var gain: Double = 0.9 {
    didSet {
      session?.setGain(Float(gain))
      UserDefaults.standard.set(gain, forKey: "gain")
    }
  }
  @Published var blockSeconds: Double = 0.1 {
    didSet {
      UserDefaults.standard.set(blockSeconds, forKey: "blockSeconds")
      guard oldValue != blockSeconds else { return }
      // A running session restarts through the route-change path with the new buffer.
      if session != nil { reconcileOutput(force: true) } else { prepareModel() }
    }
  }
  @Published var message = "Ready when you are."
  @Published var failure: String?
  @Published var computeMS: Double = 0
  @Published var realtimeFactor: Double = 0
  @Published var queuedSeconds: Double = 0
  @Published var latencySeconds: Double = 0
  @Published var missingSamples: UInt64 = 0
  @Published var capturedSeconds: Double = 0
  @Published var inputDB: Double = -120
  @Published var outputDB: Double = -120
  /// Recent output peaks (0...1), oldest first, for the panel's LED meter.
  @Published private(set) var levelHistory = [Double](repeating: 0, count: 28)
  private var meterTimer: Timer?
  /// Sample levels at 12 Hz only while the panel is open; 1 Hz otherwise.
  var meterVisible = false {
    didSet {
      guard meterVisible != oldValue else { return }
      meterTimer?.invalidate()
      meterTimer = nil
      guard meterVisible else { return }
      meterTimer = Timer.scheduledTimer(withTimeInterval: 1 / 12, repeats: true) { [weak self] _ in
        Task { @MainActor in self?.sampleLevels() }
      }
    }
  }
  @Published private(set) var shortcutAvailable = false
  /// nil means the user cleared the shortcut.
  @Published private(set) var hotKey: HotKey? = .standard
  let purchases = PurchaseStore()
  private var accessObserver: AnyCancellable?
  private let modelStore = ModelStore()
  private var preparationID = UUID()
  private var quitting = false
  private var session: LiveSession?
  private var healthTimer: Timer?
  private let outputMonitor = SystemOutputMonitor()
  private var routedOutput: OutputDevice?
  private var wantsListening = false
  private var routeUpdate: Task<Void, Never>?
  private var shortcut: ListeningShortcut?

  init() {
    accessObserver = purchases.objectWillChange.sink { [weak self] _ in
      DispatchQueue.main.async {
        guard let self, !self.purchases.canListen, self.wantsListening else { return }
        self.stop()
        self.purchases.showPaywall = true
      }
    }
    if let saved = UserDefaults.standard.object(forKey: "blockSeconds") as? Double,
      [0.1, 0.2, 0.5].contains(saved)
    {
      blockSeconds = saved
    }
    if let saved = UserDefaults.standard.object(forKey: "gain") as? Double {
      gain = max(0, min(1.5, saved))
    }
    reconcileOutput()
    do {
      try outputMonitor.start { [weak self] in
        Task { @MainActor in self?.scheduleOutputUpdate() }
      }
    } catch { failure = error.localizedDescription }
    healthTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
      Task { @MainActor in
        guard let self else { return }
        self.capturedSeconds = self.session?.capturedSeconds ?? 0
        if self.meterTimer == nil { self.sampleLevels() }
        self.reconcileOutput()
      }
    }
    if let data = UserDefaults.standard.data(forKey: "hotKey"),
      let saved = try? JSONDecoder().decode(HotKey?.self, from: data)
    {
      hotKey = saved
    }
    registerHotKey()
    prepareModel()
  }

  /// Peaks reset on each read, so only one timer samples at a time.
  private func sampleLevels() {
    let levels = session?.signalLevels() ?? (0, 0)
    inputDB = 20 * log10(max(Double(levels.0), 0.000001))
    outputDB = 20 * log10(max(Double(levels.1), 0.000001))
    levelHistory.removeFirst()
    levelHistory.append(max(0, min(1, (outputDB + 54) / 54)))
  }

  func setHotKey(_ key: HotKey?) {
    hotKey = key
    UserDefaults.standard.set(try? JSONEncoder().encode(key), forKey: "hotKey")
    registerHotKey()
  }

  /// Releases the global shortcut so a recorder can see the key press.
  func pauseHotKey() {
    shortcut = nil
  }

  func registerHotKey() {
    shortcut = nil
    if let hotKey {
      shortcut = ListeningShortcut(hotKey) { [weak self] in
        Task { @MainActor in self?.toggleListening() }
      }
    }
    shortcutAvailable = shortcut?.isRegistered == true
  }

  func toggleListening() {
    guard !stopping, !quitting else { return }
    if wantsListening { stop() } else { start() }
  }

  /// Start loading at application launch; changing an idle buffer prepares
  /// that variant once. Start/reconnect requests reuse the same store.
  private func prepareModel() {
    guard session == nil, !quitting else { return }
    let generation = UUID()
    preparationID = generation
    modelPreparing = true
    message = "Preparing voice enhancement…"
    do {
      let runtime = try ModelRuntime.resolve(blockSeconds: blockSeconds)
      modelStore.prepare(
        runtime: runtime,
        progress: { [weak self] percent in
          DispatchQueue.main.async {
            guard let self, self.preparationID == generation, !self.quitting,
              self.session == nil, !self.wantsListening
            else { return }
            self.message = "Preparing voice enhancement… \(percent)%"
          }
        },
        completion: { [weak self] result in
          DispatchQueue.main.async {
            guard let self, self.preparationID == generation, !self.quitting else { return }
            self.modelPreparing = false
            guard self.session == nil, !self.wantsListening else { return }
            switch result {
            case .success:
              self.message = "Ready when you are."
            case .failure(let error):
              self.failure = error.localizedDescription
              self.message = "Voice enhancement could not prepare. Start listening to retry."
            }
          }
        })
    } catch {
      modelPreparing = false
      failure = error.localizedDescription
    }
  }

  private func scheduleOutputUpdate() {
    routeUpdate?.cancel()
    routeUpdate = Task { @MainActor [weak self] in
      do { try await Task.sleep(nanoseconds: 250_000_000) } catch { return }
      self?.reconcileOutput()
    }
  }

  private func reconcileOutput(force: Bool = false) {
    let device = try? OutputDevice.systemDefault()
    systemOutputName = device?.name ?? "No macOS audio output"
    guard wantsListening, !quitting, purchases.canListen else { return }
    if session != nil {
      if !stopping && (force || device != routedOutput) { endSession() }
      return
    }
    guard !stopping else { return }
    guard let device else {
      loading = true
      message = "Waiting for the macOS audio output…"
      return
    }
    startSession(device: device)
  }

  func start() {
    guard !wantsListening else { return }
    guard purchases.canListen else { purchases.showPaywall = true; return }
    wantsListening = true
    failure = nil
    reconcileOutput()
  }

  private func startSession(device: OutputDevice) {
    let runtime: ModelRuntime
    do { runtime = try ModelRuntime.resolve(blockSeconds: blockSeconds) } catch {
      wantsListening = false
      loading = false
      failure = error.localizedDescription
      return
    }
    failure = nil
    loading = true
    recovering = false
    computeMS = 0
    missingSamples = 0
    capturedSeconds = 0
    queuedSeconds = 0
    latencySeconds = 0
    inputDB = -120
    outputDB = -120
    message = "Getting voice enhancement ready…"
    let active = LiveSession(blockSeconds: blockSeconds, modelStore: modelStore)
    session = active
    routedOutput = device
    active.setEnhanced(true)
    active.setGain(Float(gain))
    active.onPreparing = { [weak self, weak active] percent in
      guard let self, let active, self.session === active, !self.stopping else { return }
      self.message = "Preparing voice enhancement… \(percent)%"
    }
    var modelReady = false
    active.onConnecting = { [weak self, weak active] in
      guard let self, let active, self.session === active else { return }
      modelReady = true
      self.message = "Connecting macOS audio output…"
      DispatchQueue.main.asyncAfter(deadline: .now() + 20) { [weak self, weak active] in
        guard let self, let active, self.session === active, self.loading else { return }
        self.stop()
        self.failure =
          "Audio capture startup timed out. Refresh HearME’s system-audio permission for this build, then reopen the app."
      }
    }
    active.onRunning = { [weak self, weak active] in
      guard let self, let active, self.session === active else { return }
      self.loading = false
      self.running = true
      self.purchases.setListening(true)
      self.message = "Listening through your selected output."
      self.reconcileOutput()
    }
    active.onRouteInvalidated = { [weak self, weak active] in
      guard let self, let active, self.session === active else { return }
      self.reconcileOutput(force: true)
    }
    active.onRecovery = { [weak self, weak active] cooling in
      guard let self, let active, self.session === active, !self.stopping else { return }
      self.recovering = cooling
      self.message = cooling
        ? "Audio is still playing. Enhancement will resume shortly…"
        : "Listening through your selected output."
    }
    active.onFailure = { [weak self, weak active] detail in
      guard let self, let active, self.session === active, !self.stopping else { return }
      // Route changes can interrupt the old stream before HAL's listener
      // arrives. Treat that as a reconnect, not a user-requested Stop.
      if self.wantsListening, (try? OutputDevice.systemDefault()) != self.routedOutput {
        self.reconcileOutput(force: true)
        return
      }
      self.failure = detail
      self.stop()
    }
    active.onMetrics = { [weak self, weak active] compute, rtf, queued, missing, latency in
      DispatchQueue.main.async {
        guard let self, let active, self.session === active, !self.stopping else { return }
        self.latencySeconds = latency
        self.computeMS = compute
        self.realtimeFactor = rtf
        self.queuedSeconds = queued
        self.missingSamples = missing
      }
    }
    do { try active.start(runtime: runtime, device: device) } catch {
      stop()
      failure = error.localizedDescription
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + 180) { [weak self, weak active] in
      guard let self, let active, self.session === active, self.loading, !modelReady else { return }
      self.stop()
      self.failure = "Model preparation timed out. Original system audio is unchanged."
    }
  }

  func stop() {
    wantsListening = false
    routeUpdate?.cancel()
    routeUpdate = nil
    endSession()
  }

  private func endSession() {
    purchases.setListening(false)
    recovering = false
    guard let active = session else {
      running = false
      loading = false
      stopping = false
      message = "Stopped. Original system audio is restored."
      return
    }
    guard !stopping else { return }
    running = false
    loading = false
    stopping = true
    message =
      wantsListening ? "Following the macOS audio output…" : "Restoring original system audio…"
    active.stop()
    active.afterShutdown { [weak self, weak active] in
      guard let self, let active, self.session === active else { return }
      self.session = nil
      self.routedOutput = nil
      self.stopping = false
      self.message = "Stopped. Original system audio is restored."
      self.reconcileOutput()
    }
  }

  func prepareToQuit(_ completion: @escaping () -> Void) {
    quitting = true
    modelStore.cancelPreparation()
    let active = session
    stop()
    let finish = { [modelStore] in
      modelStore.shutdown { DispatchQueue.main.async { completion() } }
    }
    if let active { active.afterShutdown(finish) } else { finish() }
  }
}

struct ContentView: View {
  @ObservedObject var model: AppModel
  @ObservedObject private var purchases: PurchaseStore
  @State private var showDiagnostics = false
  @State private var showShortcut = false
  @State private var recording = false
  @State private var keyMonitor: Any?
  @State private var launchAtLogin = false

  init(model: AppModel) {
    self.model = model
    self.purchases = model.purchases
  }

  var body: some View {
    Group {
      if purchases.showPaywall {
        PurchaseView(store: purchases)
      } else if showShortcut {
        shortcutSettings
      } else if showDiagnostics {
        diagnostics
      } else {
        mainControls
      }
    }
    .onAppear { model.meterVisible = true }
    .onDisappear { model.meterVisible = false }
  }

  private enum Palette {
    static let body = Color(red: 0.086, green: 0.086, blue: 0.086)
    static let screen = Color(red: 0.06, green: 0.06, blue: 0.06)
    static let pad = Color(white: 0.17)
    static let chip = Color(white: 0.15)
    static let orange = Color(red: 1.0, green: 0.36, blue: 0.17)
    static let cream = Color(red: 0.97, green: 0.92, blue: 0.82)
    static let ink = Color(red: 0.80, green: 0.22, blue: 0.10)
    static let yellow = Color(red: 1.0, green: 0.88, blue: 0.21)
    static let green = Color(red: 0.16, green: 0.54, blue: 0.33)
  }

  private var mainControls: some View {
    VStack(spacing: 8) {
      display

      HStack(spacing: 8) {
        volumePad
        powerKey
      }

      if let failure = model.failure {
        VStack(alignment: .leading, spacing: 6) {
          Label("HearME needs attention", systemImage: "exclamationmark.triangle.fill")
            .font(.callout.weight(.semibold)).foregroundStyle(Palette.orange)
          Text(failure).font(.callout).foregroundStyle(.white.opacity(0.85))
            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
          Button("Open audio permissions") {
            NSWorkspace.shared.open(
              URL(string:
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
          }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12).background(Palette.pad, in: RoundedRectangle(cornerRadius: 10))
      }

      #if !HEARME_DIRECT
      AccessSummary(store: model.purchases)
        .padding(.horizontal, 4)
      #endif

      footer
    }
    .padding(12).frame(width: 320).fixedSize(horizontal: false, vertical: true)
    .background(Palette.body)
    .environment(\.colorScheme, .dark)
    .onAppear { launchAtLogin = SMAppService.mainApp.status == .enabled }
    .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification))
    { _ in model.stop() }
  }

  private var status: String {
    let output = model.systemOutputName
    if model.loading || model.stopping || (model.modelPreparing && !model.running) {
      return model.message
    }
    if model.running { return model.recovering ? model.message : "Enhancing · \(output)" }
    return "Off"
  }

  /// Dot-matrix output meter: one column per recent peak, newest on the right.
  private var display: some View {
    let live = model.running
    let history = model.levelHistory
    return VStack(alignment: .leading, spacing: 10) {
      Canvas { context, size in
        let columns = history.count, rows = 9
        let step = size.width / CGFloat(columns)
        let dot = step - 3
        for (column, level) in history.enumerated() {
          let lit = live ? Int((level * Double(rows)).rounded()) : 0
          for row in 0..<rows {
            let on = rows - row <= max(lit, live ? 1 : 0)
            let rect = CGRect(x: CGFloat(column) * step + 1.5,
                              y: CGFloat(row) * step + 1.5, width: dot, height: dot)
            let color: Color = !on ? Color(white: 0.16)
              : rows - row > 7 ? Palette.yellow : Palette.orange
            context.fill(Path(roundedRect: rect, cornerRadius: 1.5), with: .color(color))
          }
        }
      }
      .aspectRatio(28 / 9, contentMode: .fit)
      .accessibilityLabel("Output level")
      .accessibilityValue(live ? String(format: "%.0f dBFS", model.outputDB) : "Off")

      Text(status.uppercased())
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .foregroundStyle(live ? Palette.orange : Color(white: 0.55))
        .lineLimit(2).fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
    .padding(12)
    .background(Palette.screen, in: RoundedRectangle(cornerRadius: 12))
  }

  private var volumePad: some View {
    let boost = model.gain > 1.005
    let percent = Int((model.gain * 100).rounded())
    return VStack(alignment: .leading, spacing: 8) {
      HStack(alignment: .firstTextBaseline, spacing: 2) {
        Text("\(percent)").font(.system(size: 26, weight: .regular)).monospacedDigit()
        Text("%").font(.callout)
        Spacer()
        Text(boost ? "Boost" : "Volume").font(.caption)
      }
      .foregroundStyle(boost ? Palette.yellow : .white)
      Slider(value: Binding(
        get: { model.gain },
        // Snap to 100% so the unboosted level is easy to find.
        set: { model.gain = abs($0 - 1) < 0.04 ? 1 : $0 }), in: 0...1.5)
      .controlSize(.small)
      .tint(boost ? Palette.yellow : Palette.orange)
      .background(alignment: .bottomLeading) {
        // ponytail: tick placed by ratio with an estimated knob inset; adjust if the track moves.
        GeometryReader { geo in
          Capsule().fill(.white.opacity(0.5)).frame(width: 2, height: 5)
            .offset(x: 7 + (geo.size.width - 14) * (1 / 1.5) - 1, y: geo.size.height + 1)
        }
      }
      .accessibilityLabel("Playback volume")
      .accessibilityValue("\(percent) percent")
    }
    .padding(10)
    .frame(maxWidth: .infinity, minHeight: 74, alignment: .leading)
    .background(Palette.pad, in: RoundedRectangle(cornerRadius: 10))
  }

  private var powerKey: some View {
    let on = model.running || model.loading
    return Button {
      model.toggleListening()
    } label: {
      ZStack {
        Circle().fill(on ? Palette.green : Palette.pad)
        if model.loading || model.stopping {
          ProgressView().controlSize(.small)
        } else {
          Image(systemName: "power").font(.system(size: 22, weight: .semibold))
            .foregroundStyle(on ? Color.white : Color(white: 0.6))
        }
      }
      .frame(width: 74, height: 74)
      .contentShape(Circle())
    }
    .buttonStyle(.plain)
    .frame(width: (296 - 16) / 3)
    .disabled(model.stopping)
    .keyboardShortcut(.return, modifiers: [])
    .accessibilityLabel("Voice enhancement")
    .accessibilityValue(model.running ? "On" : "Off")
    .accessibilityHint(model.loading ? "Cancel turning on" : model.running ? "Turn off" : "Turn on")
    .help(model.hotKey.map { model.shortcutAvailable ? "Turn on or off · \($0.label)"
                                : "Turn on or off (\($0.label) unavailable)" } ?? "Turn on or off")
  }

  private var footer: some View {
    HStack(spacing: 6) {
      Circle()
        .fill(model.running ? Palette.green : model.loading ? Palette.yellow : Color(white: 0.4))
        .frame(width: 8, height: 8)
        .frame(width: 28, height: 26)
        .background(Palette.chip, in: RoundedRectangle(cornerRadius: 7))
        .accessibilityHidden(true)
      HStack(spacing: 5) {
        Image(systemName: "headphones").font(.caption)
        Text(model.systemOutputName).lineLimit(1).truncationMode(.middle)
      }
      .font(.callout).foregroundStyle(Color(white: 0.75))
      .padding(.horizontal, 10).frame(height: 26)
      .background(Palette.chip, in: RoundedRectangle(cornerRadius: 7))
      .help("macOS audio output")
      Spacer(minLength: 0)
      moreMenu
        .frame(width: 28, height: 26)
        .background(Palette.chip, in: RoundedRectangle(cornerRadius: 7))
    }
  }

  private var moreMenu: some View {
    Menu {
      Toggle("Launch at Login", isOn: Binding(
        get: { launchAtLogin },
        set: { enable in
          do {
            if enable { try SMAppService.mainApp.register() }
            else { try SMAppService.mainApp.unregister() }
          } catch { model.failure = error.localizedDescription }
          launchAtLogin = SMAppService.mainApp.status == .enabled
        }))
      Button("Keyboard Shortcut…") { showShortcut = true }
      Button("Diagnostics…") { showDiagnostics = true }
      Button("About HearME") {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.orderFrontStandardAboutPanel(options: [
          .credits: NSAttributedString(
            string: "All app audio is enhanced on your Mac. The microphone stays private.",
            attributes: [.font: NSFont.systemFont(ofSize: NSFont.smallSystemFontSize)]),
        ])
      }
      Divider()
      Button("Quit HearME") { NSApplication.shared.terminate(nil) }
        .keyboardShortcut("q")
    } label: {
      Image(systemName: "ellipsis")
    }
    .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
    .accessibilityLabel("More")
  }

  private var shortcutSettings: some View {
    VStack(alignment: .leading, spacing: 12) {
      VStack(alignment: .leading, spacing: 4) {
        Text("Keyboard shortcut").font(.title3.weight(.semibold))
        Text("Turns HearME on or off from any app.")
          .font(.callout).foregroundStyle(.secondary)
      }
      Button {
        recording ? stopRecording() : startRecording()
      } label: {
        Text(recording ? "Press keys…" : model.hotKey?.label ?? "None")
          .font(.system(size: 22, weight: .medium, design: .monospaced))
          .foregroundStyle(recording ? Palette.orange : .white)
          .frame(maxWidth: .infinity, minHeight: 56)
          .background(Palette.screen, in: RoundedRectangle(cornerRadius: 12))
          .overlay(RoundedRectangle(cornerRadius: 12)
            .strokeBorder(recording ? Palette.orange : .clear, lineWidth: 1.5))
          .contentShape(RoundedRectangle(cornerRadius: 12))
      }
      .buttonStyle(.plain)
      .accessibilityLabel("Shortcut")
      .accessibilityValue(model.hotKey?.label ?? "None")
      .accessibilityHint(recording ? "Press the new key combination" : "Record a new shortcut")
      Text(recording ? "Hold ⌘, ⌃ or ⌥ and press a key. Esc cancels."
           : "Click to record a new shortcut.")
        .font(.caption).foregroundStyle(.secondary)
      if !recording, model.hotKey != nil, !model.shortcutAvailable {
        Text("Another app is using this shortcut. Choose a different one.")
          .font(.callout).foregroundStyle(Palette.orange)
          .fixedSize(horizontal: false, vertical: true)
      }
      HStack {
        Button("Reset to \(HotKey.standard.label)") { stopRecording(); model.setHotKey(.standard) }
          .disabled(model.hotKey == .standard)
        Button("Clear") { stopRecording(); model.setHotKey(nil) }
          .disabled(model.hotKey == nil)
        Spacer()
        Button("Done") { stopRecording(); showShortcut = false }.keyboardShortcut(.defaultAction)
      }
    }
    .padding(16).frame(width: 320)
    .background(Palette.body)
    .environment(\.colorScheme, .dark)
    .onDisappear { stopRecording() }
  }

  private func startRecording() {
    recording = true
    model.pauseHotKey()
    // Menu bar panels don't receive keys while another app is active.
    NSApp.activate()
    NSApp.windows.first { $0.isVisible && $0.canBecomeKey }?.makeKey()
    keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
      if event.keyCode == 53 {  // Esc
        stopRecording()
      } else if let key = HotKey(event: event) {
        recording = false
        removeMonitor()
        model.setHotKey(key)
      } else {
        NSSound.beep()
      }
      return nil
    }
  }

  private func stopRecording() {
    guard recording else { return }
    recording = false
    removeMonitor()
    model.registerHotKey()
  }

  private func removeMonitor() {
    if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
    keyMonitor = nil
  }

  private var diagnostics: some View {
    VStack(alignment: .leading, spacing: 12) {
      VStack(alignment: .leading, spacing: 4) {
        Text("Audio diagnostics").font(.title3.weight(.semibold))
        Text(!model.running ? "Turn on to see live measurements."
             : model.capturedSeconds == 0 ? "Waiting for audio on \(model.systemOutputName)…"
             : model.recovering ? model.message
             : "Live · \(model.systemOutputName)")
          .font(.callout).foregroundStyle(.secondary)
      }
      Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 8) {
        diagnostic("Audio received", measured(String(format: "%.1f s", model.capturedSeconds)))
        diagnostic("Model time", measured(String(format: "%.0f of %.0f ms",
                                                 model.computeMS, model.blockSeconds * 1000)))
        diagnostic("Queued audio", measured(String(format: "%.0f ms", model.queuedSeconds * 1000)))
        diagnostic("Dropped audio",
                   measured(String(format: "%.0f ms", Double(model.missingSamples) / 48)))
        diagnostic("Added delay", measured(String(format: "%.0f ms", model.latencySeconds * 1000)))
        diagnostic("Input level", level(model.inputDB))
        diagnostic("Output level", level(model.outputDB))
      }
      .font(.system(.callout, design: .monospaced)).textSelection(.enabled)
      .frame(maxWidth: .infinity, alignment: .leading)
      .padding(12).background(Palette.screen, in: RoundedRectangle(cornerRadius: 12))
      VStack(alignment: .leading, spacing: 6) {
        Picker("Buffer", selection: $model.blockSeconds) {
          Text("100 ms").tag(0.1)
          Text("200 ms").tag(0.2)
          Text("500 ms").tag(0.5)
        }
        .pickerStyle(.segmented)
        Text("Larger buffers add delay but help if audio drops out. Changing it restarts enhancement briefly.")
          .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
      }
      Text("Added delay is measured from capture to playback (tap → render). It excludes speaker or headphone latency.")
        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
      if model.running && model.realtimeFactor > 0.95 {
        Text("Your Mac is barely keeping up. Choose a larger buffer if audio drops out.")
          .font(.callout).fixedSize(horizontal: false, vertical: true).foregroundStyle(Palette.orange)
      }
      HStack {
        Text("TF-Restormer · MLX · 48 kHz mono · v\(Self.version)").font(.caption).foregroundStyle(.secondary)
        Spacer()
        Button("Done") { showDiagnostics = false }.keyboardShortcut(.defaultAction)
      }
    }
    .padding(16).frame(width: 320)
    .background(Palette.body)
    .environment(\.colorScheme, .dark)
  }

  private static let version =
    Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"

  /// Show a dash instead of zeros when nothing is being measured.
  private func measured(_ value: String) -> String { model.running ? value : "—" }

  private func level(_ db: Double) -> String {
    model.running && db > -119 ? "\(Int(db.rounded())) dBFS" : "—"
  }

  private func diagnostic(_ title: String, _ value: String) -> some View {
    GridRow {
      Text(title).foregroundStyle(.secondary)
      Text(value).monospacedDigit()
    }
  }
}

struct AccessSummary: View {
  @ObservedObject var store: PurchaseStore
  var body: some View {
    HStack {
      Text(store.status).font(.caption).foregroundStyle(.secondary)
      Spacer()
      Button(store.paid ? "Purchases…" : "Unlock…") { store.showPaywall = true }
        .disabled(!store.ready)
    }
  }
}

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
  weak var model: AppModel?
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
  func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
    guard let model else { return .terminateNow }
    model.prepareToQuit { sender.reply(toApplicationShouldTerminate: true) }
    return .terminateLater
  }
}

@main struct HearMEApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
  @StateObject private var model = AppModel()
  var body: some Scene {
    MenuBarExtra {
      ContentView(model: model)
    } label: {
      Image(systemName: model.running ? "ear.fill" : "ear")
        .font(.system(size: 16, weight: .medium))
        .accessibilityLabel(model.running ? "HearME on" : "HearME off")
        .onAppear { delegate.model = model }
    }
    .menuBarExtraStyle(.window)
  }
}
