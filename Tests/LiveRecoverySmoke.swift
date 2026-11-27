// Build with the actual LiveSession/ModelWorker, plus the inert hardware and
// fake model below. No capture permission or live audio output is needed.
import AVFoundation
import AudioRing
import Foundation

struct AudioFailure: LocalizedError {
  let message: String
  var errorDescription: String? { message }
}
struct OutputDevice { let id: UInt32 = 0 }
final class SystemAudioTap {
  let sampleRate = 48000.0
  func start(ring: OpaquePointer, output: OutputDevice) throws {}
  func stop() {}
}
final class AudioPlayback {
  var onConfigurationChange: (() -> Void)?
  func start(device: OutputDevice, ring: OpaquePointer) throws {}
  func stop() {}
}
final class DelayedEngine: ModelProcessing {
  let sampleCount = 4800
  private let lock = NSLock()
  private var delay = 0.0
  private var state: Float = 0
  func setDelay(_ value: Double) { lock.lock(); delay = value; lock.unlock() }
  func process(_ samples: [Float]) throws -> [Float] {
    lock.lock(); let seconds = delay; lock.unlock()
    if seconds > 0 { Thread.sleep(forTimeInterval: seconds) }
    let result = samples.map { $0 * 0.5 + state }
    if samples.contains(where: { abs($0) > 1e-6 }) { state += 0.001 }
    return result
  }
  func reset() { state = 0 }
}
private func spin(_ seconds: Double) {
  RunLoop.current.run(until: Date().addingTimeInterval(seconds))
}
@main struct LiveRecoverySmoke {
  static func main() throws {
    let root = URL(fileURLWithPath: "/test")
    let runtime = ModelRuntime(assets: root, frames: 5)
    let engine = DelayedEngine()
    let store = ModelStore { _, _, _ in engine }
    let session = LiveSession(blockSeconds: 0.1, modelStore: store)
    var running = false, failure: String?, cooling = false, recoveries = 0
    session.onRunning = { running = true }
    session.onFailure = { failure = $0 }
    session.onRecovery = { cooling = $0; if $0 { recoveries += 1 } }
    try session.start(runtime: runtime, device: OutputDevice())
    let startup = Date()
    while !running && failure == nil && Date().timeIntervalSince(startup) < 5 { spin(0.01) }
    precondition(running && failure == nil, "Session failed to start")
    func write(_ value: Float, blocks: Int = 1) {
      let samples = [Float](repeating: value, count: 4800 * blocks)
      samples.withUnsafeBufferPointer { p in
        precondition(hm_ring_write(session.captureRing.pointer, p.baseAddress, UInt32(p.count)) == p.count)
      }
    }
    func drain() -> [Float] {
      let n = hm_ring_count(session.playbackRing.pointer)
      var result = [Float](repeating: 0, count: Int(n))
      result.withUnsafeMutableBufferPointer { _ = hm_ring_read(session.playbackRing.pointer, $0.baseAddress, n) }
      return result
    }
    write(0.125); spin(0.15); _ = drain() // Prime nonzero recurrent state.
    // A silent callback burst plus delayed worker used to stop immediately.
    engine.setDelay(0.3)
    write(0, blocks: 4)
    spin(0.15)
    precondition(cooling && failure == nil, "Silent queue burst stopped listening")
    engine.setDelay(0)
    let silentStart = Date()
    while Date().timeIntervalSince(silentStart) < 3.3 {
      write(0); spin(0.1); _ = drain()
      precondition(failure == nil, "Silence caused fatal overload")
    }
    precondition(!cooling && recoveries == 1, "Did not recover after 3 seconds")
    _ = drain()
    write(0.125); spin(0.15)
    let resumed = drain()
    precondition(resumed.count == 4800 && resumed.allSatisfy { $0 == 0.0625 },
                 "Stale output/state leaked through recovery")

    // Persistent audible overload: bypass during cooldown, then stop only when
    // the retry also falls behind. No real-time guard is disabled for this test.
    engine.setDelay(0.25)
    let overloadStart = Date()
    var sawBypass = false
    while failure == nil && Date().timeIntervalSince(overloadStart) < 6 {
      write(0.125); spin(0.1)
      let output = drain()
      if cooling && output.contains(0.125) { sawBypass = true }
    }
    precondition(sawBypass, "Cooldown did not play original PCM")
    precondition(failure != nil && Date().timeIntervalSince(overloadStart) >= 3,
                 "Persistent overload did not stop after cooldown/retry")
    var stopped = false
    session.afterShutdown { stopped = true }
    while !stopped { spin(0.01) }
    print("Actual session: silent burst survives, 3s bypass/retry, clean state reset, persistent audible overload stops; passed")
  }
}
