import AudioRing
import Foundation

final class AudioRing {
  let pointer: OpaquePointer
  init(capacity: UInt32) { pointer = hm_ring_create(capacity)! }
  deinit { hm_ring_destroy(pointer) }
}

/// A loaded engine is retained for the application lifetime. All cache access,
/// warmup and inference use one queue; only one session may own streaming state.
protocol ModelProcessing: AnyObject {
  var sampleCount: Int { get }
  func process(_ samples: [Float]) throws -> [Float]
  func reset()
}
#if canImport(MLX)
  extension MLXEngine: ModelProcessing {}
#endif

final class ModelStore {
  typealias Loader = (ModelRuntime, @escaping (Int) -> Void, @escaping () -> Bool) throws ->
    any ModelProcessing
  let queue = DispatchQueue(label: "HearME.Model", qos: .userInitiated)
  private let loader: Loader
  private let lock = NSLock()
  private var cancelled = false
  private var engines = [ModelRuntime: any ModelProcessing]()
  private var owner: UUID?
  private var ownedEngine: (any ModelProcessing)?

  init(
    loader: @escaping Loader = { runtime, progress, cancelled in
      #if canImport(MLX)
        progress(0)
        guard !cancelled() else { throw CancellationError() }
        let engine = try MLXEngine(assets: runtime.assets, frames: runtime.frames)
        guard !cancelled() else { throw CancellationError() }
        progress(100)
        return engine
      #else
        throw AudioFailure(message: "This build does not include the native GPU runtime.")
      #endif
    }
  ) { self.loader = loader }

  private var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }
  func cancelPreparation() {
    lock.lock()
    cancelled = true
    lock.unlock()
  }

  func prepare(
    runtime: ModelRuntime, progress: @escaping (Int) -> Void,
    completion: @escaping (Result<Void, Error>) -> Void
  ) {
    queue.async { [self] in
      do {
        _ = try prepared(runtime, progress: progress)
        completion(.success(()))
      } catch { completion(.failure(error)) }
    }
  }

  // Only called on queue. Concurrent prepare/start requests join the same
  // serialized load: the first fills the cache, later requests reuse it.
  private func prepared(_ runtime: ModelRuntime, progress: @escaping (Int) -> Void) throws
    -> any ModelProcessing
  {
    dispatchPrecondition(condition: .onQueue(queue))
    guard !isCancelled else { throw CancellationError() }
    if let cached = engines[runtime] {
      progress(100)
      return cached
    }
    guard owner == nil else {
      throw AudioFailure(message: "A listening session is already using the model.")
    }
    progress(0)
    let loaded = try loader(
      runtime, { progress($0 * 90 / 100) }, { [weak self] in self?.isCancelled ?? true })
    _ = try Self.measure(loaded, iterations: 9, cancelled: { self.isCancelled })
    guard !isCancelled else { throw CancellationError() }
    loaded.reset()
    engines[runtime] = loaded
    progress(100)
    return loaded
  }

  func acquire(runtime: ModelRuntime, owner token: UUID, progress: @escaping (Int) -> Void) throws
    -> any ModelProcessing
  {
    guard owner == nil else {
      throw AudioFailure(message: "A listening session is already using the model.")
    }
    let loaded = try prepared(runtime, progress: progress)
    loaded.reset()
    owner = token
    ownedEngine = loaded
    return loaded
  }
  func release(owner token: UUID) {
    dispatchPrecondition(condition: .onQueue(queue))
    guard owner == token else { return }
    ownedEngine?.reset()
    ownedEngine = nil
    owner = nil
  }

  static func measure(_ engine: any ModelProcessing, iterations: Int, cancelled: () -> Bool) throws
    -> Double
  {
    var random: UInt32 = 1
    let probe: [Float] = (0..<engine.sampleCount).map { index in
      random = random &* 1_664_525 &+ 1_013_904_223
      let noise = Float(random >> 8) / 16_777_216 - 0.5
      return noise * 0.06 + Float(sin(Double(index) * 2 * .pi * 180 / 48000)) * 0.06
    }
    var times = [Double]()
    for iteration in 0..<iterations {
      guard !cancelled() else { throw CancellationError() }
      let started = DispatchTime.now().uptimeNanoseconds
      let output = try engine.process(probe)
      guard output.count == engine.sampleCount, output.allSatisfy({ $0.isFinite }) else {
        throw AudioFailure(message: "Model warmup returned invalid audio.")
      }
      if iteration >= iterations - 3 {
        times.append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
      }
    }
    return times.reduce(0, +) / Double(times.count)
  }

  func shutdown(_ completion: @escaping () -> Void) {
    cancelPreparation()
    queue.async { [self] in
      ownedEngine = nil
      owner = nil
      engines.removeAll()
      completion()
    }
  }
}

/// Serialized native Core ML/Metal inference, entirely outside the audio callbacks.
final class ModelWorker {
  private let store: ModelStore
  private var queue: DispatchQueue { store.queue }
  private let owner = UUID()
  private var started = false
  private var ready = false
  private let lock = NSLock()
  private var pending = 0
  private var streamRevision: UInt64 = 0
  private var stopping = false
  private var engine: (any ModelProcessing)?
  private var previousEnhanced = true
  private var quietSamples = 0
  private var gated = false
  private var inputPCM = [Float]()
  private var outputPCM = [Float]()
  init(store: ModelStore = ModelStore()) { self.store = store }
  var onReady: (() -> Void)?
  var onPreparing: ((Int) -> Void)?
  var onMetrics: ((Double, Double) -> Void)?
  var onFailure: ((String) -> Void)?
  var onAudio: ((Data, Double) -> Void)?
  var pendingRequests: Int {
    lock.lock()
    defer { lock.unlock() }
    return pending
  }
  private var isStopped: Bool {
    lock.lock()
    defer { lock.unlock() }
    return stopping
  }

  func start(runtime: ModelRuntime, enforceRealTime: Bool = true) throws {
    lock.lock()
    guard !started, !stopping else {
      lock.unlock()
      throw AudioFailure(message: "This model worker has already started.")
    }
    started = true
    lock.unlock()
    queue.async { [self] in
      do {
        guard !isStopped else { return }
        let loaded = try store.acquire(
          runtime: runtime, owner: owner,
          progress: { [weak self] value in if self?.isStopped == false { self?.onPreparing?(value) }
          })
        engine = loaded
        // Recheck current throughput without reloading or repeating full warmup.
        let average = try ModelStore.measure(loaded, iterations: 3, cancelled: { self.isStopped })
        let budget = Double(loaded.sampleCount) / 48
        if enforceRealTime && average > budget * 0.98 {
          throw AudioFailure(
            message: String(
              format:
                "Enhancement needs %.0f ms to process %.0f ms of audio on this Mac. Choose a larger audio buffer or close other GPU work.",
              average, budget))
        }
        reset()
        guard !isStopped else { return }
        lock.lock()
        ready = true
        lock.unlock()
        onReady?()
      } catch {
        store.release(owner: owner)
        engine = nil
        if !isStopped {
          onFailure?("Voice enhancement could not start: \(error.localizedDescription)")
        }
      }
    }
  }

  private func reset() {
    engine?.reset()
    inputPCM.removeAll(keepingCapacity: true)
    outputPCM.removeAll(keepingCapacity: true)
  }

  /// Drop queued/in-flight results after a live discontinuity. The engine is
  /// reset on its serial queue before any subsequently submitted audio.
  func resetStream() {
    lock.lock()
    streamRevision &+= 1
    lock.unlock()
    queue.async { [self] in
      guard !isStopped else { return }
      reset()
      quietSamples = 0
      gated = false
    }
  }

  private func accepts(_ revision: UInt64) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return !stopping && streamRevision == revision
  }

  private func enhance(_ samples: [Float]) throws -> [Float] {
    guard let engine else { throw AudioFailure(message: "The model is not ready.") }
    inputPCM.append(contentsOf: samples)
    while inputPCM.count >= engine.sampleCount {
      let block = Array(inputPCM.prefix(engine.sampleCount))
      inputPCM.removeFirst(engine.sampleCount)
      outputPCM.append(contentsOf: try engine.process(block))
    }
    let available = min(samples.count, outputPCM.count)
    var result = [Float](repeating: 0, count: samples.count - available)
    result.append(contentsOf: outputPCM.prefix(available))
    outputPCM.removeFirst(available)
    return result
  }

  @discardableResult func submit(_ data: Data, enhanced: Bool, inputTime: Double) -> Bool {
    lock.lock()
    guard !stopping, ready, pending < 2 else {
      lock.unlock()
      return false
    }
    pending += 1
    let revision = streamRevision
    lock.unlock()
    queue.async { [self] in
      defer {
        lock.lock()
        pending -= 1
        lock.unlock()
      }
      guard accepts(revision) else { return }
      do {
        let start = DispatchTime.now().uptimeNanoseconds
        let samples = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        if enhanced != previousEnhanced {
          reset()
          previousEnhanced = enhanced
          quietSamples = 0
          gated = false
        }
        var output = samples
        if enhanced {
          let quiet = samples.allSatisfy { abs($0) < 1e-6 }
          quietSamples = quiet ? quietSamples + samples.count : 0
          if quietSamples >= 24000 {
            if !gated { reset() }
            gated = true
            output = [Float](repeating: 0, count: samples.count)
          } else {
            if gated {
              reset()
              gated = false
            }
            output = try enhance(samples)
          }
        }
        guard output.allSatisfy({ $0.isFinite }) else {
          throw AudioFailure(message: "Enhancement returned invalid audio.")
        }
        guard accepts(revision) else { return }
        let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        onAudio?(output.withUnsafeBytes { Data($0) }, inputTime)
        onMetrics?(milliseconds, milliseconds / max(Double(samples.count) / 48, 0.001))
      } catch { if accepts(revision) { onFailure?(error.localizedDescription) } }
    }
    return true
  }
  func stop() {
    lock.lock()
    stopping = true
    ready = false
    lock.unlock()
    queue.async { [self] in
      store.release(owner: owner)
      engine = nil
      inputPCM = []
      outputPCM = []
    }
  }
  func afterShutdown(_ completion: @escaping () -> Void) { queue.async(execute: completion) }
}
