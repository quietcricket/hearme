import AVFoundation
import AudioRing
import Foundation

final class LiveSession {
  let captureRing = AudioRing(capacity: 48000 * 3)
  let playbackRing = AudioRing(capacity: 48000 * 3)
  private let tap = SystemAudioTap()
  private let playback = AudioPlayback()
  private let worker: ModelWorker
  private let pumpQueue = DispatchQueue(label: "HearME.pump", qos: .userInitiated)
  private let routingQueue = DispatchQueue(label: "HearME.routing", qos: .userInitiated)
  private var timer: DispatchSourceTimer?
  private let lock = NSLock()
  private var enhanced = true
  private var stopped = false
  private var recovery = OverloadRecovery()
  private var lastAudible = false
  private var converter: AVAudioConverter?
  private var captureFormat: AVAudioFormat!
  private let processingFormat = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
  private var firstAudio = true
  private var missingBaseline: UInt64 = 0
  let blockSeconds: Double
  var capturedSeconds: Double {
    Double(hm_ring_total_written(captureRing.pointer)) / tap.sampleRate
  }
  func signalLevels() -> (Float, Float) {
    (hm_ring_take_peak(captureRing.pointer), hm_ring_take_peak(playbackRing.pointer))
  }
  var onRunning: (() -> Void)?
  var onConnecting: (() -> Void)?
  var onPreparing: ((Int) -> Void)?
  var onFailure: ((String) -> Void)?
  var onRecovery: ((Bool) -> Void)?
  var onRouteInvalidated: (() -> Void)?
  var onMetrics: ((Double, Double, Double, UInt64, Double) -> Void)?

  init(blockSeconds: Double, prerollSeconds: Double? = nil, modelStore: ModelStore = ModelStore()) {
    worker = ModelWorker(store: modelStore)
    self.blockSeconds = blockSeconds
    hm_ring_start_delay(playbackRing.pointer, prerollSeconds ?? (blockSeconds <= 0.1 ? 0.06 : 0.15))
  }

  func start(runtime: ModelRuntime, device: OutputDevice) throws {
    let outputRing = playbackRing
    playback.onConfigurationChange = { [weak self] in
      DispatchQueue.main.async { [weak self] in
        guard let self, !self.isStopped else { return }
        self.onRouteInvalidated?()
      }
    }
    worker.onPreparing = { [weak self] value in
      DispatchQueue.main.async { [weak self] in
        guard let self, !self.isStopped else { return }
        self.onPreparing?(value)
      }
    }
    worker.onReady = { [weak self] in
      DispatchQueue.main.async { [weak self] in
        guard let self, !self.isStopped else { return }
        self.onConnecting?()
      }
      self?.routingQueue.async { [weak self] in
        guard let self, !self.isStopped else { return }
        do {
          try self.tap.start(ring: self.captureRing.pointer, output: device)
          guard !self.isStopped else {
            self.tap.stop()
            return
          }
          self.captureFormat = AVAudioFormat(
            standardFormatWithSampleRate: self.tap.sampleRate, channels: 1)!
          if abs(self.tap.sampleRate - 48000) > 1 {
            self.converter = AVAudioConverter(from: self.captureFormat, to: self.processingFormat)
          }
          try self.playback.start(device: device, ring: outputRing.pointer)
          // Capture runs while HAL/AVAudioEngine connects the output.
          // Keep only audio arriving after both ends are ready.
          hm_ring_discard(self.captureRing.pointer)
          self.startPump()
          DispatchQueue.main.async { [weak self] in
            guard let self, !self.isStopped else { return }
            self.onRunning?()
          }
        } catch { self.fail(error.localizedDescription) }
      }
    }
    worker.onAudio = { [weak self] data, inputTime in
      self?.pumpQueue.async { [weak self] in
        guard let self, !self.isStopped, !self.recovery.isCooling else { return }
        self.writePlayback(data, inputTime: inputTime)
      }
    }
    worker.onMetrics = { [weak self] compute, rtf in
      self?.pumpQueue.async { [weak self] in
        guard let self, !self.isStopped, !self.recovery.isCooling else { return }
        self.reportMetrics(compute: compute, rtf: rtf)
      }
    }
    worker.onFailure = { [weak self] error in self?.fail(error) }
    try worker.start(runtime: runtime)
  }

  private var isStopped: Bool {
    lock.lock()
    defer { lock.unlock() }
    return stopped
  }
  func setEnhanced(_ value: Bool) {
    lock.lock()
    enhanced = value
    lock.unlock()
  }
  func setGain(_ value: Float) { hm_ring_gain(playbackRing.pointer, value) }

  private func startPump() {
    let timer = DispatchSource.makeTimerSource(queue: pumpQueue)
    timer.schedule(deadline: .now(), repeating: .milliseconds(10))
    timer.setEventHandler { [weak self] in self?.pump() }
    self.timer = timer
    timer.resume()
  }

  private func pump() {
    guard !isStopped else { return }
    lock.lock()
    let enabled = enhanced
    lock.unlock()
    let count = Int((tap.sampleRate * blockSeconds).rounded())
    guard hm_ring_count(captureRing.pointer) >= count else {
      // Some devices stop supplying callbacks while idle. Finish the cooldown
      // on wall time once drained, without treating absent audio as a timeout.
      if recovery.isCooling && worker.pendingRequests == 0 {
        let action = recovery.update(
          now: ProcessInfo.processInfo.systemUptime, audible: false, overloaded: false)
        if action == .retry { notifyRecovery(false) }
      }
      return
    }
    let inputTime = hm_ring_head_time(captureRing.pointer)
    var samples = [Float](repeating: 0, count: count)
    _ = samples.withUnsafeMutableBufferPointer {
      hm_ring_read(captureRing.pointer, $0.baseAddress, UInt32(count))
    }
    var data: Data
    if let converter {
      let input = AVAudioPCMBuffer(
        pcmFormat: captureFormat, frameCapacity: AVAudioFrameCount(count))!
      input.frameLength = AVAudioFrameCount(count)
      samples.withUnsafeBufferPointer {
        input.floatChannelData![0].update(from: $0.baseAddress!, count: count)
      }
      let capacity = AVAudioFrameCount(ceil(Double(count) * 48000 / tap.sampleRate) + 64)
      let output = AVAudioPCMBuffer(pcmFormat: processingFormat, frameCapacity: capacity)!
      var supplied = false
      var error: NSError?
      let status = converter.convert(to: output, error: &error) { _, state in
        if supplied {
          state.pointee = .noDataNow
          return nil
        }
        supplied = true
        state.pointee = .haveData
        return input
      }
      if status == .error {
        fail(error?.localizedDescription ?? "Audio resampling failed.")
        return
      }
      data = Data(bytes: output.floatChannelData![0], count: Int(output.frameLength) * 4)
    } else {
      data = samples.withUnsafeBytes { Data($0) }
    }
    let audible = data.withUnsafeBytes {
      // Ignore idle-device numerical noise below -80 dBFS for the failure
      // decision; this does not gate or alter the audio sent to the model.
      $0.bindMemory(to: Float.self).contains { abs($0) >= 1e-4 }
    }
    lastAudible = audible
    let overflow = hm_ring_dropped(captureRing.pointer) > 0
    // Do not repeatedly submit a backlog from a burst of silent callbacks.
    // Queue saturation is backpressure, not an immediate fatal model error.
    let overloaded = overflow || worker.pendingRequests >= 2
    let action = recovery.update(
      now: ProcessInfo.processInfo.systemUptime, audible: audible, overloaded: overloaded)
    if action == .stop {
      fail("Enhancement still could not keep up after a 3-second cooldown. Original system audio has been restored.")
      return
    }
    if action == .beginCooldown {
      beginCooldown()
    } else if action == .retry {
      notifyRecovery(false)
    }
    if overflow {
      hm_ring_discard(captureRing.pointer)
      converter?.reset()
    }
    if recovery.isCooling {
      writePlayback(data, inputTime: inputTime)
      reportMetrics(compute: 0, rtf: 0)
      return
    }
    if !worker.submit(data, enhanced: enabled, inputTime: inputTime - (enabled ? 0.0613125 : 0)) {
      // Stop can race a submit; never start recovery on a stopped session.
      guard !isStopped else { return }
      let action = recovery.update(
        now: ProcessInfo.processInfo.systemUptime, audible: audible, overloaded: true)
      if action == .stop {
        fail("Enhancement still could not keep up after a 3-second cooldown. Original system audio has been restored.")
      } else {
        if action == .beginCooldown { beginCooldown() }
        writePlayback(data, inputTime: inputTime)
        reportMetrics(compute: 0, rtf: 0)
      }
    }
  }

  private func beginCooldown() {
    worker.resetStream()
    // This is the capture ring's sole consumer. Discarding here is safe; the
    // playback ring must only be consumed by the audio render callback.
    hm_ring_discard(captureRing.pointer)
    converter?.reset()
    notifyRecovery(true)
  }

  private func notifyRecovery(_ cooling: Bool) {
    DispatchQueue.main.async { [weak self] in
      guard let self, !self.isStopped else { return }
      self.onRecovery?(cooling)
    }
  }

  private func writePlayback(_ data: Data, inputTime: Double) {
    let count = data.count / 4
    let written = data.withUnsafeBytes { raw in
      hm_ring_write_timed(
        playbackRing.pointer, raw.bindMemory(to: Float.self).baseAddress, UInt32(count), inputTime)
    }
    if firstAudio {
      missingBaseline = hm_ring_missing(playbackRing.pointer)
      firstAudio = false
    }
    if written < count {
      // Apply the same grace to a temporarily stalled output, including silence.
      let action = recovery.update(
        now: ProcessInfo.processInfo.systemUptime, audible: lastAudible, overloaded: true)
      if action == .stop {
        fail("Playback still could not keep up after a 3-second cooldown. Original system audio has been restored.")
      } else if action == .beginCooldown { beginCooldown() }
    }
  }

  private func reportMetrics(compute: Double, rtf: Double) {
    let queued = Double(hm_ring_count(captureRing.pointer)) / tap.sampleRate
      + Double(hm_ring_count(playbackRing.pointer)) / 48000
    onMetrics?(
      compute, rtf, queued, hm_ring_missing(playbackRing.pointer) - missingBaseline,
      hm_ring_latency(playbackRing.pointer))
  }

  private func fail(_ message: String) {
    DispatchQueue.main.async { [weak self] in
      guard let self, !self.isStopped else { return }
      self.stop()
      self.onFailure?(message)
    }
  }

  func stop() {
    lock.lock()
    let already = stopped
    stopped = true
    lock.unlock()
    guard !already else { return }
    // Releasing the tap restores original audio even if the model crashes.
    timer?.cancel()
    timer = nil
    pumpQueue.sync {}
    worker.stop()
    routingQueue.async { [self] in
      playback.stop()
      tap.stop()
    }
  }
  func afterShutdown(_ completion: @escaping () -> Void) {
    routingQueue.async { [self] in
      worker.afterShutdown { DispatchQueue.main.async(execute: completion) }
    }
  }
  deinit {
    timer?.cancel()
    tap.stop()
    playback.stop()
    worker.stop()
  }
}
