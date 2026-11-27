import Foundation

/// Uses monotonic time: clock changes cannot increase the listening allowance.
struct TrialAllowance {
  static let duration: TimeInterval = 999 * 60 * 60
  private(set) var used: TimeInterval
  private var lastTick: TimeInterval?

  init(used: TimeInterval) {
    self.used = used.isFinite ? min(Self.duration, max(0, used)) : Self.duration
  }
  var remaining: TimeInterval { max(0, Self.duration - used) }

  mutating func update(listening: Bool, paid: Bool, uptime: TimeInterval) {
    if let previous = lastTick {
      used = min(Self.duration, used + max(0, uptime - previous))
    }
    lastTick = listening && !paid && remaining > 0 ? uptime : nil
  }
}
