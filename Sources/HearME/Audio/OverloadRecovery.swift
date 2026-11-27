import Foundation

/// Pump-queue confined. Silence cannot turn a scheduling hiccup into a fatal
/// overload. After bypassing for three seconds, allow a three-second trial;
/// another overload during audible input in that trial is persistent failure.
struct OverloadRecovery {
  enum Action: Equatable { case process, bypass, beginCooldown, retry, stop }
  private var cooldownUntil: Double?
  private var trialUntil: Double?
  private var audibleFailure = false
  var isCooling: Bool { cooldownUntil != nil }

  mutating func update(now: Double, audible: Bool, overloaded: Bool) -> Action {
    if let until = cooldownUntil {
      guard now >= until else { return .bypass }
      if overloaded {
        if audible && audibleFailure { return .stop }
        // An idle/stalled worker cannot stop listening on silence. Wait for it
        // to drain, or give newly audible input its own recovery interval.
        cooldownUntil = now + 3
        audibleFailure = audible
        return .bypass
      }
      cooldownUntil = nil
      trialUntil = audibleFailure ? now + 3 : nil
      return .retry
    }
    if let until = trialUntil, now >= until { trialUntil = nil }
    guard overloaded else { return .process }
    if audible && trialUntil != nil { return .stop }
    cooldownUntil = now + 3
    audibleFailure = audible
    return .beginCooldown
  }
}
