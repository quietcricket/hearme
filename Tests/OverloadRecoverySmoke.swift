import Foundation

private func check(_ action: OverloadRecovery.Action, _ expected: OverloadRecovery.Action) {
  precondition(action == expected, "Expected \(expected), got \(action)")
}
@main struct OverloadRecoverySmoke {
  static func main() {
    var r = OverloadRecovery()
    check(r.update(now: 0, audible: true, overloaded: true), .beginCooldown)
    check(r.update(now: 2.999, audible: true, overloaded: true), .bypass)
    check(r.update(now: 3, audible: true, overloaded: false), .retry)
    check(r.update(now: 3.1, audible: true, overloaded: false), .process)
    check(r.update(now: 3.2, audible: true, overloaded: true), .stop)

    r = OverloadRecovery()
    check(r.update(now: 0, audible: true, overloaded: true), .beginCooldown)
    check(r.update(now: 3, audible: true, overloaded: true), .stop)

    r = OverloadRecovery()
    for t in stride(from: 0.0, through: 60.0, by: 0.1) {
      let result = r.update(now: t, audible: false, overloaded: true)
      precondition(result != .stop, "Silent overload stopped listening")
    }
    // Speech arriving after an idle stall still gets its own grace period.
    check(r.update(now: 63.1, audible: true, overloaded: true), .bypass)
    check(r.update(now: 66.2, audible: true, overloaded: false), .retry)
    check(r.update(now: 69.3, audible: true, overloaded: true), .beginCooldown)
    check(r.update(now: 72.3, audible: true, overloaded: false), .retry)
    check(r.update(now: 72.4, audible: false, overloaded: true), .beginCooldown)
    check(r.update(now: 75.5, audible: false, overloaded: true), .bypass)
    check(r.update(now: 78.6, audible: false, overloaded: false), .retry)
    check(r.update(now: 78.7, audible: true, overloaded: true), .beginCooldown)
    print("Cooldown boundary, recovery, persistent overload, healthy reset and silent-stall checks passed")
  }
}
