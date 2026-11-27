# HearME contributor notes

- Keep audio processing receive-side and on-device. Use only the official
  streaming TF-Restormer model and keep Python out of the app bundle.
- Preserve streaming state between blocks and reset it between sessions.
  Preserve ordered audio cleanup on Stop, Quit, output changes, and errors.
- Share audio and enhancement code across the direct and App Store editions.
  The direct edition (`HEARME_DIRECT`) is the default in Xcode builds.
  Preserve App Sandbox and StoreKit behavior in the App Store build.
- Use the [README](README.md#build-from-source) for setup. Run
  `zsh scripts/check-recovery.sh` for audio recovery changes and the
  shared HearME Xcode scheme’s tests for audio ring changes. StoreKit tests
  (`PurchaseTests`) are skipped in the scheme while direct is the default;
  re-enable them with `SWIFT_ACTIVE_COMPILATION_CONDITIONS=` for App Store work.
  Use `scripts/check_package.py` to validate app bundles.
- Distinguish model processing time, tap-to-render delay, and hardware latency
  when reporting performance.
- Keep recordings, checkpoints, generated model assets, signing credentials,
  and build artifacts out of Git. Preserve upstream notices in distributions.
  Use `scripts/notarize-direct.py --help` for direct release signing.
