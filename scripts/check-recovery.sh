#!/bin/zsh
set -euo pipefail
task_mac="${0:A:h:h}"
task_out="$task_mac/build/recovery-check"
mkdir -p "$task_out"
swiftc "$task_mac/Sources/HearME/Audio/OverloadRecovery.swift" \
  "$task_mac/Tests/OverloadRecoverySmoke.swift" -o "$task_out/policy"
"$task_out/policy"
clang -c "$task_mac/Sources/AudioRing/AudioRing.c" \
  -I "$task_mac/Sources/AudioRing/include" -o "$task_out/ring.o"
swiftc -swift-version 5 -I "$task_mac/Sources/AudioRing/include" \
  "$task_mac/Sources/HearME/Enhancement/ModelRuntime.swift" \
  "$task_mac/Sources/HearME/Enhancement/ModelWorker.swift" \
  "$task_mac/Sources/HearME/Audio/OverloadRecovery.swift" \
  "$task_mac/Sources/HearME/Audio/LiveSession.swift" \
  "$task_mac/Tests/LiveRecoverySmoke.swift" "$task_out/ring.o" \
  -framework AVFoundation -framework CoreAudio -o "$task_out/session"
"$task_out/session"
