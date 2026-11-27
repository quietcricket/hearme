#!/bin/zsh
set -euo pipefail
task_root="${0:A:h}"
task_distribution="${1:---direct}"
case "$task_distribution" in
  --direct)
    task_variant="Direct"
    task_bundle_id="${HEARME_DIRECT_BUNDLE_ID:-local.hearme.direct}"
    task_settings=("ENABLE_APP_SANDBOX=NO" "ENABLE_HARDENED_RUNTIME=NO")
    ;;
  --app-store)
    task_variant="AppStore"
    task_bundle_id="${HEARME_APP_STORE_BUNDLE_ID:-local.hearme.app}"
    task_settings=("SWIFT_ACTIVE_COMPILATION_CONDITIONS=" "ENABLE_APP_SANDBOX=YES" "ENABLE_HARDENED_RUNTIME=YES")
    ;;
  *) print -u2 "Usage: $0 [--direct | --app-store]"; exit 2 ;;
esac
if (( $# > 1 )); then print -u2 "Usage: $0 [--direct | --app-store]"; exit 2; fi
task_assets="$task_root/models/MLX"
for task_file in weights.safetensors metadata.json Scan.metal fir.f32 analysis.f32 synthesis.f32; do
  if [[ ! -f "$task_assets/$task_file" ]]; then
    print -u2 "Missing model asset: $task_assets/$task_file"
    print -u2 "Follow the model setup in README.md before building."
    exit 1
  fi
done
xcodebuild -project "$task_root/HearME.xcodeproj" \
  -scheme HearME -configuration Release \
  -destination "platform=macOS,arch=arm64" ARCHS=arm64 ONLY_ACTIVE_ARCH=YES \
  "PRODUCT_BUNDLE_IDENTIFIER=$task_bundle_id" "${task_settings[@]}" \
  -derivedDataPath "$task_root/.build-xcode" build
task_app="$task_root/build/HearME-$task_variant/HearME.app"
mkdir -p "${task_app:h}"
# Replace only the generated development app, avoiding stale bundle resources.
if [[ -d "$task_app" ]]; then rm -rf "$task_app"; fi
ditto "$task_root/.build-xcode/Build/Products/Release/HearME.app" "$task_app"
codesign --verify --deep --strict "$task_app"
print "Built: $task_app"
