# HearME

HearME enhances the voices you hear in Zoom, Teams, Google Meet, and other Mac
apps. It restores incoming speech on-device using the streaming TF-Restormer
model and follows the audio output you select in macOS. HearME captures system
playback with your permission and leaves your microphone alone.

The app runs on Apple Silicon with macOS 15 or later. The direct build offers
unrestricted listening. The Mac App Store edition uses StoreKit for a listening trial
and paid unlocks.

## Build from source

You’ll need Xcode with Swift 6.3 or later and the Metal toolchain. Converting the
model with the pinned Python dependencies also requires Python 3.11 and macOS
26 or later. Run the commands below from the repository root.

### Set up the Python environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
a project-local virtual environment and install the dependencies used by the
model and utility scripts:

```sh
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Activate `.venv` in each new shell before running Python scripts. Run
`deactivate` when finished. The environment is local to your checkout and is
not committed to Git or included in the app.

### Prepare the model

HearME includes the [converter](scripts/Model/export_mlx.py). Download
the upstream source and checkpoint separately. Git ignores everything under
`models/`.

1. Get the TF-Restormer source at the tested revision:

   ```sh
   mkdir -p models
   git clone https://github.com/dmlguq456/TF_Restormer.git models/TF_Restormer
   git -C models/TF_Restormer checkout eaefe1cb3811c13f324eee34a81564cff8b7da7e
   ```

2. Download the pinned [TF-Restormer streaming checkpoint](https://huggingface.co/shinuh/tf-restormer-streaming):

   ```sh
   weights=models/weights
   mkdir -p "$weights"
   model_url=https://huggingface.co/shinuh/tf-restormer-streaming/resolve/e00f89541038e34e1d535f83d6b9d02fea9a97db
   curl -fL "$model_url/config.yaml" -o "$weights/config.yaml"
   curl -fL "$model_url/model.pt" -o "$weights/model.pt"
   ```

The [Mamba reference files](scripts/Model/vendor/mamba/) are included for
conversion. Downloading the full Mamba repository is unnecessary.

<details>
<summary>SHA-256 checksums for the tested inputs</summary>

| File | SHA-256 |
| --- | --- |
| `config.yaml` | `a968a229a0d7ab45596636d7b9cc7226a1444c593d46482bedd4919a984f7f82` |
| `model.pt` | `632e24d74513a96bccd93eea026259c164bb3b9d42f7fd2204c6f2ae3e16e520` |

</details>

```sh
python scripts/Model/export_mlx.py
```

The converter writes the app’s model assets to `models/MLX/`.

### Build and run

```sh
xcodebuild -downloadComponent MetalToolchain
./build-app.sh --direct
```

Open `build/HearME-Direct/HearME.app`, grant system audio access when
prompted, and turn HearME on. Use `--app-store` to build the sandboxed edition
locally. Both local builds use ad-hoc signing. See the [transfer instructions](docs/transfer.txt) for installing a
local test build.

## Contribute

For audio issues, include your Mac model, macOS version, audio output, buffer
setting, and steps to reproduce. Use a synthetic or shareable clip if a
recording is needed. See [AGENTS.md](AGENTS.md) for development checks.
