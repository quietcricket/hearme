#!/usr/bin/env python3
"""Build a native MLX Apple Silicon transfer package; no Python runtime."""
from pathlib import Path
import hashlib,json,plistlib,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
VERSION='0.5.2'
BUILD=ROOT/'build';STAGE=BUILD/f'HearME-{VERSION}-Mac';APP=STAGE/'HearME.app'
def run(*args):subprocess.run([str(a) for a in args],check=True)
run(ROOT/'build-app.sh','--direct')
if STAGE.exists():shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
run('ditto',BUILD/'HearME-Direct/HearME.app',APP)
resources=APP/'Contents/Resources'
assert not list(resources.glob('TF*.mlmodelc')),'Old Core ML stages must not ship'
assert (resources/'MLX/weights.safetensors').is_file(),'Native GPU weights missing'
assert (resources/'MLX/metadata.json').is_file()
assert list(resources.rglob('default.metallib')),'Native GPU kernels missing'
assert not (resources/'Runtime').exists() and not (resources/'Engine').exists()
assert not list(APP.rglob('*.py')),'Python files must not ship in the native app'
licenses=resources/'Licenses';licenses.mkdir(exist_ok=True)
shutil.copy2(ROOT/'models/TF_Restormer/LICENSE',licenses/'TF-Restormer.txt')
shutil.copy2(ROOT/'scripts/Model/vendor/mamba/LICENSE',licenses/'Mamba.txt')
for source,name in [
 ('mlx-swift/LICENSE','MLX-Swift.txt'),('mlx-swift/Source/Cmlx/mlx/LICENSE','MLX.txt'),
 ('mlx-swift/Source/Cmlx/mlx-c/LICENSE','MLX-C.txt'),('mlx-swift/Source/Cmlx/metal-cpp/LICENSE.txt','Metal-cpp.txt'),
 ('mlx-swift/Source/Cmlx/json/LICENSE.MIT','JSON.txt'),('mlx-swift/Source/Cmlx/fmt/LICENSE','fmt.txt'),
 ('swift-numerics/LICENSE.txt','Swift-Numerics.txt')]:
 shutil.copy2(ROOT/'.build-xcode/SourcePackages/checkouts'/source,licenses/name)
info=plistlib.loads((APP/'Contents/Info.plist').read_bytes())
assert 'HearMEWorkspaceRoot' not in info
run('codesign','--force','--sign','-','--identifier',info['CFBundleIdentifier'],APP)
run('codesign','--verify','--deep','--strict',APP)
shutil.copy2(ROOT/'docs/transfer.txt',STAGE/'READ ME FIRST.txt')
(STAGE/'BUILD-INFO.json').write_text(json.dumps(dict(version=VERSION,architecture='arm64',runtime='Native Swift + MLX + Metal',minimum_macos='15.0',signing='ad-hoc; not notarized'),indent=2)+'\n')
archive=BUILD/f'HearME-{VERSION}-Apple-Silicon.zip'
if archive.exists():archive.unlink()
run('ditto','-c','-k','--keepParent',STAGE,archive)
def checksum(path):
 value=hashlib.sha256(path.read_bytes()).hexdigest()
 Path(str(path)+'.sha256').write_text(f'{value}  {path.name}\n')
 return value
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=checksum(archive)),indent=2),flush=True)
if '--dmg' in sys.argv:
 image=BUILD/f'HearME-{VERSION}-Apple-Silicon.dmg'
 run('hdiutil','create','-volname','HearME','-srcfolder',STAGE,'-format','UDZO','-ov',image)
 print(json.dumps(dict(disk_image=str(image),bytes=image.stat().st_size,sha256=checksum(image)),indent=2),flush=True)
