"""Verify the native transfer bundle, model resources and absence of Python."""
import argparse,json,plistlib,subprocess,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('app',type=Path);args=p.parse_args();app=args.app.resolve()
info=plistlib.loads((app/'Contents/Info.plist').read_bytes());resources=app/'Contents/Resources'
assert 'HearMEWorkspaceRoot' not in info
mlx=resources/'MLX'
assert mlx.is_dir()
assert not (resources/'Hybrid').exists()
assert not list(resources.glob('TF*.mlmodelc'))
assert list(resources.rglob('default.metallib'))
metadata=json.loads((mlx/'metadata.json').read_text())
assert metadata['architecture']['format_version']==1
weights=mlx/'weights.safetensors'
with weights.open('rb') as f:
 size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size))
header.pop("__metadata__",None)
assert len(header)==358
for value in header.values():
 assert value['dtype'] in ('F16','F32')
 start,end=value['data_offsets'];assert 0<=start<=end<=weights.stat().st_size-size-8
for key in metadata['aliases'].values():assert key in header
for name in ['Scan.metal','fir.f32','analysis.f32','synthesis.f32']:assert (mlx/name).is_file()
assert not list(app.rglob('*.py')) and not list(app.rglob('*.pyc'))
assert not (resources/'Runtime').exists() and not (resources/'Engine').exists()
arch=subprocess.check_output(['lipo','-archs',str(app/'Contents/MacOS/HearME')],text=True).strip();assert arch=='arm64'
subprocess.run(['codesign','--verify','--deep','--strict',str(app)],check=True)
print(json.dumps(dict(app=str(app),version=info['CFBundleShortVersionString'],runtime='MLX + Metal',python_embedded=False,signature_verified=True,architecture=arch),indent=2))
