#!/usr/bin/env python3
"""Build, Developer ID sign, notarize, and package the direct edition."""
import argparse
import hashlib
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', required=True, help='Developer ID Application signing identity')
    parser.add_argument('--keychain-profile', required=True, help='notarytool credentials profile')
    parser.add_argument('--bundle-id', required=True, help='stable reverse-DNS bundle identifier')
    args = parser.parse_args()
    if not args.identity.startswith('Developer ID Application:'):
        parser.error('--identity must be a Developer ID Application identity')
    if '.' not in args.bundle_id or args.bundle_id.startswith('local.'):
        parser.error('--bundle-id must be your stable reverse-DNS identifier')
    identities = subprocess.check_output(
        ['security', 'find-identity', '-v', '-p', 'codesigning'], text=True)
    if f'"{args.identity}"' not in identities:
        parser.error('Developer ID Application identity was not found in the keychain')

    env = dict(os.environ, HEARME_DIRECT_BUNDLE_ID=args.bundle_id)
    run(ROOT / 'build-app.sh', '--direct', env=env)
    BUILD.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hearme-notary-', dir=BUILD) as temporary:
        work = Path(temporary)
        app = work / 'HearME.app'
        run('ditto', BUILD / 'HearME-Direct/HearME.app', app)
        resources = app / 'Contents/Resources'
        licenses = resources / 'Licenses'
        licenses.mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'models/TF_Restormer/LICENSE',
                     licenses / 'TF-Restormer.txt')
        shutil.copy2(ROOT / 'scripts/Model/vendor/mamba/LICENSE', licenses / 'Mamba.txt')
        for source, name in [
            ('mlx-swift/LICENSE', 'MLX-Swift.txt'),
            ('mlx-swift/Source/Cmlx/mlx/LICENSE', 'MLX.txt'),
            ('mlx-swift/Source/Cmlx/mlx-c/LICENSE', 'MLX-C.txt'),
            ('mlx-swift/Source/Cmlx/metal-cpp/LICENSE.txt', 'Metal-cpp.txt'),
            ('mlx-swift/Source/Cmlx/json/LICENSE.MIT', 'JSON.txt'),
            ('mlx-swift/Source/Cmlx/fmt/LICENSE', 'fmt.txt'),
            ('swift-numerics/LICENSE.txt', 'Swift-Numerics.txt'),
        ]:
            shutil.copy2(ROOT / '.build-xcode/SourcePackages/checkouts' / source, licenses / name)
        info = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
        assert info['CFBundleIdentifier'] == args.bundle_id
        assert (resources / 'MLX/weights.safetensors').is_file()
        assert not list(app.rglob('*.py'))
        run('codesign', '--force', '--sign', args.identity, '--options', 'runtime',
            '--timestamp', '--identifier', args.bundle_id, app)
        run('codesign', '--verify', '--deep', '--strict', '--verbose=2', app)
        submission = work / 'submission.zip'
        run('ditto', '-c', '-k', '--keepParent', app, submission)
        run('xcrun', 'notarytool', 'submit', submission, '--keychain-profile',
            args.keychain_profile, '--wait')
        run('xcrun', 'stapler', 'staple', app)
        run('xcrun', 'stapler', 'validate', app)
        run('codesign', '--verify', '--deep', '--strict', app)
        version = info['CFBundleShortVersionString']
        output = BUILD / f'HearME-{version}-Direct-Notarized.zip'
        if output.exists():
            raise FileExistsError(f'Refusing to replace {output}; move or rename it first')
        run('ditto', '-c', '-k', '--keepParent', app, output)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        Path(str(output) + '.sha256').write_text(f'{digest}  {output.name}\n')
        print(f'Notarized archive: {output}\nSHA-256: {digest}')


if __name__ == '__main__':
    main()
