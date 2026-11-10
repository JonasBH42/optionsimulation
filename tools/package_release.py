"""Assemble notices and evidence after verified Windows standalone smoke runs."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT=Path(r'D:\optionsimulation')
sys.path.insert(0,str(ROOT/'src'))
from qqq_sim.config import writable_path
from qqq_sim.serialization import save_json
from build_source_manifest import current


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    destination=writable_path('dist/QQQSimulator.dist')
    build_id=current()[0]
    executable_sha256=sha(destination/'QQQSimulator.exe')
    for platform in ('offscreen','windows'):
        report=json.loads((ROOT/f'runtime/results/windows-standalone-smoke-{platform}.json').read_text(encoding='utf-8'))
        if report.get('status')!='PASS' or report.get('actual_build_id')!=build_id:
            raise RuntimeError(f'Current standalone smoke evidence missing: {platform}')
        if report.get('executable_sha256')!=executable_sha256 or not report.get('executable_unchanged_during_checks'):
            raise RuntimeError(f'Standalone binary differs from tested executable: {platform}')
    release=json.loads((ROOT/'runtime/results/release-manifest.json').read_text(encoding='utf-8'))
    status=json.loads((ROOT/'docs/release-status.json').read_text(encoding='utf-8'))
    if release['tests'].get('status')!='PASS':raise RuntimeError('Passing full test report required')
    if release['tests'].get('artifact_sha256')!=sha(ROOT/'runtime/results/tests.xml'):
        raise RuntimeError('Test report changed after evidence collection')
    if release.get('build_id')!=build_id:raise RuntimeError('Release evidence belongs to a different build')
    native=json.loads((ROOT/'runtime/results/clean-account/final-same-account/result.json').read_text(encoding='utf-8'))
    if (native.get('workflows_passed') is not True or native.get('actual_build_id')!=build_id
            or native.get('executable_sha256')!=executable_sha256
            or native.get('build_manifest_sha256')!=sha(ROOT/'runtime/results/release-manifest.json')):
        raise RuntimeError('Native verifier evidence does not match this release manifest and executable')
    cross_runtime=json.loads((ROOT/'runtime/results/cross-runtime/result.json').read_text(encoding='utf-8'))
    visual=json.loads((ROOT/'runtime/results/visual-review.json').read_text(encoding='utf-8'))
    if cross_runtime.get('status')!='PASS' or cross_runtime.get('build_id')!=build_id:
        raise RuntimeError('Matching cross-runtime session evidence required')
    if visual.get('status')!='PASS_SCOPED_VISUAL_AND_GEOMETRY_REVIEW' or visual.get('build_id')!=build_id:
        raise RuntimeError('Matching desktop visual review required')
    pack_folders=('runtime/model_packs/validation_fixture_v2',
                  'runtime/model_packs/research_frozen_v6/55880437bd1f')
    for folder in pack_folders:
        if not (ROOT/folder/'manifest.json').is_file():
            raise RuntimeError(f'Required unqualified pack is missing: {folder}')
    dependency_sources=json.loads((ROOT/'packaging/third_party_sources/manifest.json').read_text(encoding='utf-8'))
    for record in dependency_sources['records']:
        path=writable_path(record['path'])
        if not path.is_file() or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
            raise RuntimeError(f'Dependency source/license integrity failed: {record["path"]}')
    files={
        'README.md':'README.md',
        'config.example.json':'config.example.json',
        'packaging/THIRD_PARTY_NOTICES.txt':'THIRD_PARTY_NOTICES.txt',
        'packaging/dependency-notices.json':'dependency-notices.json',
        'packaging/LICENSE_REVIEW.md':'LICENSE_REVIEW.md',
        'packaging/verify-clean-account.ps1':'verify-clean-account.ps1',
        'requirements.lock':'requirements.lock',
        'runtime/results/release-manifest.json':'evidence/release-manifest.json',
        'runtime/results/build-source-manifest.json':'evidence/build-source-manifest.json',
        'runtime/results/tests.xml':'evidence/tests.xml',
        'runtime/results/windows-standalone-smoke-offscreen.json':'evidence/windows-smoke-offscreen.json',
        'runtime/results/windows-standalone-smoke-windows.json':'evidence/windows-smoke-windows.json',
        'docs/release-status.json':'evidence/release-status.json',
        'docs/runtime-identity.md':'runtime-identity.md',
        'docs/startup-checks.md':'startup-checks.md',
        'docs/clean-account-verification.md':'clean-account-verification.md',
    }
    numerical_evidence=(
        'runtime/risk_validation/final_risk_status.json',
        'runtime/risk_validation/nontrivial_path_domain_v5_reference.json',
        'runtime/risk_validation/nontrivial_path_domain_v5_job.json',
        'runtime/risk_validation/reference_v5_memory.jsonl',
        'runtime/risk_validation/production_frozen_v6_config.json',
        'runtime/risk_validation/research_frozen_v6_config.json',
        'runtime/builds/production_frozen_v6/build_report.json',
        'runtime/builds/research_frozen_v6/build_report.json',
        'runtime/results/query-benchmark.json',
        'runtime/results/validation-fixture-report.json',
        'runtime/results/cross-runtime/result.json',
        'runtime/results/clean-account/final-same-account/result.json',
        'runtime/results/compiled-fixture-build.json',
        'runtime/results/compiled-fixture-build-config.json',
        'runtime/results/windows-build-final.log',
        'runtime/results/visual-review.json',
    )
    for source in numerical_evidence:
        files[source]=source
    for path in sorted((ROOT/'docs').rglob('*')):
        if path.is_file():
            relative=path.relative_to(ROOT).as_posix()
            files[relative]=relative
    for platform in ('windows','offscreen'):
        for path in sorted((ROOT/f'runtime/results/standalone/{platform}').glob('desktop-*/*')):
            if path.is_file() and path.suffix in ('.png','.json'):
                relative=path.relative_to(ROOT).as_posix()
                files[relative]=relative
    for path in sorted((ROOT/'runtime/results/clean-account/final-same-account').glob('*')):
        candidates=path.glob('*') if path.is_dir() else [path]
        for item in candidates:
            if item.is_file() and item.suffix in ('.png','.json','.txt'):
                relative=item.relative_to(ROOT).as_posix()
                files[relative]=relative
    for name in ('windows_smoke.py','cross_runtime_smoke.py','collect_evidence.py','package_release.py'):
        files[f'tools/{name}']=f'evidence/verification-tools/{name}'
    for source,target in files.items():
        output=writable_path(destination/target,create_parent=True)
        shutil.copy2(ROOT/source,output)
    for folder in ('licenses','third_party_sources'):
        source=ROOT/'packaging'/folder
        if not source.is_dir():raise RuntimeError(f'Missing dependency material: {source}')
        for item in sorted(source.rglob('*')):
            if item.is_file():
                output=writable_path(destination/folder/item.relative_to(source),create_parent=True)
                shutil.copy2(item,output)
    for folder in pack_folders:
        for item in sorted((ROOT/folder).rglob('*')):
            if item.is_file():
                output=writable_path(destination/item.relative_to(ROOT),create_parent=True)
                shutil.copy2(item,output)
    (destination/'PORTABLE.md').write_text(
        '# Portable application handoff\n\n'
        'Keep this entire folder at `D:\\optionsimulation\\dist\\QQQSimulator.dist`. '
        'The application enforces `D:\\optionsimulation` as its only writable project root. '
        'External QQQ archives are read only.\n\n'
        'Run `QQQSimulator.exe` for the desktop, or `QQQSimulator.exe --demo` for synthetic data. '
        'The executable includes Python and its runtime dependencies. '
        'Development and offline Torch training use the project `.venv`; see the project README.\n\n'
        '**Production probability qualification is incomplete.** '
        'Both included model packs are explicitly unqualified and must abstain from supported P50 claims. '
        'See `docs/release-status.json`, `docs/risk_build_and_validation.md` and '
        '`runtime/risk_validation/final_risk_status.json` for measured outcomes and remaining gates. '
        'The tests on the build account are not a separate clean Windows account certification.\n\n'
        'The `docs/` tree and selected numerical/screenshot evidence are included using their original relative paths. '
        'Absolute paths within captured reports identify the original project run. '
        'Large resumable build caches and the development environment remain in the project and are not duplicated here.\n\n'
        '`artifact-manifest.json` inventories every packaged file with SHA-256. '
        '`LICENSE_REVIEW.md`, `THIRD_PARTY_NOTICES.txt`, `licenses/`, and `third_party_sources/` '
        'contain the dependency notices, replacement guidance, and matching Qt source archives.\n',encoding='utf-8')
    records=[]
    for path in sorted(destination.rglob('*')):
        if path.is_file() and path.name not in ('artifact-manifest.json','artifact-manifest.sha256'):
            records.append({'path':path.relative_to(destination).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'schema':'QQQ_PORTABLE_ARTIFACT_V1','build_id':build_id,'files':records,
              'bytes':sum(record['bytes'] for record in records),'release_status':'evidence/release-status.json',
              'production_probability_qualified':status.get('production_probability_qualified',False),
              'separate_clean_windows_account_tested':release.get('windows_clean_account')=='PASS'}
    output=save_json(destination/'artifact-manifest.json',manifest)
    (destination/'artifact-manifest.sha256').write_text(sha(output)+'\n',encoding='ascii')
    print(json.dumps({'build_id':build_id,'files':len(records),'bytes':manifest['bytes'],'path':str(destination)}))


if __name__=='__main__':main()
