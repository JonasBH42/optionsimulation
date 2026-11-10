"""Verify saved sessions can move between matching source and compiled runtimes."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(r'D:\optionsimulation')
sys.path.insert(0,str(ROOT/'src'))
from qqq_sim.config import configure_environment
from qqq_sim.serialization import save_json
from build_source_manifest import current


def main():
    configure_environment()
    build_id=current()[0]
    binary=ROOT/'dist/QQQSimulator.dist/QQQSimulator.exe'
    source=[str(ROOT/'.venv/Scripts/python.exe'),'-m','qqq_sim']
    output='runtime/results/cross-runtime'
    records=[]
    def run(name,prefix,args):
        env=dict(os.environ)
        if prefix==[str(binary)]:
            for key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','QT_PLUGIN_PATH'):
                env.pop(key,None)
            env['PATH']=r'C:\Windows\System32;C:\Windows'
        result=subprocess.run([*prefix,*args],cwd=ROOT,env=env,capture_output=True,
                              text=True,encoding='utf-8',errors='replace',timeout=120)
        records.append({'name':name,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        if result.returncode:
            raise RuntimeError(f'{name} failed: {result.stderr}')
        return json.loads(result.stdout)
    report={'status':'FAIL','build_id':build_id,'commands':records}
    try:
        doctor=run('compiled_identity',[str(binary)],['doctor'])
        if doctor['build_id']!=build_id:
            raise RuntimeError('Compiled source identity is stale')
        compiled_run=json.loads((ROOT/'runtime/results/standalone/windows/replay.json').read_text(encoding='utf-8'))
        # The exported replay record and CLI summary both retain the session ID.
        compiled_session=compiled_run['session_id']
        run('source_restores_compiled_session',source,['export',compiled_session,'--output',f'{output}/source-export-of-compiled-session.json'])
        created=run('source_creates_session',source,['--demo','replay','--at','2026-11-09T10:01:00-05:00',
                    '--minutes','1','--output',f'{output}/source-created-session.json'])
        run('compiled_restores_source_session',[str(binary)],['export',created['session_id'],
                    '--output',f'{output}/compiled-export-of-source-session.json'])
        report.update(status='PASS',compiled_to_source=True,source_to_compiled=True)
    except Exception as exc:
        report.update(error=type(exc).__name__,detail=str(exc))
    save_json(f'{output}/result.json',report)
    print(json.dumps({key:value for key,value in report.items() if key!='commands'},indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
