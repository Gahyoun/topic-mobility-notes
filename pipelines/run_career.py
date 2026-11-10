"""Rebuild the frozen career analysis stages and write a local provenance manifest."""
import argparse
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
INPUTS=('archive','appointments','pilot_reference','cohort','appointment_evidence','metadata_db')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--only-portfolio',action='store_true')
    p.add_argument('--figures',action='store_true')
    p.add_argument('--font',type=Path,nargs='+')
    args=p.parse_args()
    out=args.output.resolve()
    if out.is_relative_to(HERE.parent):
        raise SystemExit('Select a private output directory outside the source checkout.')
    cfg=json.loads(args.config.read_text())
    needed=INPUTS[3:] if args.only_portfolio else INPUTS
    paths={key:Path(cfg[key]).expanduser().resolve(strict=True) for key in needed}
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',LOKY_MAX_CPU_COUNT='4',PYTHONHASHSEED='0')
    executed=[]

    def run(script,*values):
        cmd=[sys.executable,str(HERE/script),*(str(v) for v in values)]
        subprocess.run(cmd,env=env,check=True)
        executed.append(script)

    def stage(code):
        folder=out/code
        folder.mkdir()
        return folder

    if not args.only_portfolio:
        s22=stage('S22')
        rebin=s22/'results.json'
        career=s22/'career.json'
        run('career/analyze.py',paths['archive'],paths['appointments'],paths['pilot_reference'],rebin)
        run('career/career.py',paths['cohort'],paths['appointment_evidence'],career)
        inputs=[paths['archive'],paths['appointments'],paths['pilot_reference'],rebin,
                paths['cohort'],paths['appointment_evidence'],career]
        previous=None
        for step,script in (('S23','aggregation'),('S24','window_width'),
                            ('S25','motion'),('S26','turns')):
            folder=stage(step)
            result=folder/'results.json'
            run(f'career/{script}.py',*inputs,*([previous] if previous else []),result)
            previous=result
            if args.figures:
                run(f'figures/{step.lower()}.py','--input',result,'--output',folder/'figure.png')
        if args.figures:
            run('figures/s22.py','--input',rebin,'--output',s22/'pilot.png')
            run('figures/s22.py','--career','--input',career,'--output',s22/'career.png')
    s27=stage('S27')
    run('career/portfolio.py',paths['cohort'],paths['appointment_evidence'],paths['metadata_db'],s27)
    run('career/uncertainty.py',s27/'portfolio.sqlite',s27/'results.json')
    s28=stage('S28')
    database=s27/'portfolio.sqlite'
    tensor_directory=s28/'tensors'
    run('career/audit_citation_cycles.py',paths['cohort'],database,s28/'cycle-audit.json')
    run('career/native_dag.py',paths['cohort'],database,s28/'dag-graph.json')
    run('career/native_flow.py',database,tensor_directory)
    run('career/faculty_tensors.py',database,tensor_directory)

    s29=out/'S29'
    run('career/appointment_dag.py',database,tensor_directory,s29)
    if args.figures:
        run('figures/s29.py',s29/'summary.json',s29/'figure.png',*(['--font',*args.font] if args.font else []))
        run('figures/stream.py',tensor_directory,s28/'annual-stream.html')
        run('figures/s27.py',s27/'results.json',s27/'figure.png',*(['--font',*args.font] if args.font else []))
    packages=(line.split('==',1)[0] for line in (HERE/'requirements.txt').read_text().splitlines())
    provenance={'python':sys.version.split()[0],'packages':{p:version(p) for p in packages},
                'inputs_sha256':{k:digest(v) for k,v in paths.items()},
                'executed_scripts':executed,'script_sha256':{str(f.relative_to(HERE)):digest(f) for f in sorted(HERE.rglob('*.py'))}}
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    main()
