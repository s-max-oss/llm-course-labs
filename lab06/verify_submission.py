"""Hash actual staged Git blobs, compare local/archive bytes, scan new text."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MANIFEST = ROOT/'provenance/improvement_manifest.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write-manifest',action='store_true')
    args=parser.parse_args()
    changed=set(git('diff','--cached','--name-only','-z').decode('utf-8').split('\0'))
    records=[]
    for item in git('ls-files','--stage','-z','--','lab06').decode('utf-8').split('\0'):
        if not item:continue
        meta,path=item.split('\t',1);mode,oid,stage=meta.split()
        assert stage=='0' and re.fullmatch('[0-9a-f]{40,64}',oid)
        if path=='lab06/provenance/improvement_manifest.json':continue
        records.append((path,oid))
    process=subprocess.Popen(['git','cat-file','--batch'],cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    hashes={};archive=REPO.parent/'lab06_rsi'; checked_text=0
    secret=re.compile(rb'(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
    text_suffixes={'.py','.json','.md','.slurm','.ps1','.log','.txt','.toml'}
    try:
        for path,oid in records:
            process.stdin.write((oid+'\n').encode());process.stdin.flush()
            header=process.stdout.readline().decode().split()
            assert header[0]==oid and header[1]=='blob'
            remaining=int(header[2]);digest=hashlib.sha256();data=[]
            scan=path in changed and Path(path).suffix in text_suffixes
            while remaining:
                chunk=process.stdout.read(min(65536,remaining));assert chunk
                digest.update(chunk);remaining-=len(chunk)
                if scan:data.append(chunk)
            assert process.stdout.read(1)==b'\n'
            relative=path.removeprefix('lab06/');value=digest.hexdigest()
            assert hashlib.sha256((REPO/path).read_bytes()).hexdigest()==value, f'Git conversion changed bytes: {path}'
            hashes[relative]=value
            if scan:
                assert not secret.search(b''.join(data)), f'Credential pattern found in staged file: {path}'
                checked_text+=1
            if not args.write_manifest:
                assert hashlib.sha256((archive/relative).read_bytes()).hexdigest()==value, f'Archive differs: {path}'
    finally:
        process.stdin.close();process.wait()
    assert process.returncode==0
    if args.write_manifest:
        payload={'algorithm':'SHA-256','excludes_self':'provenance/improvement_manifest.json',
                 'historical_v1_manifest':'provenance/improvement_manifest_v1.json (compare commit b50b220)',
                 'files':dict(sorted(hashes.items()))}
        MANIFEST.write_bytes((json.dumps(payload,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    else:
        assert json.loads(MANIFEST.read_text(encoding='utf-8'))['files']==hashes
        assert git('show',':lab06/provenance/improvement_manifest.json')==MANIFEST.read_bytes()
        assert (archive/'provenance/improvement_manifest.json').read_bytes()==MANIFEST.read_bytes()
    print(json.dumps({'staged_blobs_sha256_verified':len(hashes),'new_text_files_scanned':checked_text,
                      'archive_verified':not args.write_manifest,'manifest_written':args.write_manifest}))


if __name__=='__main__':main()
