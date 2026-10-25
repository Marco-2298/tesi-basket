"""Inventario SHA256 e archivio reversibile delle sole cartelle identiche.

Nessuna dipendenza esterna, nessuna cancellazione. --organize sposta soltanto
una cartella se l'intero albero di file coincide con un'altra più recente.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_csv(path, rows, fields):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--organize', action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    outputs=(root/'outputs').resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report=root/'reports'/('output_audit_'+stamp)
    report.mkdir(parents=True,exist_ok=False)
    files=sorted(p for p in outputs.rglob('*') if p.is_file())
    inventory=[]; hashes={}; groups=collections.defaultdict(list)
    for p in files:
        sha=digest(p); hashes[p]=sha; groups[sha].append(p)
        row=dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha,rows='',columns='',error='')
        if p.suffix.lower()=='.csv':
            try:
                with p.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.DictReader(f)
                    row['columns']=' | '.join(reader.fieldnames or [])
                    row['rows']=sum(1 for _ in reader)
            except (UnicodeError,csv.Error) as e: row['error']=str(e)
        inventory.append(row)
    write_csv(report/'inventory_before.csv',inventory,['path','bytes','sha256','rows','columns','error'])
    duplicates=[dict(sha256=sha,path=str(p.relative_to(root)),bytes=p.stat().st_size,copies=len(ps))
                for sha,ps in groups.items() if len(ps)>1 for p in ps]
    write_csv(report/'duplicate_files.csv',duplicates,['sha256','path','bytes','copies'])
    signatures=collections.defaultdict(list)
    for folder in sorted(outputs.glob('tracking_diagnostics_*')):
        if folder.is_dir():
            sig=tuple((str(p.relative_to(folder)),hashes[p]) for p in sorted(folder.rglob('*')) if p.is_file())
            if sig: signatures[sig].append(folder)
    moves=[]
    for folders in signatures.values():
        if len(folders)<2: continue
        keeper=folders[-1]
        for folder in folders[:-1]:
            target=outputs/'archive_exact_duplicates'/folder.name
            if not folder.resolve().is_relative_to(outputs) or not target.resolve().is_relative_to(outputs):
                raise ValueError('Percorso fuori dalla cartella output')
            if target.exists(): raise FileExistsError(target)
            moves.append(dict(source=str(folder.relative_to(root)),destination=str(target.relative_to(root)),
                              identical_to=str(keeper.relative_to(root)),applied=False))
    (report/'move_plan.json').write_text(json.dumps(moves,indent=2),encoding='utf-8')
    if args.organize:
        for move in moves:
            source=root/move['source']; destination=root/move['destination']
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(destination))
            move['applied']=True
            # Journal dopo ogni movimento per ripristino anche in caso di interruzione.
            (report/'moves.json').write_text(json.dumps(moves,indent=2),encoding='utf-8')
    after=[]
    for row in inventory:
        p=root/row['path']
        for move in moves:
            source=root/move['source']
            if move['applied'] and p.is_relative_to(source):
                p=root/move['destination']/p.relative_to(source)
                break
        if not p.is_file() or digest(p)!=row['sha256']:
            raise RuntimeError(f'Verifica conservazione fallita: {p}')
        after.append(dict(original_path=row['path'],current_path=str(p.relative_to(root)),sha256=row['sha256']))
    write_csv(report/'verified_paths.csv',after,['original_path','current_path','sha256'])
    summary=dict(files=len(files),bytes=sum(x['bytes'] for x in inventory),duplicate_groups=sum(len(ps)>1 for ps in groups.values()),
                 redundant_bytes=sum((len(ps)-1)*ps[0].stat().st_size for ps in groups.values() if len(ps)>1 and ps[0].exists()),
                 archived_directories=sum(m['applied'] for m in moves),all_original_hashes_verified=True,deleted_files=0)
    # Dimensioni ridondanti prima del movimento: usare inventario, non percorsi spostati.
    sizes={x['sha256']:x['bytes'] for x in inventory}
    summary['redundant_bytes']=sum((len(ps)-1)*sizes[sha] for sha,ps in groups.items() if len(ps)>1)
    (report/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(report),**summary),indent=2))


if __name__=='__main__': main()
