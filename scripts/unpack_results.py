"""Extract bundled records without overwriting any existing local result."""
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'recorded_results.zip') as z:
    for name in z.namelist():
        p=root/name
        if p.exists() and p.read_bytes()!=z.read(name):
            raise SystemExit('Different existing result: '+str(p))
    for name in z.namelist():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists(): p.write_bytes(z.read(name))
print('Recorded results are available; no training was started.')
