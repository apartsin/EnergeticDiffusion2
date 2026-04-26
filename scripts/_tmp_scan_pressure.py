"""Scan raw folder for pressure-like columns we may have missed."""
import pandas as pd
from pathlib import Path

RAW = Path('E:/Projects/EnergeticDiffusion2/data/raw/energetic_external')

def try_read(p, nrows=5):
    if p.suffix.lower() == '.csv':
        try:
            return pd.read_csv(p, nrows=nrows, low_memory=False)
        except Exception:
            return pd.read_excel(p, nrows=nrows)
    return pd.read_excel(p, nrows=nrows)

hits = []
for p in RAW.rglob('*'):
    if p.is_dir() or p.suffix.lower() not in ('.csv','.xlsx','.xls'):
        continue
    try:
        df = try_read(p)
    except Exception:
        continue
    press_cols = [c for c in df.columns
                  if any(k in str(c).lower() for k in ('press','pcj','pdet','p_cj','detp'))]
    if press_cols:
        hits.append((p, press_cols))

print(f'Files with pressure-like columns: {len(hits)}')
for p, pcols in hits:
    try:
        full = try_read(p, nrows=None)
        n = len(full)
        nz = {c: int(full[c].notna().sum()) for c in pcols}
    except Exception:
        n, nz = '?', {}
    rel = str(p).replace(str(Path.cwd()), '').lstrip('\\/')
    print(f'  {rel}')
    print(f'    rows={n}  pressure cols (non-null): {nz}')
