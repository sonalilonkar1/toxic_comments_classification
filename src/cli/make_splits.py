import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def temporal_blocks(n_rows: int, folds: int):
    """Yield contiguous, chronological index blocks."""
    indices = np.arange(n_rows)
    for block in np.array_split(indices, folds):
        yield block


def chrono_split_indices(block: np.ndarray, train=0.7, dev=0.1, test=0.2):
    if not np.isclose(train + dev + test, 1.0):
        raise ValueError("train/dev/test must sum to 1.0")
    n_rows = len(block)
    n_tr = int(train * n_rows)
    n_dv = int(dev * n_rows)
    tr = block[:n_tr].tolist()
    dv = block[n_tr:n_tr + n_dv].tolist()
    te = block[n_tr + n_dv :].tolist()
    return tr, dv, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--dev", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    df = pd.read_csv(ROOT / "data" / "raw" / "train.csv")

    outdir = ROOT / "data" / "splits"
    outdir.mkdir(parents=True, exist_ok=True)

    for f, block in enumerate(temporal_blocks(len(df), args.folds), start=1):
        tr, dv, te = chrono_split_indices(block, train=args.train, dev=args.dev, test=args.test)
        p = outdir / f"fold{f}.json"
        with open(p, "w") as fh:
            json.dump({"train": tr, "dev": dv, "test": te}, fh)
        print(f"Saved {p} :: n_train={len(tr)} n_dev={len(dv)} n_test={len(te)} (block size={len(block)})")

if __name__ == "__main__":
    main()
