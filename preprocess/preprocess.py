"""
Unified preprocessing script for all supported datasets.

Usage:
    python -m preprocess.preprocess --dataset xmno --data_root /path/to/data
    python -m preprocess.preprocess --dataset mp   --data_root /path/to/data --num_workers 4
"""

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import warnings
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from ase.io import read
from tqdm import tqdm

from preprocess.graph_constructor import AtomsToGraphs

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Per-dataset configuration
# ---------------------------------------------------------------------------
# Each dataset provides:
#   load_splits(data_root) -> dict[str, list]
#       Returns {"train": [...], "val": [...], "test": [...]}
#       Each element is a tuple (id_str, extra_dict | None)
#
#   get_paths(data_root, item_id) -> (unrelaxed_path, relaxed_path)
#
#   min_atoms: int – skip structures smaller than this
# ---------------------------------------------------------------------------


def _load_splits_csv_column(data_root, filenames, id_column):
    """Load train/val/test CSVs that have a single ID column."""
    splits = {}
    for split, fname in filenames.items():
        df = pd.read_csv(os.path.join(data_root, fname))
        ids = df[id_column].astype(str).to_numpy()
        splits[split] = [(sid, None) for sid in ids]
    return splits


# ---- xmno ----------------------------------------------------------------

def _load_splits_xmno(data_root):
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(data_root, f"id_prop_{split}_all.csv")
        with open(path) as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row[0].split("_")[-1] == "relaxed"]
        splits[split] = [(row[0], {"y": float(row[1])}) for row in rows]
    return splits


def _get_paths_xmno(data_root, item_id):
    # New XMNO layout: <data_root>/CIF/*.cif
    cif_root = os.path.join(data_root, "CIF")
    relaxed = os.path.join(cif_root, f"{item_id}.cif")
    unrelaxed = os.path.join(cif_root, f"{item_id.replace('relaxed', 'unrelaxed')}.cif")

    # Backward compatibility for older layout: <data_root>/*.cif
    if not (os.path.exists(relaxed) and os.path.exists(unrelaxed)):
        relaxed = os.path.join(data_root, f"{item_id}.cif")
        unrelaxed = os.path.join(data_root, f"{item_id.replace('relaxed', 'unrelaxed')}.cif")
    return unrelaxed, relaxed


# ---- mp -------------------------------------------------------------------

def _load_splits_mp(data_root):
    return _load_splits_csv_column(
        data_root,
        {"train": "train.csv", "val": "val.csv", "test": "test.csv"},
        id_column="mp_id",
    )


def _get_paths_mp(data_root, item_id):
    relaxed = os.path.join(data_root, "CIF", f"{item_id}_R.cif")
    unrelaxed = os.path.join(data_root, "CIF", f"{item_id}_U.cif")
    return unrelaxed, relaxed


# ---- c2db / vdW / jarvis (same layout) -----------------------------------

def _load_splits_atoms_id(data_root):
    return _load_splits_csv_column(
        data_root,
        {"train": "train.csv", "val": "val.csv", "test": "test.csv"},
        id_column="atoms_id",
    )


def _get_paths_atoms_id(data_root, item_id):
    relaxed = os.path.join(data_root, "CIF", f"{item_id}_relaxed.cif")
    unrelaxed = os.path.join(data_root, "CIF", f"{item_id}_unrelaxed.cif")
    return unrelaxed, relaxed


def _cif_id_identity(item_id):
    return item_id


def _cif_id_mp(item_id):
    return f"{item_id}_U"


# ---- registry -------------------------------------------------------------

DATASET_REGISTRY = {
    "xmno": {
        "load_splits": _load_splits_xmno,
        "get_paths": _get_paths_xmno,
        "min_atoms": 0,
        "cif_id_fn": _cif_id_identity,
    },
    "mp": {
        "load_splits": _load_splits_mp,
        "get_paths": _get_paths_mp,
        "min_atoms": 0,
        "cif_id_fn": _cif_id_mp,
    },
    "c2db": {
        "load_splits": _load_splits_atoms_id,
        "get_paths": _get_paths_atoms_id,
        "min_atoms": 3,
        "cif_id_fn": _cif_id_identity,
    },
    "vdW": {
        "load_splits": _load_splits_atoms_id,
        "get_paths": _get_paths_atoms_id,
        "min_atoms": 3,
        "cif_id_fn": _cif_id_identity,
    },
    "jarvis": {
        "load_splits": _load_splits_atoms_id,
        "get_paths": _get_paths_atoms_id,
        "min_atoms": 3,
        "cif_id_fn": _cif_id_identity,
    },
}


# ---------------------------------------------------------------------------
# LMDB writing
# ---------------------------------------------------------------------------

def write_data(mp_args):
    """Worker: convert CIF pairs → Data objects and write one LMDB shard."""
    a2g, data_root, items, db_path, data_indices, dataset_cfg = mp_args

    get_paths = dataset_cfg["get_paths"]
    min_atoms = dataset_cfg["min_atoms"]
    cif_id_fn = dataset_cfg["cif_id_fn"]

    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
        lock=True,
        readahead=False,
    )

    idx = 0
    with db.begin(write=True) as txn:
        txn.put("version".encode("ascii"), pickle.dumps(1, protocol=-1))

    for index in tqdm(data_indices, desc=f"Writing {os.path.basename(db_path)}", leave=True):
        item_id, extra = items[index]
        unrelaxed_path, relaxed_path = get_paths(data_root, item_id)

        if not (os.path.exists(relaxed_path) and os.path.exists(unrelaxed_path)):
            continue

        try:
            atoms_r = read(relaxed_path)
            if len(atoms_r) < min_atoms:
                continue
            atoms_u = read(unrelaxed_path)
        except Exception:
            continue

        try:
            data = a2g.convert_pairs(atoms_u, atoms_r)
            data.cif_id = cif_id_fn(item_id)
            if extra is not None:
                for k, v in extra.items():
                    setattr(data, k, v)
        except Exception:
            continue

        with db.begin(write=True) as txn:
            txn.put(str(idx).encode("ascii"), pickle.dumps(data, protocol=-1))
        idx += 1

    with db.begin(write=True) as txn:
        txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))

    db.sync()
    db.close()


def build_split(split_name, items, data_root, num_workers, a2g, out_dir, dataset_cfg):
    """Build LMDB shards for one split (train / val / test)."""
    data_len = len(items)
    print(f"{split_name}: {data_len} samples")

    save_path = Path(out_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    data_indices = np.arange(data_len, dtype=np.int64)
    mp_db_paths = [str(save_path / f"data.{i:04d}.lmdb") for i in range(num_workers)]
    mp_data_indices = np.array_split(data_indices, num_workers)

    mp_args = [
        (a2g, data_root, items, mp_db_paths[i], mp_data_indices[i], dataset_cfg)
        for i in range(num_workers)
    ]

    with mp.Pool(processes=num_workers) as pool:
        list(pool.imap_unordered(write_data, mp_args, chunksize=1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess crystal CIF pairs into LMDB for E3RelaxH2."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset name",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Root data directory")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--radius", type=float, default=6.0, help="Neighbor search radius")
    parser.add_argument("--max_neigh", type=int, default=50, help="Max neighbors per atom")
    args = parser.parse_args()

    data_root = args.data_root
    num_workers = max(1, args.num_workers)

    dataset_cfg = DATASET_REGISTRY[args.dataset]
    splits = dataset_cfg["load_splits"](data_root)

    for name, items in splits.items():
        print(f"  {name}: {len(items)} samples")

    a2g = AtomsToGraphs(radius=args.radius, max_neigh=args.max_neigh, use_offset_list=True)

    for split_name, items in splits.items():
        out_dir = os.path.join(data_root, split_name)  # e.g. .../train, .../val, .../test
        build_split(split_name, items, data_root, num_workers, a2g, out_dir, dataset_cfg)

    print("Done.")


if __name__ == "__main__":
    main()
