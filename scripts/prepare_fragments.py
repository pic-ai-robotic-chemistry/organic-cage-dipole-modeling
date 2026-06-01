#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ase.io import read, write

from cage_anion_mc.fragments import detect_fragments, subset_atoms
from cage_anion_mc.utils import dump_json, ensure_dir


def main():
    ap = argparse.ArgumentParser(description="Detect cage/anion fragments and optionally write cage/template/start files.")
    ap.add_argument("xyz", help="Full cage + anions XYZ/EXTXYZ")
    ap.add_argument("--mode", default="auto", choices=["auto", "br", "otf"])
    ap.add_argument("--out", default="prepared")
    ap.add_argument("--keep-anions", type=int, default=None, help="Write a reduced start structure with only the first N anions.")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    atoms = read(args.xyz)
    info = detect_fragments(atoms, mode=args.mode)
    dump_json(info.to_dict(), out / "fragments.json")
    write(out / "cage.xyz", subset_atoms(atoms, info.cage_indices), format="extxyz")
    write(out / "anion_template.xyz", subset_atoms(atoms, info.anion_fragments[0]), format="extxyz")

    if args.keep_anions is not None:
        n = args.keep_anions
        if n > len(info.anion_fragments):
            raise ValueError(f"Requested {n} anions but only detected {len(info.anion_fragments)}")
        keep = info.cage_indices[:]
        for frag in info.anion_fragments[:n]:
            keep.extend(frag)
        write(out / f"start_N{n}.xyz", subset_atoms(atoms, keep), format="extxyz")

    print(f"Detected anion_type={info.anion_type}, n_anions={len(info.anion_fragments)}, cage_atoms={len(info.cage_indices)}")
    print(f"Wrote {out / 'fragments.json'}")


if __name__ == "__main__":
    main()
