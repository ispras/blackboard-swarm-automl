#!/usr/bin/env python3
"""Board file mtime manifest: `save` and `restore` via blackboard/.mtime.tsv.

Why: transferring the board between containers/archives (tar/rsync/web) loses mtime,
and timelines break. The manifest is data: it travels with the board and survives the transfer.

Usage:
  bin/py/blackboard_mtime.py save    [--bb sandbox/blackboard]
  bin/py/blackboard_mtime.py restore [--bb sandbox/blackboard]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BB = ROOT / "sandbox" / "blackboard"
MANIFEST = ".mtime.tsv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["save", "restore"])
    ap.add_argument("--bb", type=Path, default=DEFAULT_BB, help="каталог доски (default: sandbox/blackboard)")
    return ap.parse_args()


def safe_name(name: str) -> bool:
    return name not in ("", ".", "..") and Path(name).name == name and "\t" not in name


def cmd_save(bb: Path) -> None:
    if not bb.is_dir():
        sys.exit(f"нет каталога доски: {bb}")
    entries: list[tuple[float, str]] = []
    for p in bb.iterdir():
        if p.is_file() and p.name != MANIFEST and not p.name.endswith(".tmp"):
            try:
                entries.append((p.stat().st_mtime, p.name))
            except OSError:
                continue
    entries.sort(key=lambda x: x[1])
    text = "".join(f"{int(m)}\t{name}\n" for m, name in entries)
    tmp = bb / (MANIFEST + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, bb / MANIFEST)
    print(f"mtime: сохранено {len(entries)} записей -> {bb / MANIFEST}")


def cmd_restore(bb: Path) -> None:
    mf = bb / MANIFEST
    if not mf.is_file():
        sys.exit(f"нет манифеста: {mf}")
    done = missing = 0
    for line in mf.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        epoch, _, name = line.partition("\t")
        epoch, name = epoch.strip(), name.strip()
        if not epoch.isdigit() or not safe_name(name):
            continue
        p = bb / name
        if not p.is_file():
            missing += 1
            continue
        os.utime(p, (int(epoch), int(epoch)))
        done += 1
    msg = f"mtime: восстановлено {done} файлов из {mf}"
    if missing:
        msg += f" (нет на месте: {missing})"
    print(msg)


def main() -> None:
    a = parse_args()
    bb = a.bb if a.bb.is_absolute() else ROOT / a.bb
    if a.action == "save":
        cmd_save(bb)
    else:
        cmd_restore(bb)


if __name__ == "__main__":
    main()
