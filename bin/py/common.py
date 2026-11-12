#!/usr/bin/env python3
"""Shared utilities for bin/py/: MLE-bench task selection, paths, and mlebench package integration."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "sandbox"
DATA = SANDBOX / "data"
PRIVATE = ROOT / "private"
COMPETITIONS_DIR = PRIVATE / "competitions"
ACTIVE_FILE = PRIVATE / "competition.json"

MLEBENCH_REPO = Path(
    os.environ.get("MLEBENCH_REPO", str(Path.home() / "Desktop" / "mle-bench"))
).resolve()
MLEBENCH_DATA_DIR = Path(
    os.environ.get("MLEBENCH_DATA_DIR", str(Path.home() / ".cache" / "mle-bench" / "data"))
).resolve()
FALLBACK_COMPETITION = "new-york-city-taxi-fare-prediction"

# legacy layout for taxi (before moving to private/competitions/<id>/)
LEGACY_ANSWERS = PRIVATE / "test.csv"
LEGACY_THRESHOLDS = PRIVATE / "thresholds.json"
LEGACY_LEADERBOARD = PRIVATE / "leaderboard.csv"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fail(msg: str, code: int = 2) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------- mle-bench


# For some mle-bench tasks `grader.name` does not match the metric actually computed
# (copy-paste in their config.yaml); use the actual metric for labels.
METRIC_OVERRIDES = {
    "stanford-covid-vaccine": "MCRMSE",          # grade.py computes mean column-wise RMSE
    "ventilator-pressure-prediction": "MAE",     # grade.py computes mean absolute error
}


def short_metric(name: str | None, comp_id: str | None = None) -> str:
    """Short metric name for labels.

    In mle-bench, for some tasks `grader.name` repeats the task id with a `-score` suffix
    (e.g., `jigsaw-unintended-bias-in-toxicity-classification-score`);
    that is extra noise for charts/labels. Return `score`, `accuracy`, `auc-roc`, etc.
    """
    s = (name or "").strip()
    if not s:
        return "metric"
    cid_any = str(comp_id or "").strip().lower()
    if cid_any in METRIC_OVERRIDES:
        return METRIC_OVERRIDES[cid_any]
    low = s.lower()
    if comp_id:
        cid = str(comp_id).strip().lower()
        if cid and (low == cid or low == f"{cid}-score"):
            return "score"
        if cid and low.startswith(cid + "-"):
            s = s[len(cid) + 1:]
            low = s.lower()
    if low == "score":
        return "score"
    if low.endswith("-score"):
        base = s[: -len("-score")].strip("-_ ")
        return base or "score"
    return s


def require_mlebench() -> None:
    import warnings

    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if MLEBENCH_REPO.is_dir() and str(MLEBENCH_REPO) not in sys.path:
        sys.path.insert(0, str(MLEBENCH_REPO))
    try:
        import mlebench  # noqa: F401
    except ImportError:
        fail(
            "текущий python не видит пакет mlebench.\n"
            "Запускай через bin/cli/prepare_data.sh / bin/cli/grade.sh: в Docker они работают\n"
            "в tools-контейнере, где MLE-bench уже установлен."
        )


def registry():
    require_mlebench()
    from mlebench.registry import registry as _registry

    return _registry.set_data_dir(MLEBENCH_DATA_DIR)


def list_competition_ids() -> list[str]:
    return registry().list_competition_ids()


def get_competition(comp_id: str):
    reg = registry()
    if comp_id not in reg.list_competition_ids():
        fail(f"неизвестная задача `{comp_id}`; список: bin/cli/prepare_data.sh list")
    return reg.get_competition(comp_id)


def run_mlebench_cli(args: list[str], cwd: Path) -> None:
    require_mlebench()
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-c", "from mlebench.cli import main; main()", *args]
    print("==> " + " ".join(str(a) for a in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def is_dataset_prepared(comp) -> bool:
    return (
        _non_empty_dir(comp.public_dir)
        and _non_empty_dir(comp.private_dir)
        and comp.answers.is_file()
        and comp.sample_submission.is_file()
    )


def _non_empty_dir(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def resolve_prepared_source(source: Path) -> tuple[Path, Path]:
    source = source.resolve()
    for base in (source, source / "prepared"):
        public, private = base / "public", base / "private"
        if public.is_dir() and private.is_dir():
            return public, private
    fail(
        f"в `{source}` нет public/ и private/: ожидаются "
        f"{source}/public + {source}/private или {source}/prepared/{{public,private}}"
    )


def expected_checksums(comp) -> dict:
    from mlebench.utils import load_yaml

    return load_yaml(comp.checksums)


def verify_source(
    comp, public_dir: Path, private_dir: Path, skip: bool = False
) -> dict[str, str]:
    """Verifies files in the source prepared layout against MLE-bench checksums.yaml."""
    expected = expected_checksums(comp)
    computed: dict[str, str] = {}
    problems: list[str] = []
    for scope, base in (("public", public_dir), ("private", private_dir)):
        for name, exp_md5 in expected.get(scope, {}).items():
            key = f"{scope}/{name}"
            path = base / name
            if not path.is_file():
                problems.append(f"{key}: файл отсутствует")
                continue
            actual = md5(path)
            computed[key] = actual
            if actual != exp_md5:
                problems.append(f"{key}: md5 {actual} != {exp_md5}")
    if problems:
        text = "проверка контрольных сумм MLE-bench не прошла:\n  " + "\n  ".join(problems)
        if skip:
            print(f"WARN: {text}", file=sys.stderr)
        else:
            fail(text)
    return computed


# --------------------------------------------------------------------------- active task


def active_competition_id() -> str | None:
    if not ACTIVE_FILE.exists():
        return None
    try:
        return json.loads(ACTIVE_FILE.read_text()).get("competition")
    except (OSError, json.JSONDecodeError):
        return None


def resolve_competition(explicit: str | None, required: bool = True) -> str | None:
    comp_id = explicit or os.environ.get("COMPETITION") or active_competition_id()
    if not comp_id and required:
        fail(
            "задача не выбрана.\n"
            "Укажи `-c <competition-id>` (список: bin/cli/prepare_data.sh list)\n"
            "или установи активную: bin/cli/prepare_data.sh install -c <competition-id>"
        )
    return comp_id


def competition_dir(comp_id: str) -> Path:
    return COMPETITIONS_DIR / comp_id


def load_meta(comp_id: str) -> dict | None:
    path = competition_dir(comp_id) / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def answers_path(comp_id: str) -> Path:
    meta = load_meta(comp_id)
    if not meta:
        fail(
            f"задача `{comp_id}` не установлена: нет {rel(competition_dir(comp_id) / 'meta.json')}.\n"
            f"Установи: bin/cli/prepare_data.sh install -c {comp_id}"
        )
    path = competition_dir(comp_id) / meta["answers_file"]
    if not path.is_file():
        fail(f"нет файла приватных ответов {rel(path)}; переустанови задачу")
    return path


def leaderboard_snapshot(comp_id: str) -> Path:
    path = competition_dir(comp_id) / "leaderboard.csv"
    if not path.is_file():
        fail(f"нет лидерборда {rel(path)}; переустанови задачу")
    return path


def legacy_layout_available() -> bool:
    return LEGACY_ANSWERS.is_file() and LEGACY_THRESHOLDS.is_file()
