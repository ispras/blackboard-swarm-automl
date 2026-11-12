#!/usr/bin/env python3
"""Grade a submission using the MLE-bench protocol for the selected task.

Examples:
  bin/cli/grade.sh                                  # active task
  bin/cli/grade.sh -c spaceship-titanic --post
  bin/cli/grade.sh --submission sandbox/final_solution/submission.csv --official
  bin/cli/grade.sh --silent                         # metric only, no artifacts on disk

The score is computed by the official task grader from mle-bench (like `mlebench grade-sample`);
thresholds/medals come from the leaderboard snapshot in `private/competitions/<id>/`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

SANDBOX = common.SANDBOX
DEFAULT_SUBMISSION = SANDBOX / "final_solution" / "submission.csv"
DEFAULT_JSON = SANDBOX / "final_solution" / "grade.json"


def active_swarm_units() -> list[str]:
    # No systemd in the Docker harness: the swarm supervisor keeps a marker file.
    if os.environ.get("SWARM_IN_CONTAINER") == "1":
        marker = SANDBOX / "blackboard" / ".swarm_running"
        return ["swarm"] if marker.is_file() else []
    try:
        out = subprocess.run(
            ["systemctl", "--user", "list-units", "--state=active", "--plain", "--no-legend", "swarm-*"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    units = []
    for line in out.splitlines():
        name = line.split()[0] if line.split() else ""
        if name and name not in ("swarm-grade.service", "swarm-final-phase.service"):
            units.append(name)
    return units


def load_context(args: argparse.Namespace):
    comp_id = common.resolve_competition(args.competition, required=False)
    legacy = False
    if comp_id is None:
        if common.legacy_layout_available():
            comp_id = common.FALLBACK_COMPETITION
            legacy = True
        else:
            common.resolve_competition(None)  # exits with a clear message
    comp = common.get_competition(comp_id)
    meta = common.load_meta(comp_id)
    if meta is None and comp_id == common.FALLBACK_COMPETITION and common.legacy_layout_available():
        legacy = True
    if meta is None and not legacy:
        common.fail(
            f"задача `{comp_id}` не установлена: нет {common.rel(common.competition_dir(comp_id))}/meta.json.\n"
            f"Установи: bin/cli/prepare_data.sh install -c {comp_id}"
        )
    if legacy:
        answers = common.LEGACY_ANSWERS
        leaderboard = common.LEGACY_LEADERBOARD
    else:
        answers = common.answers_path(comp_id)
        leaderboard = common.leaderboard_snapshot(comp_id)
    return comp, comp_id, meta, legacy, answers, leaderboard


def grade_submission(comp, submission: Path, answers: Path) -> float:
    from mlebench.utils import load_answers, read_csv

    sub = read_csv(submission)
    ans = load_answers(answers)
    try:
        raw = comp.grader.grade_fn(sub, ans)
    except Exception as e:  # noqa: BLE001 — the official grader signals any invalid submission
        print(f"INVALID: grader отклонил сабмит: {type(e).__name__}: {e}")
        sys.exit(2)
    if raw is None:
        print("INVALID: grader вернул None (невалидный сабмит)")
        sys.exit(2)
    return round(float(raw), 5)


def rank_info(comp, score: float, leaderboard_path: Path) -> tuple[dict, dict, bool]:
    import pandas as pd

    leaderboard = pd.read_csv(leaderboard_path)
    info = comp.grader.rank_score(score, leaderboard)
    thresholds = {
        "gold": float(info["gold_threshold"]),
        "silver": float(info["silver_threshold"]),
        "bronze": float(info["bronze_threshold"]),
        "median": float(info["median_threshold"]),
    }
    flags = {
        key: bool(info[key])
        for key in ("gold_medal", "silver_medal", "bronze_medal", "above_median")
    }
    return thresholds, flags, bool(comp.grader.is_lower_better(leaderboard))


def official_cross_check(comp, submission: Path, score: float) -> float | None:
    if not common.is_dataset_prepared(comp):
        print("  (official cross-check пропущен: prepared-данных нет в кэше MLE-bench)")
        return None
    common.MLEBENCH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(common.MLEBENCH_DATA_DIR.parent)
    try:
        from mlebench.grade import grade_csv

        report = grade_csv(submission, comp)
    finally:
        os.chdir(cwd)
    official = report.score
    if official is None:
        print("  WARN: официальный grade_csv вернул None")
        return None
    delta = abs(float(official) - score)
    print(f"  official mle-bench grade: {float(official):.10f} (delta {delta:.2e})")
    if delta > 1e-9:
        common.fail("наш score расходится с официальным grade_csv")
    return float(official)


def post_to_blackboard(
    comp, score: float, thresholds: dict, flags: dict, lower: bool, path: Path
) -> None:
    bb = SANDBOX / "blackboard"
    if not bb.is_dir():
        return
    from datetime import datetime

    stamp = datetime.now().strftime("%F_%H-%M-%S")
    out = bb / f"{stamp}_SYSTEM_grade.md"
    medal = (
        "gold" if flags["gold_medal"] else
        "silver" if flags["silver_medal"] else
        "bronze" if flags["bronze_medal"] else "нет"
    )
    direction = "меньше — лучше" if lower else "больше — лучше"
    out.write_text(
        "# SYSTEM: оценка финального сабмита (MLE-bench)\n\n"
        f"- задача: `{comp.id}` ({comp.name})\n"
        f"- сабмит: `{common.rel(path)}` (md5 {common.md5(path)})\n"
        f"- метрика: `{common.short_metric(comp.grader.name, comp.id)}` ({direction})\n"
        f"- score: **<metric>{score:.5f}</metric>**\n"
        f"- пороги: gold {thresholds['gold']:.5f} / silver {thresholds['silver']:.5f} / "
        f"bronze {thresholds['bronze']:.5f} / median {thresholds['median']:.5f}\n"
        f"- медаль: {medal}; выше медианы: {'да' if flags['above_median'] else 'нет'}\n"
    )
    print(f"  blackboard post: {common.rel(out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--competition", help="ID задачи MLE-bench (по умолчанию активная)")
    ap.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--force", action="store_true", help="оценивать, даже если сессии роя активны")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--post", action="store_true", help="написать SYSTEM-сообщение в blackboard/")
    mode.add_argument("--silent", action="store_true",
                      help="напечатать только метрику и не создавать артефакты на диске")
    ap.add_argument("--no-official", action="store_true",
                    help="не сверяться с официальным mlebench grade_csv")
    a = ap.parse_args()
    a.submission = a.submission if a.submission.is_absolute() else common.ROOT / a.submission
    a.json_out = a.json_out if a.json_out.is_absolute() else common.ROOT / a.json_out

    running = active_swarm_units()
    if running and not a.force:
        print("ОЦЕНКА ЗАПРЕЩЕНА: сейчас работают сессии роя: " + ", ".join(running))
        print("Тест-фидбек во время работы = test-set leakage. Дождись hard_end или запусти с --force (только координатор).")
        sys.exit(3)

    comp, comp_id, meta, legacy, answers, leaderboard = load_context(a)
    if not a.submission.is_file():
        common.fail(f"нет файла сабмита {a.submission}")
    if legacy and not a.silent:
        print("(legacy-раскладка: private/test.csv + private/thresholds.json)")
    score = grade_submission(comp, a.submission, answers)

    if a.silent:
        print(f"{score:.5f}")
        return

    thresholds, flags, lower = rank_info(comp, score, leaderboard)

    direction = "lower is better" if lower else "higher is better"
    print(f"MLE-bench grade — {comp.name} (`{comp_id}`)")
    print(f"submission: {common.rel(a.submission)} (md5 {common.md5(a.submission)})")
    print(f"metric: {comp.grader.name} ({direction})")
    print(f"score: {score:.5f}")
    print(
        f"thresholds: gold {thresholds['gold']:.5f} | silver {thresholds['silver']:.5f} | "
        f"bronze {thresholds['bronze']:.5f} | median {thresholds['median']:.5f}"
    )
    print(
        f"medals: gold={flags['gold_medal']} silver={flags['silver_medal']} "
        f"bronze={flags['bronze_medal']} | above_median={flags['above_median']}"
    )

    official = None
    if not a.no_official:
        official = official_cross_check(comp, a.submission, score)

    payload = {
        "competition": comp_id,
        "protocol": "mle-bench",
        "metric": common.short_metric(comp.grader.name, comp_id),
        "is_lower_better": lower,
        "submission": common.rel(a.submission),
        "submission_md5": common.md5(a.submission),
        "score": score,
        "official_score": official,
        "thresholds": thresholds,
        "flags": flags,
        "graded_at": common.utc_now(),
    }
    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"artifact: {common.rel(a.json_out)}")
    if a.post:
        post_to_blackboard(comp, score, thresholds, flags, lower, a.submission)


if __name__ == "__main__":
    main()
