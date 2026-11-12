#!/usr/bin/env python3
"""Prepare MLE-bench data for the selected task.

Examples:
  bin/cli/prepare_data.sh list
  bin/cli/prepare_data.sh install -c tabular-playground-series-dec-2021
  bin/cli/prepare_data.sh install -c new-york-city-taxi-fare-prediction --source private/rebuilt
  bin/cli/prepare_data.sh check [-c <competition-id>]

install:
  1) takes the task's prepared data (from the MLE-bench cache via `mlebench prepare -c <id>`,
     or from --source <dir> with public/ and private/);
  2) verifies it against MLE-bench checksums.yaml;
  3) puts the public part into sandbox/data/ (agent sandbox),
     and private answers/leaderboard/metadata into private/competitions/<id>/;
  4) writes the task description to sandbox/TASK.md and generates sandbox/data/README.md;
  5) records the active task in private/competition.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

RULES = [
    "обучаться можно только на обучающей выборке из `data/`;",
    "нельзя читать `private/`, `bin/`, кэш MLE-bench, raw-данные и любые производные исходного train.csv;",
    "нельзя получать test-set feedback (запускать `bin/cli/grade.sh`/`bin/py/grade.py`, читать приватные метки, "
    "пользоваться внешними сервисами для оценки);",
    "валидацию и early stopping делай только на holdout, вырезанном из обучающей выборки;",
    "сабмит кладётся в `final_solution/submission.csv`;",
    "официальная оценка выполняется координатором автоматически после hard_end и публикуется в `blackboard/`.",
]


def fmt_score(value: float) -> str:
    return f"{value:.5f}"


def read_header(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            return next(csv.reader(f), [])
    except OSError:
        return []


def count_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    rows = 0
    with open(path, "rb") as f:
        for _ in f:
            rows += 1
    return max(0, rows - 1)


ID_LIKE = {"id", "key", "image_id", "filename", "file", "name", "index"}


def is_id_like(column: str) -> bool:
    name = column.strip().lower()
    return name in ID_LIKE or name.endswith("_id")


def detect_layout(public: Path) -> dict:
    test = public / "test.csv"
    sample = public / "sample_submission.csv"
    test_cols = set(read_header(test))
    sample_cols = read_header(sample)

    # target columns = submission columns except identifiers; otherwise — train/test difference
    target = [c for c in sample_cols if not is_id_like(c)]
    candidates = sorted(
        p for p in public.glob("*.csv") if p.name not in {"test.csv", "sample_submission.csv"}
    )
    preferred = [p for p in candidates if p.name in {"train.csv", "labels.csv"}]
    ordered = preferred + [p for p in candidates if p not in preferred]

    train: Path | None = None
    for path in ordered:
        cols = read_header(path)
        if not target:
            extra = [c for c in cols if c not in test_cols and not is_id_like(c)]
            if extra:
                train, target = path, extra
                break
        if path.name in {"train.csv", "labels.csv"} and train is None:
            train = path
    if train is None and ordered:
        train = ordered[0]
    if not target:
        target = [c for c in read_header(train) if not is_id_like(c)] if train else []

    return {
        "train_file": train.name if train else None,
        "train_rows": count_rows(train) if train else None,
        "target_columns": target,
        "test_file": test.name if test.is_file() else None,
        "test_rows": count_rows(test) if test.is_file() else None,
        "sample_submission": sample.name if sample.is_file() else None,
        "submission_columns": read_header(sample),
    }


def describe_train(meta: dict) -> str:
    if not meta["train_file"]:
        return "обучающая выборка: см. файлы в `data/`"
    text = f"обучающая выборка: `data/{meta['train_file']}`"
    if meta["train_rows"] is not None:
        text += f" ({meta['train_rows']:,} строк)"
    if meta["target_columns"]:
        cols = ", ".join(f"`{c}`" for c in meta["target_columns"])
        text += f", целевые колонки: {cols}"
    return text


def describe_test(meta: dict) -> str:
    if not meta["test_file"]:
        return "тест: см. файлы в `data/`"
    text = f"тест без метки: `data/{meta['test_file']}`"
    if meta["test_rows"] is not None:
        text += f" ({meta['test_rows']:,} строк)"
    return text


def describe_submission(meta: dict) -> str:
    if not meta["sample_submission"]:
        return "формат сабмита: см. `data/`"
    text = f"формат сабмита: `data/{meta['sample_submission']}`"
    if meta["submission_columns"]:
        cols = ", ".join(f"`{c}`" for c in meta["submission_columns"])
        text += f" (колонки: {cols})"
    return text


def metric_line(meta: dict) -> str:
    direction = "меньше — лучше" if meta["is_lower_better"] else "больше — лучше"
    return f"метрика: `{meta['metric']}` ({direction})"


def threshold_items(meta: dict) -> list[str]:
    th = meta["thresholds"]
    if not th:
        return ["- пороги: лидерборд недоступен"]
    teams = f" ({meta['num_teams']} команд в лидерборде)" if meta.get("num_teams") else ""
    return [
        f"- пороги{teams}:",
        f"- gold {fmt_score(th['gold'])}",
        f"- silver {fmt_score(th['silver'])}",
        f"- bronze {fmt_score(th['bronze'])}",
        f"- median {fmt_score(th['median'])}",
    ]


def write_readme(meta: dict, installed: dict[str, str]) -> None:
    lines = [
        f"# data — {meta['name']} (`{meta['competition']}`) — MLE-bench",
        "",
        "В `data/` лежит только public-часть сплита MLE-bench для активной задачи.",
        "",
        f"- {describe_train(meta)};",
        f"- {describe_test(meta)};",
        f"- {describe_submission(meta)};",
        "- полное описание задачи: `TASK.md` / `data/description.md`.",
        "",
        metric_line(meta),
        "",
        "Правила:",
        "",
        *[f"- {rule}" for rule in RULES],
        "",
        "Хэши установленных файлов (md5, сверены с `checksums.yaml` MLE-bench):",
        "",
        "| файл | md5 |",
        "|---|---|",
        *[f"| data/{name} | `{digest}` |" for name, digest in sorted(installed.items())],
        "",
    ]
    (common.DATA / "README.md").write_text("\n".join(lines))
    print("  written data/README.md")


def write_task(description: str) -> None:
    (common.SANDBOX / "TASK.md").write_text(description.strip() + "\n")
    print("  written TASK.md (только description)")


# --------------------------------------------------------------------------- install


def resolve_prepared(comp, source: Path | None, keep_raw: bool) -> tuple[Path, Path, str]:
    if source is not None:
        public, private = common.resolve_prepared_source(source)
        return public, private, f"source:{source}"
    if common.is_dataset_prepared(comp):
        return comp.public_dir, comp.private_dir, f"cache:{common.MLEBENCH_DATA_DIR}"
    args = ["prepare", "-c", comp.id, "--data-dir", str(common.MLEBENCH_DATA_DIR)]
    if keep_raw:
        args.append("--keep-raw")
    common.run_mlebench_cli(args, cwd=common.MLEBENCH_DATA_DIR.parent)
    if not common.is_dataset_prepared(comp):
        common.fail(f"после `mlebench prepare -c {comp.id}` данные не найдены в {comp.public_dir}")
    return comp.public_dir, comp.private_dir, "mlebench-prepare"


def write_snapshot(
    comp, private_dir: Path, computed: dict[str, str], source_label: str
) -> dict:
    import pandas as pd

    out = common.competition_dir(comp.id)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    answers_src = private_dir / Path(comp.answers).name
    if not answers_src.is_file():
        common.fail(f"не найден файл приватных ответов {answers_src}")
    answers_dst = out / f"answers{answers_src.suffix or '.csv'}"
    shutil.copy2(answers_src, answers_dst)
    print(f"  snapshot {common.rel(answers_dst)} ← {answers_src}")

    if not comp.leaderboard.is_file():
        common.fail(f"не найден лидерборд {comp.leaderboard}")
    shutil.copy2(comp.leaderboard, out / "leaderboard.csv")
    leaderboard = pd.read_csv(comp.leaderboard)
    rank = comp.grader.rank_score(None, leaderboard)

    layout = detect_layout(private_dir.parent / "public")
    meta = {
        "competition": comp.id,
        "name": comp.name,
        "competition_type": comp.competition_type,
        "metric": common.short_metric(comp.grader.name, comp.id),
        "is_lower_better": bool(comp.grader.is_lower_better(leaderboard)),
        "num_teams": int(len(leaderboard)),
        "thresholds": {
            "gold": rank["gold_threshold"],
            "silver": rank["silver_threshold"],
            "bronze": rank["bronze_threshold"],
            "median": rank["median_threshold"],
        },
        "answers_file": answers_dst.name,
        "answers_original": Path(comp.answers).name,
        "answers_source": str(answers_src),
        "leaderboard_file": "leaderboard.csv",
        "source": source_label,
        "prepared_at": common.utc_now(),
        "public_checksums": {
            key.split("/", 1)[1]: value
            for key, value in computed.items()
            if key.startswith("public/")
        },
        **layout,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"  written {common.rel(out / 'meta.json')}")
    return meta


def activate(comp, public_dir: Path, meta: dict, computed: dict[str, str]) -> dict[str, str]:
    if common.DATA.exists():
        shutil.rmtree(common.DATA)
    common.DATA.mkdir(parents=True)

    installed: dict[str, str] = {}
    for src in sorted(public_dir.iterdir()):
        dst = common.DATA / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
            print(f"  installed data/{src.name}/ (каталог)")
            continue
        if not src.is_file():
            continue
        shutil.copy2(src, dst)
        digest = computed.get(f"public/{src.name}") or common.md5(dst)
        installed[src.name] = digest
        print(
            f"  installed data/{src.name} ({common.human_size(dst.stat().st_size)}) md5 {digest}"
        )

    if not (common.DATA / "description.md").exists():
        (common.DATA / "description.md").write_text(comp.description)
        print("  installed data/description.md")

    common.ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    common.ACTIVE_FILE.write_text(
        json.dumps(
            {
                "competition": comp.id,
                "name": comp.name,
                "activated_at": common.utc_now(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return installed


# --------------------------------------------------------------------------- commands


def cmd_list(args: argparse.Namespace) -> int:
    reg = common.registry()
    active = common.active_competition_id()
    installed = {
        p.name for p in common.COMPETITIONS_DIR.iterdir() if p.is_dir()
    } if common.COMPETITIONS_DIR.is_dir() else set()
    print(f"активная задача: {active or '—'}")
    print()
    print(f"{'':2s} {'competition':50s} {'prepared':9s} installed")
    for comp_id in reg.list_competition_ids():
        try:
            comp = reg.get_competition(comp_id)
            prepared = "yes" if common.is_dataset_prepared(comp) else "-"
        except Exception:
            prepared = "?"
        marks = []
        if comp_id == active:
            marks.append("active")
        if comp_id in installed:
            marks.append("installed")
        print(f"{'*' if comp_id == active else ' ':2s} {comp_id:50s} {prepared:9s} {' '.join(marks)}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    comp_id = common.resolve_competition(args.competition)
    comp = common.get_competition(comp_id)
    print(f"== задача: {comp.name} (`{comp.id}`)")
    public_dir, private_dir, source_label = resolve_prepared(comp, args.source, args.keep_raw)
    print(f"== prepared: {public_dir.parent} ({source_label})")

    print("== проверка контрольных сумм")
    computed = common.verify_source(comp, public_dir, private_dir, skip=args.skip_verification)

    print("== снапшот приватной части")
    meta = write_snapshot(comp, private_dir, computed, source_label)

    if args.no_activate:
        print("== активация пропущена (--no-activate)")
        print_summary(meta)
        return 0

    print("== установка public-данных в sandbox/data")
    installed = activate(comp, public_dir, meta, computed)

    if args.no_task:
        print("== TASK.md не перегенерируется (--no-task)")
    else:
        description = comp.description
        write_task(description)
    write_readme(meta, installed)
    print_summary(meta)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    comp_id = common.resolve_competition(args.competition)
    comp = common.get_competition(comp_id)
    meta = common.load_meta(comp_id)
    if meta is None:
        common.fail(f"задача `{comp_id}` не установлена; выполни bin/cli/prepare_data.sh install -c {comp_id}")
    expected = common.expected_checksums(comp)
    active = common.active_competition_id()

    ok = True
    print(f"задача: {comp.name} (`{comp.id}`); активная: {active or '—'}")
    print()
    print(f"{'file':48s} {'expected':34s} {'actual':34s} status")

    def check_file(label: str, path: Path, expected_md5: str) -> None:
        nonlocal ok
        if not path.exists():
            print(f"{label:48s} {expected_md5:34s} {'-':34s} MISSING")
            ok = False
            return
        actual = common.md5(path)
        status = "OK" if actual == expected_md5 else "FAIL"
        ok = ok and actual == expected_md5
        print(f"{label:48s} {expected_md5:34s} {actual:34s} {status}")

    if active == comp_id:
        for name, expected_md5 in sorted(expected.get("public", {}).items()):
            check_file(f"sandbox/data/{name}", common.DATA / name, expected_md5)
    else:
        print("(public-часть не активна: проверяется только приватный снапшот)")

    answers = common.answers_path(comp_id)
    original = meta.get("answers_original")
    expected_private = expected.get("private", {})
    if original in expected_private:
        check_file(f"private/competitions/{comp_id}/{answers.name}", answers, expected_private[original])
    else:
        print(f"{'private answers':48s} {'(нет в checksums)':34s} {'-':34s} SKIP")

    for label in (
        f"private/competitions/{comp_id}/leaderboard.csv",
        f"private/competitions/{comp_id}/meta.json",
        "sandbox/TASK.md",
        "sandbox/data/README.md",
        "sandbox/data/description.md",
    ):
        path = common.ROOT / label
        exists = path.exists()
        ok = ok and exists
        print(f"{label:48s} {'':34s} {'':34s} {'OK' if exists else 'FAIL'}")
    return 0 if ok else 1


def print_summary(meta: dict) -> None:
    print()
    print(f"задача: {meta['name']} (`{meta['competition']}`)")
    print(f"  {metric_line(meta)}")
    for line in threshold_items(meta):
        print(f"  {line}")
    if meta["train_file"]:
        print(f"  train: data/{meta['train_file']}")
    if meta["test_file"]:
        print(f"  test:  data/{meta['test_file']}")
    print("  сабмит агентов: sandbox/final_solution/submission.csv")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].lstrip("-") not in {"list", "install", "prepare", "check", "help"}:
        argv.insert(0, "install")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="показать доступные задачи MLE-bench")
    p_list.set_defaults(func=cmd_list)

    for name in ("install", "prepare"):
        p = sub.add_parser(name, help="установить/переключить задачу")
        p.add_argument("-c", "--competition", help="ID задачи MLE-bench")
        p.add_argument("--source", type=Path, default=None,
                       help="готовая prepared-раскладка с public/ и private/ (вместо кэша mlebench)")
        p.add_argument("--keep-raw", action="store_true", help="передать --keep-raw в mlebench prepare")
        p.add_argument("--skip-verification", action="store_true",
                       help="не падать на несовпадении контрольных сумм")
        p.add_argument("--no-activate", action="store_true",
                       help="только сделать приватный снапшот, не трогая sandbox/data")
        p.add_argument("--no-task", action="store_true", help="не перегенерировать sandbox/TASK.md")
        p.set_defaults(func=cmd_install)

    p_check = sub.add_parser("check", help="проверить установленную задачу по checksums MLE-bench")
    p_check.add_argument("-c", "--competition", help="ID задачи (по умолчанию активная)")
    p_check.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
