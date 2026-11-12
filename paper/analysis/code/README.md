# Код анализа

Весь код анализа для статьи лежит в одном месте — `paper/analysis/code/`.
Пайплайн принимает папки прогонов и генерирует данные, графики и tex-фигуры с нуля.

Зависимости (Python >= 3.10):

```bash
python3 -m pip install -r paper/analysis/code/requirements.txt
```

```bash
python paper/analysis/code/run_paper_pipeline.py \
  --runs paper/analysis/runs/1-lite10-blackboard-mlebench \
         paper/analysis/runs/2-lite10-blackboard-mlebench \
         paper/analysis/runs/3-lite10-blackboard-mlebench \
  --out paper/analysis
```

## Состав

**Per-task / per-run анализ** (перенесены из `bin/`, независимы от harness):
- `swarm_analysis.py` — синергия, граф знаний, TTA, resilience по blackboard задачи;
- `blackboard_graph.py` — самодостаточная копия harness-графа (нужна
  `swarm_analysis`), без импортов из `bin/`;
- `board_reading.py` — чтение доски, coverage, staleness, глубина чтения;
- `context_usage.py` — состав и объём контекста;
- `internet_usage.py` — сетевые обращения и веса;
- `log_composition.py` — разбор транскриптов по категориям;
- `forgetting_analysis.py` — глубина чтения/забывание (импортирует `board_reading`).

**Агрегация и фигуры**:
- `paper_metrics.py` — per-run таблицы, LLM-разметка ролей (кэш `data/<run>/roles_raw.json`);
- `combine_runs.py` — усреднение по задачам и прогонам в `paper/analysis/data/combined`;
- `paper_figures_combined.py` — fig01–fig08 (300 dpi) в `paper/analysis/figures`;
- `paper_figures_tex.py` — те же фигуры под именами `paper/tex/fig_swarm_*.png`,
  включая трёхпанельную `fig_swarm_board_depth.png`;
- `fig_tags_compact.py` — двухпанельная `fig_swarm_tags.png`.

Запуск отдельных шагов — см. `--help` каждого скрипта.
