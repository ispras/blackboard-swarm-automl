# Blackboard-Based LLM Agent Swarms for Autonomous Machine Learning Engineering

A research harness for studying whether a lightweight swarm of parallel LLM coding
agents — communicating only through a shared markdown blackboard — can solve
autonomous machine-learning engineering tasks competitively, evaluated on
[MLE-Bench](https://github.com/openai/mle-bench).

The repository contains:

- the Docker-based experiment harness (swarm, tools, and grader containers),
- the agent prompts (single-agent baseline and swarm protocols),
- the blackboard/analysis pipeline (citation graph, message metrics, token costs),
- the experiment configuration and task subsets used in the paper,
- the scripts that produce the paper figures.

## Paper

**Blackboard-Based LLM Agent Swarms for Autonomous Machine Learning Engineering**

Nikita Shevtsov, Vladimir Mayorov, Ekaterina Lobastova — Ivannikov Institute for
System Programming of the Russian Academy of Sciences (ISP RAS), Moscow, Russia.

> **Abstract.** Autonomous machine-learning engineering agents are commonly built on
> increasingly complex harnesses, including search graphs, Monte Carlo tree search,
> evolutionary loops, predefined roles, and fixed synchronization or
> resource-allocation schemes. Recent evidence suggests that strong models may not
> require such heavy scaffolding, while static one-step operators and rigid
> coordination protocols can limit adaptation to complex, multi-file ML tasks. We
> study whether a simple swarm of parallel ReAct agents can achieve strong Automated
> Machine Learning performance through blackboard coordination alone, without forced
> discrete synchronization. Agents communicate through a shared blackboard, cite each
> other's messages, propagate plans, insights, and conflicts, and allocate effort
> through discussion rather than developing through a predefined search policy. We
> evaluate the swarm on ten MLE-Bench tasks, using three independent runs per task,
> each with three homogeneous agents and a 12-hour time budget; seven task-runs that
> violated the internet rules were rerun under a tightened protocol. The swarm
> achieves competitive performance, earning 10 gold, 6 silver, and 1 bronze medals
> over 30 task-runs, and finishing above the task median in 26 of 30 task-runs. The
> analysis of blackboard citation chains provides evidence of cross-agent information
> reuse: agents cite one another's results, and citation chains reach a median depth
> of about ten. These results show that a lightweight blackboard swarm is feasible
> and competitive on this ten-task benchmark without a predefined search policy; they
> do not isolate how much of the performance is attributable to coordination rather
> than to parallel exploration, additional inference, model capability, or shared
> artifacts.

Corresponding author: Nikita Shevtsov <shevtsov@ispras.ru>.

### Headline results

| Solution | LLM model | Time (h) | Seeds | Any medal (%) | Above median (%) |
| --- | --- | --- | --- | --- | --- |
| Famou-Agent-2.0 | Gemini-3-Pro-Preview | 24 | 3 | 60.0 | 63.3 |
| AIBuildAI | Claude-Opus-4.6 | 24 | 3 | 60.0 | 63.3 |
| **Blackboard Swarm (ours)** | **DeepSeek-V4.1-Flash** | **12** | **3** | **56.7** | **86.7** |
| MARS+ | Gemini-3-Pro-Preview | 24 | 3 | 56.7 | 60.0 |
| MLEvolve | Gemini-3-Pro-Preview | 12 | 3 | 46.7 | 56.7 |
| MARS | Gemini-3-Pro-Preview | 24 | 3 | 43.3 | 46.7 |
| Famou-Agent-2.0 | Gemini-2.5-Pro | 24 | 3 | 43.3 | 43.3 |
| Thesis | GPT-5-Codex | 24 | 3 | 40.0 | 40.0 |
| ML-Master 2.0 | DeepSeek-V3.2-Speciale | 24 | 3 | 36.7 | 46.7 |
| Famou-Agent | Gemini-2.5-Pro | 24 | 3 | 33.3 | 36.7 |
| AIRA-dojo | o3 | 24 | 20 | 24.5 | 29.0 |
| AIDE | o1-preview | 24 | 21 | 7.6 | 14.8 |

Comparison with MLE-Bench leaderboard entries on the same ten tasks (from the
paper's Table II; external entries are author-recomputed on the shared subset).
The swarm runs three homogeneous agents on a flash-tier backbone
(DeepSeek-V4.1-Flash) with a 12-hour budget per task, earning 10 gold, 6 silver,
and 1 bronze medals over 30 task-runs.

## Quickstart

### Prerequisites

- Docker Engine with the Compose plugin; NVIDIA Container Toolkit if the swarm
  should use a GPU.
- An OpenCode-compatible model endpoint. Configure it in `.env`:

```bash
cp .env.example .env
# edit .env:
#   OPENCODE_AUTH_JSON='{"deepseek":{"type":"api","key":"sk-..."}}'
```

### Offline toy example (no GPU, no Kaggle credentials)

The `swarm-min`/`tools-min` image targets and the synthetic
[`examples/toy-spaceship`](examples/toy-spaceship) dataset let you exercise the
entire pipeline end-to-end offline:

```bash
make build-min          # light images for the offline toy example
make toy-prepare        # install the synthetic task into the sandbox
make toy-grade          # grade the toy baseline submission
```

For the toy demo set `SWARM_GPU_COUNT=0` in `.env`.

### Full MLE-bench run

```bash
make build              # full swarm + tools images (several GB)

# Install a competition dataset (requires Kaggle credentials in .env or .kaggle/kaggle.json):
bin/cli/prepare_data.sh install -c spooky-author-identification

# Launch a 12-hour, 3-agent swarm on one task:
make up AGENTS=3 HOURS=12 MODEL=deepseek/deepseek-flash REASONING=max PROMPT=prompts/PROMPT_SWARM_v2_internet_usage_rules.md

make logs               # follow swarm and grader logs
make status             # show Docker and swarm state
make down               # stop swarm and grader
```

The environment recommended by the paper is one NVIDIA A100 80 GiB GPU, 96 GiB RAM,
and 32 CPU cores; a GPU run additionally requires
`compose.gpu.yaml` (enabled automatically via `SWARM_GPU_COUNT`).

### Batch of tasks and post-run analysis

```bash
# Run the ten-task subset sequentially (lite10 = the paper subset):
make batch SUBSET=subsets/lite10.txt

# Rebuild the post-run analysis (citation graph, message metrics, costs):
make analysis

make clean              # remove sandbox artifacts (asks for confirmation)
```

All commands are thin wrappers over `bin/cli/swarm.sh` and friends; run them
directly for full control:

```bash
bin/cli/swarm.sh up|batch|server|final-phase|stop|status|logs|attach|shell|build
bin/cli/prepare_data.sh list|install
bin/cli/grade.sh --submission <file>
```

### Where to find results

During and after a run, artifacts are written to:

| Path | Contents |
| --- | --- |
| `sandbox/blackboard/` | All board messages, `agent_logs/`, `graph.json` / `graph.html` |
| `sandbox/workspaces/` | Per-agent working directories |
| `sandbox/final_solution/` | `submission.csv` and `grade.json` produced by the grader |
| `sandbox/analysis/` | Citation graph, message metrics, and cost breakdown |
| `costs/COSTS.md` | Token usage and cost summary |
| `results/` | Saved snapshots of finished runs |

With `SWARM_AUTO_SAVE=1` (the default), the grader automatically saves a snapshot
after grading to `results/<task>_run-<timestamp>/` containing `blackboard/`,
`final_solution/`, and `analysis/`. Batch runs write per-task directories
`results/<task>_run-*` with logs. To save the current run manually (for example
before `make clean`), run:

```bash
bin/cli/save.sh [snapshot_name]
```

`make clean` removes `sandbox/` but keeps `results/` and `costs/`.

## Architecture and blackboard design

The system is orchestrated by [`compose.yaml`](compose.yaml) with three services
built from [`docker/Dockerfile`](docker/Dockerfile):

| Service | Role |
| --- | --- |
| `swarm` | Runs the agent sessions. The container boots systemd as PID 1 so each agent session is a real `systemd-run --user --scope` unit with its own OpenCode process, resource limits, and logs. |
| `tools` | Data preparation, submission, grading, and the post-run analysis scripts. |
| `grader` | A watcher loop that detects new submissions and grades them against the private test labels via MLE-Bench. |

**Blackboard.** There is no coordinator and no forced synchronization. All agents
share one directory, `sandbox/blackboard`. Each agent writes and reads Markdown
message files named `{AGENT_ID}_{title}.md`, typically inspecting the board with
`ls -lt blackboard` before acting. Messages carry semantic hashtags from a fixed
vocabulary (e.g. `#feature_engineering`, `#ensembling`, `#insight`, `#conflict`)
and wrap numeric results in `<metric>` tokens so that the swarm's progress can be
parsed automatically after the run. Messages link to earlier messages by citing
them; these citation edges are extracted into a per-run knowledge graph that the
paper analyzes (depth, cross-agent share, citation lag, etc.).

**Prompts.** [`prompts/`](prompts) contains the single-agent baseline
(`PROMPT_SINGLE.md`) and the swarm protocols (`PROMPT_SWARM_v1.md`,
`PROMPT_SWARM_v2_internet_usage_rules.md`), which define the blackboard etiquette, message format,
hashtag vocabulary, and the `<metric>` reporting convention.

**Sandbox.** Each run uses an isolated `sandbox/` workspace with the task data
(`sandbox/data`), the shared blackboard (`sandbox/blackboard`), per-agent
workspaces, the assembled `final_solution`, and the generated `analysis/` output.

## Configuration reference

### Build variables (`compose.yaml` build args)

| Variable | Default | Meaning |
| --- | --- | --- |
| `SWARM_UID` / `SWARM_GID` | `1000` | Container user; set to `id -u` / `id -g` if different. |
| `NODE_VERSION` | `20.17.0` | Node.js version in the images. |
| `OPENCODE_VERSION` | `1.18.34` | OpenCode harness version. |
| `MLEBENCH_REF` | `507f92e1138bb6e40dac5c6ee7a6758e6424bf97` | MLE-bench repository commit used for grading. |

### Image targets

| Variable | Values | Meaning |
| --- | --- | --- |
| `SWARM_TARGET` | `swarm` (default), `swarm-min` | Full stack vs. light offline stack. |
| `SWARM_TOOLS_TARGET` | `tools` (default), `tools-min` | Full stack vs. light offline stack. |
| `SWARM_IMAGE` / `SWARM_TOOLS_IMAGE` | — | Optional image-name overrides. |

### Runtime variables (`.env`, also accepted by `bin/cli/swarm.sh up`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `SWARM_TZ` | `UTC` | Container timezone. |
| `SWARM_AGENTS` | `3` | Number of parallel agent sessions. |
| `SWARM_HOURS` | `12` | Wall-clock budget per task (work ends 30 min early, hard end at H). |
| `SWARM_MODEL` | `deepseek/deepseek-flash` | OpenCode model identifier. |
| `SWARM_REASONING` | `max` | Model reasoning variant, passed to OpenCode as `--variant` (`--reasoning` on `up`/`batch`). |
| `SWARM_PROMPT` | `prompts/PROMPT_SWARM_v2_internet_usage_rules.md` | Agent prompt file (pass explicitly via `--prompt`; see `prompts/` for available files). |
| `SWARM_GPU_COUNT` | `1` | GPUs reserved for the swarm container (`0` = CPU-only toy demo). |
| `SWARM_CPUS` | `32` | CPU limit for the swarm container. |
| `SWARM_MEM_LIMIT` | `96g` | Memory limit for the swarm container. |
| `SWARM_SHM` | `8g` | Shared memory size. |
| `SWARM_PORT` | `4096` | OpenCode server port. |
| `SWARM_BOARD_INTERVAL` | `300` | Board digest period in seconds (`0` disables). |
| `SWARM_AUTO_SAVE` | `1` | Auto-save results to `results/`. |
| `SWARM_SERVER_ONLY` | `0` | Start only the server, without agents. |
| `OPENCODE_AUTH_JSON` | — | OpenCode auth as a single-line JSON (alternative: `./.opencode/auth.json`). |
| `OPENCODE_CONFIG_JSON` | — | Optional OpenCode config override. |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | — | Kaggle credentials for `prepare_data.sh install` (alternative: `./.kaggle/kaggle.json`). |
| `TOOLS_MEM_LIMIT` / `TOOLS_CPUS` | `8g` / `4` | Resource limits for the tools and grader containers. |

## Experiments and results

The paper evaluates the swarm on a ten-task subset of MLE-Bench
([`subsets/lite10.txt`](subsets/lite10.txt)), balanced across
complexity (3 high, 4 medium, 3 low) and task category. Each task is run three times
with fresh agents, empty blackboards, and a 12-hour budget, for 30 task-runs total.
Scoring uses the official MLE-Bench grading on held-out test labels.

Summary over 30 task-runs:

| Metric | Value |
| --- | --- |
| Medals | 10 gold, 6 silver, 1 bronze |
| Any-medal rate | 56.7% |
| Above task median | 26 / 30 (86.7%) |

The raw runs, blackboards, agent logs, and per-run grades used in the paper are
archived under `paper/analysis/runs/`. The analysis pipeline
([`paper/analysis/code/`](paper/analysis/code)) produces the figures and tables in
[`paper/analysis/figures/`](paper/analysis/figures).

## Repository layout

```
bin/                 CLI wrappers (swarm, batch, grade, analysis, ...) and Python analysis code
compose.yaml         Main Docker Compose definition (swarm / tools / grader)
compose.gpu.yaml     NVIDIA GPU reservation override
docker/              Dockerfile with swarm/tools/swarm-min/tools-min stages
prompts/             Single-agent baseline and swarm protocol prompts
examples/            Offline toy example (toy-spaceship dataset + baseline submission)
subsets/             Task subsets (dev, low, high, lite10)
paper/               Analysis pipeline and artifacts used for the paper figures
sandbox/             Runtime state per run: blackboard, workspaces, final_solution, analysis
costs/, results/     Token costs and saved results produced by runs
```

## License

MIT — see the repository license file for details. The synthetic toy dataset in
`examples/toy-spaceship` is generated for smoke tests and must not be used as a
benchmark result.
