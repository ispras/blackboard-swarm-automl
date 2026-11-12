# Использование GPU, CPU и RAM агентами

Отчёт по ресурсам: как агенты делили один A100, 128 CPU и общую RAM. Источники — снимки `nvidia-smi`, `ps`, `free` и команды запуска в `agent_logs`.

## 1. Политика из промпта

Агентам было предписано: запускать любые тяжёлые job'ы только через `systemd-run --scope` с memory limit, `OMP_NUM_THREADS ≤ 4`, не более 1–2 тяжёлых задач одновременно и координировать тяжёлые запуски через доску.

## 2. Ёмкость

- **GPU:** 1× NVIDIA A100 80GB PCIe (81920 MiB).

- **CPU:** 128 ядер (`nproc`).

- **RAM:** 503 GB (по `free -g`; swap 0).

Разделения ресурсов на агентов не было: все три агента работали в одном контейнере и видели один GPU, один пул CPU и общую RAM.

## 3. GPU

| Задача | Снимков | Util median | Util max | Mem median, GiB | Mem max, GiB | Idle | >=90% | Procs med/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| billion-word | 34 | 98% | 100% | 27.7 | 64.0 | 15% | 74% | 1/2 |
| bms-mol | 41 | 95% | 100% | 13.3 | 58.5 | 44% | 51% | 0/1 |
| cassava | 38 | 100% | 100% | 34.0 | 56.0 | 13% | 82% | 2/2 |
| mlsp-birds | 9 | 0% | 100% | 0.0 | 14.7 | 89% | 11% | 0/1 |
| nfl-contact | 6 | 0% | 13% | 0.0 | 4.4 | 83% | 0% | 0/0 |
| spooky | 19 | 85% | 100% | 9.9 | 33.5 | 21% | 47% | 0/2 |
| stanford-covid | 5 | 0% | 36% | 0.0 | 3.7 | 60% | 0% | 0/4 |
| taxi | 0 | —% | —% | 0.0 | 0.0 | —% | —% | —/None |
| uw-gi | 27 | 77% | 100% | 7.1 | 23.5 | 26% | 44% | 0/2 |
| ventilator | 3 | 0% | 0% | 0.0 | 0.0 | 100% | 0% | 2/5 |

**Итого:** медиана util 95%, максимум 100%; медиана занятой памяти 13.6 GiB, максимум 64.0 GiB из 80. В снимках с информацией о процессах GPU одновременно было 0/1/2/3+ процессов: 21/9/14/2 (из 46 снимков с такой информацией). Снимки делаются агентами в моменты проверок, поэтому median util завышен; распределение процессов и idle share информативнее.

## 4. CPU и RAM

| Задача | PS snapshots | CPU median | CPU max | RAM median, GB | RAM max, GB | Jobs median | Jobs max | Agents >=2 | Dominance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| billion-word | 3 | 1.0% | 2.7% | 20 | 50 | 3.0 | 4 | 2 | 0.68 |
| bms-mol | 4 | 1.7% | 6.5% | 27 | 47 | 2.5 | 3 | 2 | 0.43 |
| cassava | 5 | 1.8% | 6.1% | 21 | 21 | 7.0 | 9 | 0 | — |
| mlsp-birds | 5 | 6.0% | 27.4% | 19 | 29 | 2.0 | 4 | 1 | 0.87 |
| nfl-contact | 13 | 10.4% | 24.9% | 33 | 45 | 2.0 | 6 | 1 | 0.05 |
| spooky | 2 | 0.6% | 1.0% | 16 | 16 | 2.5 | 3 | 0 | — |
| stanford-covid | 2 | 0.0% | 0.0% | 19 | 19 | 1.0 | 1 | 0 | — |
| taxi | 7 | 1.7% | 6.6% | 63 | 80 | 2.0 | 4 | 4 | 0.27 |
| uw-gi | 3 | 2.4% | 10.3% | 10 | 10 | 9.0 | 10 | 0 | — |
| ventilator | 3 | 0.8% | 2.0% | 15 | 15 | 3.0 | 8 | 2 | 0.30 |

`CPU median/max` — утилизация 128 ядер training-процессами, %; это нижняя оценка, потому что часть процессов не атрибутируется по workspace. `Jobs` — число training-процессов в снимке; `Agents >=2` — сколько снимков имели одновременно процессы ≥2 агентов.

## 5. Как делили

| Механизм | Событий |
|---|---:|
| systemd-run с MemoryMax/High | 1389 |
| systemd-run с CPUQuota | 61 |
| OMP_NUM_THREADS задан | 921 |
| OMP_NUM_THREADS > 4 | 217 |
| CUDA_VISIBLE_DEVICES | 0 |
| фоновые запуски (nohup/&/setsid) | 341 |
| тяжёлый python без systemd-run | 112 |
| kill-команды | 28 |

Основной механизм — не статическое разделение GPU, а **time-sharing**: агенты запускают тяжёлые job'ы в общий GPU и ограничивают RAM через cgroup. GPU-память и compute не резервируются; агенты договариваются через доску и ориентируются на лимит «1–2 тяжёлых job'а».

### 5.1. По задачам

| Задача | systemd runs | Mem limit median, G | Mem limit max, G | OMP max | OMP >4 | Bare heavy | Background | GPU procs max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| billion-word | 186 | 8.0 | 55.0 | 8 | 50 | 25 | 10 | 2 |
| bms-mol | 85 | 8.0 | 20.0 | 8 | 14 | 8 | 3 | 1 |
| cassava | 65 | 8.0 | 16.0 | 4 | 0 | 10 | 33 | 2 |
| mlsp-birds | 210 | 8.0 | 24.0 | 8 | 2 | 4 | 147 | 1 |
| nfl-contact | 218 | 16.0 | 56.0 | 48 | 78 | 1 | 7 | 0 |
| spooky | 105 | 8.0 | 30.0 | 12 | 21 | 2 | 48 | 2 |
| stanford-covid | 154 | 8.0 | 24.0 | 8 | 16 | 4 | 10 | 4 |
| taxi | 156 | 6.0 | 26.0 | 16 | 27 | 30 | 5 | None |
| uw-gi | 119 | 12.0 | 28.0 | 8 | 9 | 18 | 54 | 2 |
| ventilator | 99 | 10.0 | 16.0 | 4 | 0 | 10 | 24 | 5 |

### 5.2. По агентам (топ-10 по числу systemd-запусков)

| Задача | Агент | systemd runs | Mem limit median, G | OMP max | Active snapshots | OOM |
|---|---|---:|---:|---:|---:|---:|
| mlsp-birds | rigel | 100 | 12.0 | 4 | 2 | 0 |
| billion-word | deneb | 86 | 8.0 | 8 | 1 | 0 |
| nfl-contact | sirius | 79 | 12.0 | 24 | 4 | 0 |
| nfl-contact | fomalhaut | 77 | 24.0 | 48 | 7 | 0 |
| taxi | achernar | 76 | 6.0 | 16 | 2 | 0 |
| spooky | deneb | 71 | 8.0 | 12 | 1 | 0 |
| billion-word | aldebaran | 65 | 14.0 | 8 | 2 | 0 |
| nfl-contact | procyon | 62 | 24.0 | 16 | 1 | 0 |
| mlsp-birds | altair | 61 | 8.0 | 8 | 1 | 0 |
| stanford-covid | fomalhaut | 60 | 8.0 | 8 | 0 | 0 |

Полная таблица по 30 агентам — в `resource_agents.csv`.

## 6. Конфликты и OOM

Найдено **2** событий OOM/kill/CUDA error в логах (без строк промпта).

| Задача | Агент | Событие |
|---|---|---|
| cassava | achernar | `Note: I bumped my scope to `MemoryMax=11G` because Swin-Base@384 + workers needs ~5-6G RSS and the 8G cap OOM-killed the earlier attempt (main RSS was only ~2G,` |
| cassava | arcturus | `Note: I bumped my scope to `MemoryMax=11G` because Swin-Base@384 + workers needs ~5-6G RSS and the 8G cap OOM-killed the earlier attempt (main RSS was only ~2G,` |

## 7. Что это значит

1. **GPU — общий, без партиционирования.** В 46 снимках с информацией о процессах 21 раз GPU был idle, 9 раз — один процесс, 14 раз — два, 2 раза — три и более. То есть при активной работе агенты обычно делят GPU между 1–2 job'ами, иногда 3+. `CUDA_VISIBLE_DEVICES` не использовался ни разу.

2. **RAM ограничивали через cgroup, CPU — через OMP/CPUQuota.** Это главный механизм изоляции: `systemd-run` с MemoryMax/High встречается часто, `OMP_NUM_THREADS` задаётся, но не всегда ≤4.

3. **Координация — социальная, не техническая.** Агенты пишут на доску о тяжёлых запусках и стараются не запускать больше 1–2 job'ов; жёсткого арбитража нет.

4. **CPU и RAM не были узким местом.** Медианная утилизация CPU — 0.6–10.4% от 128 ядер, RAM — 16–63 GB из 503. Узкое место — GPU и дисциплина запусков, а не нехватка CPU/RAM.

5. **Политика соблюдалась частично.** cgroup-лимиты использовались массово (1389 запусков), но 217 запусков имели `OMP_NUM_THREADS > 4`, 112 тяжёлых python-запусков шли без `systemd-run`, и зафиксировано 2 OOM-события.

## 8. Ограничения

- Снимки `nvidia-smi/ps/free` редкие и не имеют точных timestamp; это выборочные моменты, а не непрерывная телеметрия.

- CPU/RAM атрибуция неполная: процессы без пути `workspaces/<agent>` попадают в unknown; `ps` часто фильтруется grep'ом и не видит все процессы.

- OOM-события ищутся по тексту логов; часть может быть проглочена или не попасть в вывод.

- Один прогон на задачу; выводы описательные.

## 9. Файлы

```
analysis/
├── RESOURCES.md
├── resource_tasks.csv
├── resource_agents.csv
├── resource_snapshots.csv
└── figures/
    ├── resource_fig1_gpu.png
    ├── resource_fig2_cpu_ram.png
    └── resource_fig3_concurrency.png
```
