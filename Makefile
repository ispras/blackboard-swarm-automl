SHELL := /bin/bash

AGENTS ?= 3
HOURS ?= 12
MODEL ?= deepseek/deepseek-flash
PROMPT ?= prompts/PROMPT_SWARM.md
SUBSET ?= subsets/lite10.txt

.PHONY: help build build-min up batch status logs down clean toy-prepare toy-grade analysis shell

help:
	@echo "make build        - build full swarm + tools images"
	@echo "make build-min    - build light images for the offline toy example"
	@echo "make up           - start the swarm (needs a prompt/model; use bin/cli/swarm.sh directly)"
	@echo "make batch        - run the tasks from SUBSET sequentially (default: subsets/lite10.txt)"
	@echo "make status       - show Docker and swarm state"
	@echo "make logs         - follow swarm/grader logs"
	@echo "make down         - stop swarm and grader"
	@echo "make clean        - stop and remove sandbox artifacts (asks for confirmation)"
	@echo "make toy-prepare  - install the offline toy spaceship task"
	@echo "make toy-grade    - grade the toy baseline submission"
	@echo "make analysis     - rebuild sandbox/analysis (graph, metrics, costs)"
	@echo "make shell        - open a shell in the swarm container"

build:
	bin/cli/swarm.sh build

build-min:
	SWARM_TARGET=swarm-min SWARM_TOOLS_TARGET=tools-min bin/cli/swarm.sh build
	@echo "hint: for subsequent commands set SWARM_TARGET=swarm-min and SWARM_TOOLS_TARGET=tools-min in .env"

up:
	bin/cli/swarm.sh up --agents $(AGENTS) --hours $(HOURS) --model $(MODEL) --prompt $(PROMPT)

batch:
	bin/cli/swarm.sh batch --subset $(SUBSET) \
		--agents $(AGENTS) --hours $(HOURS) --model $(MODEL) --prompt $(PROMPT)

status:
	bin/cli/swarm.sh status

logs:
	bin/cli/swarm.sh logs

down:
	bin/cli/swarm.sh stop

clean:
	bin/cli/clean.sh

toy-prepare:
	SWARM_TARGET=swarm-min SWARM_TOOLS_TARGET=tools-min \
		bin/cli/prepare_data.sh install -c spaceship-titanic \
		--source examples/toy-spaceship --skip-verification

toy-grade:
	SWARM_TARGET=swarm-min SWARM_TOOLS_TARGET=tools-min \
		bin/cli/grade.sh --submission examples/toy-spaceship/submission.csv

analysis:
	bin/cli/analysis.sh

shell:
	bin/cli/swarm.sh shell
