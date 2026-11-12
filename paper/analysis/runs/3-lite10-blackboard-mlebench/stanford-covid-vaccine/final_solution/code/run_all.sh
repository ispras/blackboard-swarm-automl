#!/bin/bash
# Reproduce base models (see run_cv.sh / run_seed.sh / run_v2.sh), then:
python aggregate.py && python ensemble2.py && python make_final.py --write
