# Toy Spaceship (offline MLE-bench example)

A fully synthetic 240-row train / 60-row test dataset whose columns match the
real MLE-bench `spaceship-titanic` competition. It exists so the Docker harness
can be exercised end-to-end without Kaggle credentials or a multi-GB dataset.

* `public/train.csv` — training rows with the `Transported` target.
* `public/test.csv` — test rows without the target.
* `public/sample_submission.csv` — submission shape.
* `private/test.csv` — synthetic test labels used by the local grader.
* `submission.csv` — a trivial `CryoSleep` baseline (~0.75 accuracy) for smoke tests.

Install it into the sandbox:

```bash
bin/cli/prepare_data.sh install -c spaceship-titanic \
  --source examples/toy-spaceship --skip-verification
```

The checksum verification is intentionally skipped: this is not the real Kaggle
data. Do not use this directory as a benchmark result.
