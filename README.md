# MotionChallenge · mimic-lite

[Author project](https://github.com/EGalahad/mimic-lite) · [Checkpoint](https://drive.google.com/drive/folders/1xRmcOX0l-YIpqxUuCmW4s6Dl0HStbJSL) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-mimic-lite.git
cd MotionChallenge-mimic-lite
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This repository adapts model inference. The pinned root `MotionChallenge-Core`
submodule owns datasets, physics, metrics and verification. See `--help` for options.

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](MotionChallenge-Core/docs/OBSERVATIONS.md).
