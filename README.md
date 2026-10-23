# Motion Challenge · mimic-lite

[mimic-lite](https://github.com/EGalahad/mimic-lite) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-mimic-lite.git
cd motion-challenge-mimic-lite
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.


## Checkpoint source

[EGalahad/mimic-lite](https://github.com/EGalahad/mimic-lite/blob/3253810ce21dbcc05d8f4942a017ea3eaffc6cf1/README.md) · [PPO / policy.onnx](https://drive.google.com/drive/folders/1xRmcOX0l-YIpqxUuCmW4s6Dl0HStbJSL). Author-released checkpoint.

The author README links this PPO Drive folder, whose recorded listing contains the exact policy file ID used by the adapter. The download host is Google Drive, not a separate model publisher.

One `uv run evaluate.py` rolls out each complete motion once, then verifies the trace.
The same trajectory provides success, completion and all six MPBPE values in one
`result.json`, with one source revision. No branch switching is needed.
