# SuaVE
Learning Weight-Variant Latent Causal Models
## Overview

This repository contains an implementation of the following paper:

**[Identifying Weight-Variant Latent Causal Models](https://arxiv.org/abs/2208.14153)** (JMLR 2026)

## Extensions

This work has been extended to applications including:

- **[Identifiable Latent Polynomial Causal Models through the Lens of Change](https://openreview.net/forum?id=ia9fKO1Vjq&noteId=lUddseJaKf)** (ICLR 2024)

- **[Towards Identifiable Latent Additive Noise Models](https://arxiv.org/abs/2403.15711)** (arXiv)

## Citation

If you find this code useful, please cite the related papers.
## Minimal CPU synthetic experiment

This is an executable smoke experiment using the existing
`gen_TCL_data_ortho` generator, `dagVAE`, ELBO, Adam optimizer and gradient
clipping. It does **not** establish the paper's recovery accuracy or reproduce
its complete tables. No original server data or SLURM installation is needed.
The model implementation in `lib/models.py` is unchanged.

Tested on Linux x86_64, Python 3.12, PyTorch 2.14.1+cpu. Create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

From the repository root, generate a small causal dataset and train for exactly
three optimizer steps:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash run \
  --data-args 16_8_3_3_2_1_gauss_lrelu_t_f \
  --batch-size 16 --hidden-dim 16 --depth 2 --lr 0.001 --max-iter 3 --seed 1
```

`run` invokes the original `main.py` locally and forwards all arguments.
`PYTHON=/path/to/python bash run ...` selects another interpreter. CPU is now
the default in both `main.py` and `lib/cmd_utils.py`; `--cuda` opts into CUDA,
and `--preload-gpu` requires `--cuda`. GPU behavior has not been tested here.

The data argument fields are
`nps_ns_dl_dd_nl_s_p_a_u_n`: samples per environment, number of environments,
latent dimension, observation dimension, mixing depth, generator seed, noise
prior, mixing activation, uncentered flag and observation-noise flag.
The command above generates 128 observations, three latent coordinates and
eight one-hot environment labels, using a weight-varying ER DAG (degree 2).
For this existing orthogonal causal generator, use `dd == dl >= 2`, Gaussian
noise, `lrelu` or `sigmoid`, and the final flag `f`. Unsupported observation
noise and rectangular mixing are rejected rather than silently ignored.
`nl` is mixing depth; training `--depth` is a separate model setting.

Generated data are cached under `data/tcl_16_8_3_3_2_1_gauss_lrelu_u_dg2.npz`.
An existing file is reused; remove it to regenerate. Data arguments override
`--file`. With neither option, a larger local default dataset is generated.
To separate generation and training:

```bash
python -m lib.data 16 8 3 3 --n-layers 2 --seed 1 --output-dir data
python main.py --file data/tcl_16_8_3_3_2_1_gauss_lrelu_u_dg2.npz \
  --batch-size 16 --hidden-dim 16 --depth 2 --lr 0.001 --max-iter 3
```

Outputs include TensorBoard events in `runs/`, checkpoints in `ckpt/<id>/`,
and metrics/metadata in `log/`. `--no-log` skips checkpoints and metric-file
persistence, but the existing TensorBoard and experiment-ID logging remains.
Longer runs can use `--epochs`; `--max-iter` takes precedence. The printed
latent metric now compares true latents with latent estimates; noise is
reported separately. Tiny-run metrics are diagnostics, not accuracy claims.

## Lightweight verification

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

The tests check deterministic generation, causal weight variation, shapes,
CPU initialization, finite forward/ELBO/gradients, parameter updates, and
an end-to-end three-step run with checkpoints and metric persistence. They
also check data-argument precedence, device-option validation and launchers.

On 2026-10-08, both tests passed on the environment above, with NumPy 2.2.6,
SciPy 1.15.3, scikit-learn 1.6.1, NetworkX 3.4.2, pandas 2.2.3 and
TensorBoardX 2.6.2.2. The direct model test produced negative ELBO values
194.6461, 217.4137 and 194.2764; stochastic losses need not decrease over
three updates. The CLI completed exactly three steps with finite diagnostic
metrics. Full paper experiments, GPU execution and cluster submission were
not run.

## Legacy SLURM launcher

The previous `run` is preserved as `run_slurm`. Static inspection of upstream
commit `8b06ea339587d4d7142a37465cd0dae9897cd697` found that `args.txt`,
`slurm_script_gpu.sbatch` and `slurm_script_cpu.sbatch` are absent from the
tracked repository. Those files are site-specific and were not reconstructed.
`run_slurm` now reports a missing configuration before generating files or
submitting jobs. Use the local `run` command above for the minimal experiment.
