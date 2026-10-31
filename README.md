<div align="center">

# Asymmetric Delta Attention (ADA)

</div>

This repository adds **Asymmetric Delta Attention (ADA)**, a generalisation of [Kimi Delta Attention (KDA)](https://arxiv.org/abs/2510.26692), to the [Preconditioned DeltaNet](https://github.com/ntumm120/preconditioned-deltanet) codebase (commit `7bd7532`). The PKDA codebase bundles its own fork of the incredible `flash-linear-attention` in `3rdparty/flash-linear-attention` and the `flame` training framework in `3rdparty/flame`. We built on Preconditioned DeltaNet's code because it was the easiest to edit into ADA.

ADA replaces the scalar write-strength 

$$\beta_t\in\mathbb{R}$$ 

to key-dimension vector 

$$\boldsymbol{\beta}_t\in\mathbb{R}^{d_k},$$

where $d_k$ is the key dimension. Intuitively, KDA uses a single write strength per token per head, whilst ADA allows the model to control the write strength **independently for each key dimension**.

## Contributions
ADA started as a verbatim copy of the Preconditioned KDA (PKDA) files, and all changes from that copy are in:
- `3rdparty/flash-linear-attention/fla/ops/ada/*.py` for ADA Triton kernels and naive implementations.
- `3rdparty/flash-linear-attention/fla/layers/ada.py` for the ADA attention layer, which can be dropped in place of an existing attention layer.
- `3rdparty/flash-linear-attention/fla/models/ada/*.py` for ADA-based models (the PKDA model with ADA attention layers instead of PKDA).
- `3rdparty/flash-linear-attention/fla/layers/__init__.py` and `fla/models/__init__.py`, where a few lines register ADA.
- `configs/ada_r8_{340M,1B}.json` for the language-model configs.
- `experiments/` for the scripts to train and test the language models (see [Language model experiments](#language-model-experiments)).

## Method
The KDA recurrence can be written as: 

$$\mathbf{S}^\text{KDA}_t = (\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)\mathrm{Diag}(\boldsymbol{\alpha}_t)\mathbf{S}^\text{KDA}_{t-1}+\beta_t\mathbf{k}_t\mathbf{v}_t^\top,$$ 

where, at time step $t$,
- $\mathbf{S}^\text{KDA}_t\in\mathbb{R}^{d_k\times d_v}$ is the attention state matrix,
- $\mathbf{k}_t\in\mathbb{R}^{d_k}$ and $\mathbf{v}_t\in\mathbb{R}^{d_v}$ are the key and value vectors,
- $\boldsymbol{\alpha}_t\in\mathbb{R}^{d_k}$ is a vector that controls the decay rate for each dimension of the state matrix, and 
- $\beta_t\in\mathbb{R}$ is a scalar that controls the write strength for the entire key vector.

**ADA replaces the scalar $\beta_t$ with a vector $\boldsymbol{\beta}_t\in\mathbb{R}^{d_k}$**, giving 

$$\mathbf{S}^\text{ADA}_t = (\mathbf{I}-\mathrm{Diag}(\boldsymbol{\beta}_t)\mathbf{k}_t\mathbf{k}_t^\top)\mathrm{Diag}(\boldsymbol{\alpha}_t)\mathbf{S}^\text{ADA}_{t-1}+\mathrm{Diag}(\boldsymbol{\beta}_t)\mathbf{k}_t\mathbf{v}_t^\top.$$ 

Equivalently, if we define a write-modulated key 

$$\tilde{\mathbf{k}}_t=\boldsymbol{\beta}_t\odot\mathbf{k}_t,$$

where $\odot$ is the element-wise product, then the recurrence becomes 

$$\mathbf{S}^\text{ADA}_t = (\mathbf{I}-\tilde{\mathbf{k}}_t\mathbf{k}_t^\top)\mathrm{Diag}(\boldsymbol{\alpha}_t)\mathbf{S}^\text{ADA}_{t-1}+\tilde{\mathbf{k}}_t\mathbf{v}_t^\top.$$

ADA can reduce to KDA if for all $t$, $\boldsymbol{\beta}_t=\beta_t\mathbf{1}$, where $\mathbf{1}\in\mathbb{R}^{d_k}$ is the all-one vector, hence the claim that ADA is a generalisation of KDA. In practice, we have found that ADA doesn't tend to learn a uniform write strength across key dimensions, which maybe suggests that the additional flexibility is indeed being utilised by the model, although on the other hand could also be a result of optimization difficulties in learning a (perhaps optimal) uniform write strength. 

## Acknowledgements
This repository is built on [Preconditioned DeltaNet](https://github.com/ntumm120/preconditioned-deltanet) by Neehal Tumma, Noel Loo and Daniela Rus, which itself builds on `fla-org/flash-linear-attention` and `fla-org/flame`. ADA **heavily** reuses their PKDA and KDA implementations and the surrounding infrastructure. Thank you to them, to the teams at **Bitdeer** and **Moonshot-AI**, and to all contributors to `flash-linear-attention`, for their open-source contributions, which made this work possible.

And thank you to my tutor, **Dr. Anirbit Mukherjee**, for his guidance, teachings, and support throughout this project.

**Small note**
This is a research codebase, not all paths from KDA have been completely adapted, and there are lines of stale code, and even in the core implementation, there are probably bugs and definitely inefficiencies. 

The original `README.md` from Preconditioned DeltaNet is included as [`PKDA_README.md`](PKDA_README.md).

## Language model experiments for Sebastien

| Priority | Model | `MODEL` | Params at 1B |
|---|---|---|---|
| 1 | ADA r8 | `ada_r8` | 995.3M |
| 2 | KDA | `kda` | 995.2M |
| 3 | PKDA | `precond_kda` | 996.4M |

### Setup

**1. Environment variables**

You will need to change `[a12345bc]` to your own username and of course make sure these env vars are set within each terminal

```bash
REPO=/net/scratch/a12345bc/asymmetric-delta-attention
DATA_ROOT=/net/scratch/a12345bc/datasets/SlimPajama-627B_Reupload
PARTITION=gpuH
ACCOUNT=gpu-fse-ugpgt01 # im not sure what this will be for you
WANDB_PROJECT=ada-1b
```

**2. Clone repo & set up the Python environment & W&B**

```bash
git clone https://github.com/jamieeeeeeeeeeeeeeee/asymmetric-delta-attention.git $REPO
cd $REPO
module load apps/binapps/anaconda3/2024.10
python3.12 -m venv venv
source venv/bin/activate
pip install -r experiments/requirements-lock.txt
pip install --no-deps -e 3rdparty/flash-linear-attention
wandb login  # optional
```

If you skip the weights and bias login `wandb login` then you should run `export WANDB_MODE=offline` otherwise the jobs will still try to log to W&B.

**3. Get the data (~180 GB)**

```bash
cd $REPO
source venv/bin/activate
python experiments/data.py $DATA_ROOT
```

This should print `READY: 32 shards in .../data` when it is finished.

**4. Check the code works with the test scripts**

```bash
cd $REPO
sbatch -p $PARTITION -A $ACCOUNT --export=ALL,VENV=$REPO/venv experiments/pre_test_lm.sbatch ada_r8_1B kda_1B precond_kda_1B
```

Then check the last line of the results file using:

```bash
tail -n 1 $REPO/pre_test_results.*.jsonl
```

It should say `{"kind": "summary", "failures": 0}`.

**5. You are ready!**

Submit the three runs to the Slurm queue via:

```bash
cd $REPO
sbatch -p $PARTITION -A $ACCOUNT -G 8 --export=ALL,VENV=$REPO/venv experiments/train_lm.sbatch ada_r8 1B $DATA_ROOT/data $WANDB_PROJECT
sbatch -p $PARTITION -A $ACCOUNT -G 8 --export=ALL,VENV=$REPO/venv experiments/train_lm.sbatch kda 1B $DATA_ROOT/data $WANDB_PROJECT
sbatch -p $PARTITION -A $ACCOUNT -G 8 --export=ALL,VENV=$REPO/venv experiments/train_lm.sbatch precond_kda 1B $DATA_ROOT/data $WANDB_PROJECT
```

The job progress will appear in `$REPO/lm-train.<jobid>.out` and in W&B (if you set it up) as `ada_r8_1B`, `kda_1B` and `precond_kda_1B`.

**6. If a job finishes before it should**

You will need to run the same command for that job again, and it will restart from the latest checkpoint (which is saved every 5000 steps). Each job requests 4 days, which I believe is the limit on `gpuH`, and so a 1B run will likely need resubmitting at least once.

**7. When a job finishes**

The log should end with `RUNNING DONE!`. The final model will be in `$REPO/exp/<MODEL>_1B/` and should have a `config.json` and `model.safetensors`. If this is the case, then the checkpoints in `$REPO/exp/<MODEL>_1B/checkpoint/` can be deleted, as they take up a lot of space on scratch and that is not polite.
