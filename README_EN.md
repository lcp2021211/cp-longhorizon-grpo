# Agentic GRPO Long-Horizon

Train a long-horizon tool agent on the current **tau2-bench airline** Gym environment with three orthogonal components:

- **SALT** refines trajectory-level GRPO advantages into assistant-step advantages for mixed-outcome groups.
- **ProGPO** supplies a weak first-visit progress fallback only when every trajectory in a group fails.
- **LATA** transmits step/trajectory advantages with real assistant-turn weights and `1/sqrt(L)` policy-token scaling.

[中文](README.md) · [Detailed Chinese training guide](agentic-grpo-longhorizon/docs/training_and_ablation_guide.md) · [Implementation notes](agentic-grpo-longhorizon/docs/salt_progpo_lata_tau2.md)

> Status: the tau2 adapter, rollout trace, combined estimator, diagnostics, full `2^3` ablation suite, and CPU tests are implemented. This repository does not claim a new Qwen2.5-7B tau2 score before a full GPU run is completed. Historical 50-task `tau-bench` results are not directly comparable.

## Method

```text
8 online rollouts for one tau2 task
  ├─ mixed terminal outcomes
  │    GRPO trajectory advantage -> optional SALT step assignment
  ├─ all failed
  │    optional ProGPO progress fallback (SALT is bypassed)
  └─ all succeeded, or progress is degenerate
       zero group-relative advantage

step/trajectory advantage
  -> optional real-turn LATA / sqrt(policy token count)
  -> PPO clipped policy update
```

SALT merges only identical `(s_prev, action, s_next)` transitions, where a state contains the latest `h=3` action-observation pairs. Repeated occurrences receive the arithmetic mean of their source trajectory advantages; unique and divergent transitions retain their original trajectory advantage.

Tool actions and observations use deterministic canonical keys. Free-form text uses conservative exact matching, so `salt/merge_rate` is an essential health metric.

## Online tau2 data flow

The pinned current tau2 revision provides the official airline splits:

| Parquet | Rows | Purpose |
|---|---:|---|
| `agentic-grpo-longhorizon/experiments/tau2/train.parquet` | 30 | official `train` tasks |
| `agentic-grpo-longhorizon/experiments/tau2/val.parquet` | 20 | official `test` tasks |

Each row stores only a task ID and minimal prompt metadata. `AgentGymEnv.reset()` supplies the domain policy and first simulated-user message online. Training tool schemas are generated from current tau2 and kept in `agentic-grpo-longhorizon/configs/tool_config/tau_bench_airline_tools.yaml`; trajectories remain fully online.

The default `train_batch_size=4` and `rollout.n=8` produce four independent task groups and 32 trajectories per rollout batch. Advantages are normalized only within each task group; all groups can then update the same shared policy in one mini-batch.

## Full-factorial ablation

All variants use one estimator and one common training configuration. Only three booleans change:

| ID | SALT | ProGPO | LATA |
|---|:---:|:---:|:---:|
| `000_vanilla` | off | off | off |
| `100_salt` | on | off | off |
| `010_progpo` | off | on | off |
| `001_lata` | off | off | on |
| `110_salt_progpo` | on | on | off |
| `101_salt_lata` | on | off | on |
| `011_progpo_lata` | off | on | on |
| `111_full` | on | on | on |

The `000` cell is numerically tested against veRL's original trajectory-level GRPO, and `111` against the dedicated combined estimator.

## Quick start

Target environment: Linux, Python 3.12, PyTorch 2.7, NVIDIA CUDA. A local 72B-AWQ user simulator normally needs a GPU separate from the 7B training policy; an OpenAI-compatible remote API can replace it.

```bash
git clone https://github.com/qiqihezh/agentic-grpo-longhorizon.git
cd agentic-grpo-longhorizon

# Omit the variable when using a remote user-simulator API.
DOWNLOAD_USER_SIMULATOR=1 bash setup.sh
conda activate agentrl
```

The default warm-start path, `agentic-grpo-longhorizon/experiments/sft_lora_merged`, is not included and is not created by setup. Supply an absolute path to your merged SFT model or the downloaded base model, and use the same model for every ablation cell:

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-7B-Instruct
test -f "$MODEL_PATH/config.json"
```

Start a local simulator:

```bash
MODEL_PATH=/absolute/path/to/Qwen2.5-72B-Instruct-AWQ \
CUDA_DEVICES=1 PORT=8001 \
bash agentic-grpo-longhorizon/scripts/vllm_server/72b.sh

curl -fsS http://localhost:8001/v1/models
```

Or configure a hosted OpenAI-compatible endpoint:

```bash
export TAU2_USER_MODEL='provider-model-name'
export TAU2_USER_PROVIDER='openai'
export TAU2_USER_BASE_URL='https://your-endpoint.example/v1'
export OPENAI_API_KEY='your-key'
```

Inspect and dry-run the matrix:

```bash
bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh --list

bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path "$MODEL_PATH" \
  --dry-run
```

Run the full method or all eight cells sequentially:

```bash
# Full method only
bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh \
  --experiments 111_full \
  --model-path "$MODEL_PATH"

# Complete matrix
bash agentic-grpo-longhorizon/scripts/train/grpo/run_ablation_matrix.sh \
  --experiments all \
  --model-path "$MODEL_PATH"
```

The runner also supports comma/space-separated subsets, `--steps`, `--skip-data`, `--rebuild-data`, `--resume-mode auto|disable`, and `--continue-on-error`. It builds the shared parquet files once, executes variants sequentially, auto-resumes incomplete checkpoints, and skips runs that already reached the requested target step.

On the first non-dry run, each variant atomically creates a `run_manifest.json`. Before resuming, skipping, and launching every training subprocess, the runner verifies the component switches, all files in local actor/reference models, common/tool/interaction configs, key implementation files, the exact pinned tau2 commit, airline data and active Python import origin, effective simulator-routing environment values, and both parquets. A mismatch, or an old output directory containing artifacts without a manifest, is rejected. The target step is deliberately excluded, so `--steps` may extend an otherwise identical run. A remote Hugging Face ID has no local content hash; use a fixed local absolute path for formal experiments.

Outputs are isolated under:

```text
agentic-grpo-longhorizon/experiments/ablations/<variant>/
├── run_manifest.json
├── checkpoints/
├── logs/training_<timestamp>.log
└── hydra/<timestamp>/
```

## Diagnostics and evaluation

Monitor at least:

- `salt/merge_rate`, `salt/graph_invalid_spans`, `salt/uncovered_token_rate`
- `progpo/all_fail_group_rate`, `progpo/trigger_rate`, `progpo/lambda_effective`
- `lata/metadata_fallback_samples`, `lata/uncovered_token_rate`

In-training evaluation on the official test split is disabled by default (`val_before_train=false`, `test_freq=-1`) to prevent checkpoint selection or tuning on held-out tasks. Evaluate every cell only after its fixed step budget with the same official 20-task setup. In the retained evaluator schema, `pass_hat_1` is the single-sample average success rate; the legacy `pass_at_1` field means at least one success among all `N` samples.

## Tests

```bash
python -m unittest discover \
  -s agentic-grpo-longhorizon/tests \
  -p 'test_*.py' -v

PYTHONPATH="$PWD/verl:$PWD/agentic-grpo-longhorizon" \
python -m pytest -q \
  verl/tests/trainer/ppo/test_progpo_lata.py \
  verl/tests/experimental/agent_loop/test_salt_trace.py \
  agentic-grpo-longhorizon/src/envs/tests/test_progpo_progress.py
```

See the [training and ablation guide](agentic-grpo-longhorizon/docs/training_and_ablation_guide.md) for checkpoint export, independent evaluation, resume details, metrics, and troubleshooting.

## References

- [SALT: EACL 2026 Findings](https://aclanthology.org/2026.findings-eacl.247/)
- [ProGPO: Progress-conditioned Group Policy Optimization](https://arxiv.org/abs/2607.22724)
- [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- [volcengine/verl](https://github.com/volcengine/verl)
