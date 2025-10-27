# tinyzero_modal.py
# A minimal, clean TinyZero-style RL runner for Countdown on Modal.
# - GRPO via VERL
# - Qwen2.5 3B (non-reasoning), LoRA
# - Two A100-80GB GPUs
# - Weights & Biases logging + a tiny live dashboard
# - Reads Countdown parquet(s) from a Modal Volume
#
# Prereqs (one-time, from your laptop):
#   modal volume create hf-cache
#   modal volume create rl-outputs
#   modal volume create countdown-data
#   modal volume put countdown-data ./path/to/*.parquet
#   modal secret create wandb-secret WANDB_API_KEY=your_key_here
#
# Deploy:
#   modal deploy tinyzero_modal.py
#
# Kick off a run (GRPO):
#   curl -X POST https://<your-app-url>/v1/rl/train -H "content-type: application/json" -d '{
#     "parquet_glob": "/data/*.parquet",
#     "algo": "grpo",
#     "train_args": {
#       "epochs": 1,
#       "batch_size": 2,
#       "grad_accum": 8,
#       "lr": 1e-6,
#       "logging_steps": 10,
#       "num_generations": 8,
#       "max_prompt_len": 512,
#       "max_completion_len": 256,
#       "loss_type": "dapo",
#       "scale_rewards": "group",
#       "beta": 0.0
#     }
#   }'
#
# Open live views:
#   - Mini dashboard:   https://<your-app-url>/ui/miniboard/<job_id>
#   - Weights & Biases:  https://wandb.ai/[your-username]/tinyzero-rl  (run "tinyzero-<job_id>")
#
# Notes:
# - This function requests 2x A100-80GB. VERL uses Hydra-based configs and supports
#   distributed training via torchrun. The setup here emphasizes simplicity and
#   logging to reproduce the “aha” curve.

import os, json, time, uuid, glob, tarfile
from typing import Any, Dict
import modal
from rich import print as rprint
from rich.pretty import pprint

# ---------- Modal image & volumes ----------
image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install(["git"])  # base image already has CUDA/PyTorch stack
    .uv_pip_install(
        # Web/API
        "fastapi>=0.112.2",
        "uvicorn==0.30.6",
        # HF stack (present in base but ensure versions)
        "transformers>=4.43.0",
        "datasets>=4.0.0",
        "accelerate>=0.33.0",
        "huggingface_hub[hf_transfer]>=0.24.5",
        # VERL + vLLM (pin as per docs)
        "verl[vllm]==0.4.1",
        # data + metrics
        "pandas>=2.2.2",
        "pyarrow>=16.1.0",
        "scipy>=1.13.1",
        "wandb>=0.17.0",
        "weave",
        "rich",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("tinyzero-rl")

HF_VOL = modal.Volume.from_name("hf-cache", create_if_missing=True)
OUT_VOL = modal.Volume.from_name("rl-outputs", create_if_missing=True)
DATA_VOL = modal.Volume.from_name("countdown-data", create_if_missing=True)

# ---------- Shared config ----------
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # non-reasoning instruct at ~3B
OUT_ROOT = "/outputs/artifacts"
DATA_DIR = "/data"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================
# 1) TRAINING WORKER (GPU job) — writes TB logs + artifacts
# ============================================================
@app.function(
    image=image,
    gpu="H200",
    volumes={
        "/root/.cache/huggingface": HF_VOL,
        "/outputs": OUT_VOL,
        "/data": DATA_VOL,
    },
    timeout=60 * 60 * 24,
    scaledown_window=5,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_rl_job(
    job_id: str,
    model_name: str,
    parquet_glob: str,
    algo: str,
    train_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Launched by the API server; performs RL via VERL (GRPO) and writes:
      - Weights & Biases logging to project "tinyzero-rl" with run "tinyzero-<job_id>"
      - Tarball of artifacts under /outputs/artifacts/<job_id>.tar.gz
    """
    import subprocess

    os.makedirs(OUT_ROOT, exist_ok=True)

    run_name = f"tinyzero-{job_id}"
    out_dir = os.path.join(OUT_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    # Ensure Weights & Biases env for VERL (VERL initializes W&B internally)
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in environment variables")
    # Propagate explicit entity/project/name so VERL's wandb.init uses correct workspace
    os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY") or "sambhavg"
    os.environ.setdefault("WANDB_PROJECT", "tinyzero-rl")
    os.environ["WANDB_NAME"] = run_name

    # --- dataset visibility from Modal volume mount at /data ---
    try:
        DATA_VOL.reload()
    except Exception:
        pass
    rprint("[bold green]Parquet glob:[/bold green]")
    pprint(parquet_glob)
    paths = sorted(glob.glob(parquet_glob))
    rprint("[bold green]Paths:[/bold green]")
    pprint(paths)

    # Verify presence of train/val
    has_train = any("train" in p for p in paths)
    has_val = any(("test" in p) or ("val" in p) for p in paths)
    if not has_train:
        raise FileNotFoundError("No train parquet found in provided glob")
    if not has_val:
        raise FileNotFoundError("No test/val parquet found in provided glob")

    # --- map args and invoke VERL GRPO ---
    train_file = next(p for p in paths if "train" in p)
    val_file = next(p for p in paths if ("test" in p) or ("val" in p))

    epochs = int(float(train_args.get("epochs", 1)))
    train_bs = int(train_args.get("batch_size", 64))
    val_bs = int(train_args.get("val_batch_size", max(1, train_bs)))
    lr = float(train_args.get("lr", 1e-6))
    max_prompt_len = int(train_args.get("max_prompt_len", 256))
    max_completion_len = int(train_args.get("max_completion_len", 1024))
    grpo_mini_bsz = int(train_args.get("grpo_mini_batch_size", 64))
    grpo_micro_bsz = int(train_args.get("grpo_micro_batch_size", 8))
    log_prob_micro_bsz = int(train_args.get("log_prob_micro_batch_size", 8))
    rollout_tp = int(train_args.get("tensor_model_parallel_size", 1))
    gpu_mem_util = float(train_args.get("gpu_memory_utilization", 0.4))

    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={train_bs}",
        f"data.max_prompt_length={max_prompt_len}",
        f"data.max_response_length={max_completion_len}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={model_name}",
        f"actor_rollout_ref.actor.optim.lr={lr}",
        "actor_rollout_ref.model.use_remove_padding=False",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={grpo_mini_bsz}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={grpo_micro_bsz}",
        "actor_rollout_ref.actor.checkpoint.save_contents='model,optimizer,extra,hf_model'",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tp}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={log_prob_micro_bsz}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_mem_util}",
        "actor_rollout_ref.rollout.n=5",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console', 'wandb']",
        "trainer.project_name=tinyzero-rl",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.test_freq=5",
        f"trainer.default_local_dir={out_dir}",
        "trainer.resume_mode=auto",
        f"trainer.save_freq={max(1, epochs)}",
        f"trainer.total_training_steps={max(1, train_args.get('max_steps', 0))}",
        f"trainer.total_epochs={epochs}",
        # "actor_rollout_ref.model.dtype=bfloat16",
        # "critic.model.dtype=bfloat16",
        "actor_rollout_ref.rollout.dtype=bfloat16",
        # "actor_rollout_ref.ref.dtype=bfloat16",
        # "trainer.precision=bfloat16",
    ]

    rprint("[bold green]Launching VERL GRPO...[/bold green]")
    proc = subprocess.run(cmd, check=False)

    # tarball artifact
    tar_path = os.path.join(OUT_ROOT, f"{job_id}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=os.path.basename(out_dir))

    # record manifest
    manifest = {
        "job_id": job_id,
        "model": model_name,
        "algo": "grpo",
        "tb_dir": tb_dir,
        "artifact_dir": out_dir,
        "artifact_tar": tar_path,
        "status": "ok" if proc.returncode == 0 else "error",
        "return_code": proc.returncode,
        "time_finished": _now(),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ============================================================
# 2) API SERVER (ASGI) — kicks jobs, serves TB & mini dashboard
# ============================================================
@app.function(
    image=image,
    # CPU is fine here; training happens in run_rl_job().
    volumes={
        "/root/.cache/huggingface": HF_VOL,
        "/outputs": OUT_VOL,
        "/data": DATA_VOL,
    },
    timeout=60 * 60,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="TinyZero RL (Modal)")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"]
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "time": _now()}

    @app.post("/v1/rl/train")
    async def start_train(req: Request):
        body = await req.json()
        model_name = body.get("model") or DEFAULT_MODEL
        parquet_glob = body.get("parquet_glob") or f"{DATA_DIR}/*.parquet"
        algo = (body.get("algo") or "grpo").lower()
        train_args = body.get("train_args", {})
        job_id = body.get("job_id") or uuid.uuid4().hex

        # spawn background GPU job
        # (logs are written to shared volume, visible to TensorBoard mount)
        run = run_rl_job.spawn(
            job_id=job_id,
            model_name=model_name,
            parquet_glob=parquet_glob,
            algo=algo,
            train_args=train_args,
        )
        return JSONResponse(
            {
                "ok": True,
                "job_id": job_id,
                "wandb_project": "tinyzero-rl",
                "wandb_run": f"tinyzero-{job_id}",
                "mini_dashboard": f"/ui/miniboard/{job_id}",
                "note": "Monitor training at https://wandb.ai/[your-username]/tinyzero-rl/runs/tinyzero-{}".format(
                    job_id
                ),
                "spawned_call_id": run.object_id,
                "time": _now(),
            }
        )

    @app.get("/ui/miniboard/{job_id}")
    def miniboard(job_id: str):
        # Simple client-side polling view; Weights & Biases is the main UI.
        html = f"""
<!doctype html>
<html><head><meta charset="utf-8"/>
<title>RL MiniBoard — {job_id}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:24px}}
.row{{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:16px}}
canvas{{width:100% !important; height:260px !important}}
code{{background:#111;color:#eee;padding:2px 6px;border-radius:4px}}
</style></head>
<body>
<h2>Live training — <code>{job_id}</code></h2>
<p>Monitor training on <a href="https://wandb.ai/[your-username]/tinyzero-rl/runs/tinyzero-{job_id}" target="_blank">Weights & Biases</a>.</p>
<p>This mini board is a simple static view - use Weights & Biases for detailed metrics and visualizations.</p>
<div class="row">
  <canvas id="pct"></canvas>
  <canvas id="len"></canvas>
  <canvas id="aha"></canvas>
  <canvas id="tta"></canvas>
</div>
<script>
// Note: Charts are static placeholders. Visit Weights & Biases for live metrics.
// You can implement custom polling logic here if needed for specific use cases.
console.log("Training job:", "{job_id}");
console.log("View metrics at: https://wandb.ai/[your-username]/tinyzero-rl/runs/tinyzero-{job_id}");
</script>
</body></html>
"""
        return HTMLResponse(html)

    return app
