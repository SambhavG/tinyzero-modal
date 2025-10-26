# tinyzero_modal.py
# A minimal, clean TinyZero-style RL runner for Countdown on Modal.
# - GRPO (default) or PPO via "algo" switch
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
# - This function requests 2x A100-80GB. TRL uses 🤗 Accelerate under the hood.
#   For true multi-GPU DDP scaling, launch distributed (accelerate/torchrun).
#   The setup here emphasizes simplicity and logging to reproduce the “aha” curve.

import os, re, io, json, time, uuid, glob, tarfile, asyncio, fnmatch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import modal
from rich import print as rprint
from rich.pretty import pprint

# ---------- Modal image & volumes ----------
image = (
    modal.Image.debian_slim(
        python_version="3.13"
    )  # Updated to 3.13 for newer kernel support
    .apt_install("git")
    .uv_pip_install(
        # web
        "fastapi==0.112.2",
        "uvicorn==0.30.6",
        # hf/rl stack
        "transformers>=4.43.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "huggingface_hub[hf_transfer]>=0.24.5",
        "torch>=2.3.0",
        "peft>=0.11.1",
        "safetensors>=0.4.3",
        "trl>=0.23.0",
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
    gpu="B200",
    volumes={
        "/root/.cache/huggingface": HF_VOL,
        "/outputs": OUT_VOL,
        "/data": DATA_VOL,
    },
    timeout=60 * 60,
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
    Launched by the API server; performs RL (GRPO or PPO) and writes:
      - Weights & Biases logging to project "tinyzero-rl" with run "tinyzero-<job_id>"
      - PEFT adapter + tokenizer under /outputs/artifacts/<job_id>
      - Tarball of artifacts under /outputs/artifacts/<job_id>.tar.gz
    """
    import math
    import pandas as pd
    import torch
    import torch.nn as nn
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer, PPOConfig, PPOTrainer
    import wandb
    import weave

    os.makedirs(OUT_ROOT, exist_ok=True)

    run_name = f"tinyzero-{job_id}"
    out_dir = os.path.join(OUT_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = os.path.join(out_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)

    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    else:
        raise ValueError("WANDB_API_KEY not found in environment variables")

    wandb_entity = os.getenv("WANDB_ENTITY", "sambhavg")  # fallback to default
    wandb.init(
        entity=wandb_entity,
        project="tinyzero-rl",
        name=run_name,
        config={"job_id": job_id, "model_name": model_name, "algo": algo, **train_args},
    )

    # --- dataset: load Countdown parquet(s) from Modal volume mount at /data ---
    # Ensure we see the latest committed snapshot of the dataset volume
    try:
        DATA_VOL.reload()
    except Exception:
        # Safe to proceed; initial mount already contains a snapshot
        pass
    rprint("[bold green]Parquet glob:[/bold green]")
    pprint(parquet_glob)
    paths = sorted(glob.glob(parquet_glob))
    rprint("[bold green]Paths:[/bold green]")
    pprint(paths)
    # dict from path to df
    dfs = {p: pd.read_parquet(p) for p in paths}
    # df is the df for which "train" is in the key
    df = dfs[[k for k in dfs.keys() if "train" in k][0]]
    # Optionally limit dataset size for faster iterations
    max_train_samples = int(train_args.get("max_train_samples", 0) or 0)
    if max_train_samples and len(df) > max_train_samples:
        seed_val = int(train_args.get("shuffle_seed", 42))
        df = df.sample(n=max_train_samples, random_state=seed_val).reset_index(
            drop=True
        )
    rprint("[bold blue]DF head:[/bold blue]")
    pprint(df.head())
    rprint(f"[bold blue]DF size used:[/bold blue] {len(df)} rows")

    # The parquet schema (as provided) includes:
    #  target:int, nums:list[int], prompt:(list-of-dicts or JSON str), reward_model:dict, extra_info:dict{index,split}, ...
    # We'll yield TRL's expected "prompt" column (chat messages) + ground_truth + id_
    def row_to_prompt(row):
        pr = row.get("prompt")
        # Use prompt from parquet directly when present
        if isinstance(pr, list):
            return pr
        # Handle numpy arrays (which may be stored as arrays in parquet)
        if hasattr(pr, "tolist"):  # numpy array
            pr_list = pr.tolist()
            if isinstance(pr_list, list):
                return pr_list
        if isinstance(pr, str):
            try:
                parsed = json.loads(pr)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except Exception:
                # It's a plain text prompt; wrap as single user message
                return [{"role": "user", "content": pr}]
        # Fallback only if prompt is missing/None
        try:
            t = int(row.get("target")) if row.get("target") is not None else 0
            nums = None
            if isinstance(row, dict):
                if "nums" in row and row.get("nums") is not None:
                    nums = row.get("nums")
                elif "numbers" in row and row.get("numbers") is not None:
                    nums = row.get("numbers")
            if hasattr(nums, "tolist"):
                nums = nums.tolist()
            if nums is None:
                nums = []
            nums = list(nums)
        except Exception:
            t, nums = 0, []
        txt = (
            "You are playing Countdown (numbers game).\n"
            f"Numbers: {nums}\nTarget: {t}\n"
            "Produce a valid arithmetic expression using only these numbers each at most once.\n"
            "At the end, output the final integer on a new line as 'Answer: <int>'."
        )
        return [{"role": "user", "content": txt}]

    def row_to_ground_truth(row):
        # Prefer reward_model.ground_truth when available
        rm = row.get("reward_model")
        gt = None
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
        elif isinstance(rm, str):
            try:
                rm_parsed = json.loads(rm)
                if isinstance(rm_parsed, dict):
                    gt = rm_parsed.get("ground_truth")
            except Exception:
                pass
        target_val = None
        numbers_val = None
        if isinstance(gt, dict):
            target_val = gt.get("target")
            if "numbers" in gt and gt.get("numbers") is not None:
                numbers_val = gt.get("numbers")
            elif "nums" in gt and gt.get("nums") is not None:
                numbers_val = gt.get("nums")
        # Fallback to top-level columns if needed
        if target_val is None:
            target_val = row.get("target")
        if numbers_val is None:
            if "nums" in row and row.get("nums") is not None:
                numbers_val = row.get("nums")
            elif "numbers" in row and row.get("numbers") is not None:
                numbers_val = row.get("numbers")
        # Coerce types
        if hasattr(numbers_val, "tolist"):
            numbers_val = numbers_val.tolist()
        if numbers_val is None:
            numbers_val = []
        try:
            numbers_val = [int(x) for x in list(numbers_val)]
        except Exception:
            numbers_val = []
        try:
            target_val = int(target_val) if target_val is not None else None
        except Exception:
            target_val = None
        return {"target": target_val, "numbers": numbers_val}

    ids, prompts, gts = [], [], []
    for i, r in df.iterrows():
        idx = (
            r.get("extra_info", {}).get("index", i)
            if isinstance(r.get("extra_info"), dict)
            else i
        )
        ids.append(int(idx))
        prompts.append(row_to_prompt(r))
        gts.append(row_to_ground_truth(r))

    # Converting dict to Dataset
    rprint("[bold green]Converting dict to Dataset...[/bold green]")
    ds = Dataset.from_dict({"prompt": prompts, "ground_truth": gts, "id_": ids})

    # --- model + tokenizer ---
    rprint("[bold green]Loading tokenizer...[/bold green]")
    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )
    tok.padding_side = "left"  # TRL expects left padding for decoder-only

    # Configure tokenizer tokens properly for Qwen models
    # This prevents the "new PAD/BOS/EOS tokens" warning
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
        # Explicitly set pad_token_id to match
        if tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

    # Ensure model config is aligned with tokenizer
    # This is especially important for Qwen models

    # Enable faster matmul paths on A100 (no accuracy loss for bf16)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Try enabling flash and mem-efficient SDP kernels if available
        try:
            # Available in torch 2.0+
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

    rprint("[bold green]Loading model...[/bold green]")
    use_flash_attn = bool(train_args.get("use_flash_attn", False))
    attn_pref = "flash_attention_2" if use_flash_attn else "sdpa"
    load_kwargs = dict(
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=None,  # Trainer/Accelerate manages placement
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_pref,
            **load_kwargs,
        )
    except Exception:
        # Safe fallback if FlashAttention-2 isn't available
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="sdpa",
            **load_kwargs,
        )

    # Sync model config with tokenizer to prevent token mismatch warnings
    if hasattr(base, "config"):
        if tok.pad_token_id is not None:
            base.config.pad_token_id = tok.pad_token_id
        if tok.bos_token_id is not None:
            base.config.bos_token_id = tok.bos_token_id
        if tok.eos_token_id is not None:
            base.config.eos_token_id = tok.eos_token_id

    # --- pick LoRA targets ---
    def guess_lora_targets(model: nn.Module) -> List[str]:
        preferred = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "c_attn",
            "c_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "Wqkv",
            "Wq",
            "Wk",
            "Wv",
            "Wo",
            "Wup",
            "Wdown",
        }
        leaf_linear_names, preferred_hits = set(), set()
        linear_like = (nn.Linear,)
        linear_like_names = {"Linear8bitLt", "Linear4bit"}
        for name, module in model.named_modules():
            leaf = name.split(".")[-1]
            cls = module.__class__.__name__
            if not (isinstance(module, linear_like) or cls in linear_like_names):
                continue
            leaf_linear_names.add(leaf)
            if (leaf in preferred) or any(k in leaf for k in preferred):
                preferred_hits.add(leaf)
        return sorted(preferred_hits or leaf_linear_names)

    # Gradient checkpointing to reduce memory, speed depends on compute/memory tradeoff
    if bool(train_args.get("gradient_checkpointing", True)):
        try:
            base.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(base, "config"):
            try:
                base.config.use_cache = False
                base.config.gradient_checkpointing = True
            except Exception:
                pass

    # Torch compile base model
    base = torch.compile(base)

    rprint("[bold green]Loading LoRA config...[/bold green]")
    lora_r = int(train_args.get("lora_r", 32))
    lora_alpha = int(train_args.get("lora_alpha", 64))
    lora_dropout = float(train_args.get("lora_dropout", 0.05))
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=guess_lora_targets(base),
    )

    # --- metrics tracking ("aha"/"wait") ---
    step_counter = {"k": 0}
    seen_correct_once: Dict[int, bool] = {}
    first_hit_step: Dict[int, int] = {}

    int_pat = re.compile(r"(-?\d+)")

    def extract_final_int(s: str) -> Optional[int]:
        m = list(int_pat.finditer(s))
        if not m:
            return None
        try:
            return int(m[-1].group(1))
        except Exception:
            return None

    def time_to_answer_chars(s: str) -> int:
        # proxy "wait": chars until "Answer:" or first final int line
        pos = s.find("Answer:")
        if pos >= 0:
            return pos
        # fallback: first number occurrence
        m = int_pat.search(s)
        return m.start() if m else len(s)

    # Reward #1: correctness (1/0 based on final integer)
    def reward_correctness(completions, ground_truth, **kw):
        outs = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        tgts = [
            gt.get("target") if isinstance(gt, dict) else None for gt in ground_truth
        ]
        corr, lengths, tta = [], [], []
        for s, tgt in zip(outs, tgts):
            ans = extract_final_int(s)
            ok = (
                1.0
                if (ans is not None and tgt is not None and ans == int(tgt))
                else 0.0
            )
            corr.append(ok)
            lengths.append(len(s))
            tta.append(time_to_answer_chars(s))
        # log batch stats
        step_counter["k"] += 1
        bstep = step_counter["k"]
        wandb.log(
            {
                "metrics/pct_correct_batch": sum(corr) / max(1, len(corr)),
                "gen/avg_output_len_chars": sum(lengths) / max(1, len(lengths)),
                "gen/avg_time_to_answer_chars": sum(tta) / max(1, len(tta)),
                "step": bstep,
            }
        )
        # "aha": first time a prompt becomes correct
        b = kw.get("batch", {})
        ids = list(b.get("id_", [])) if isinstance(b, dict) else []
        aha = 0.0
        for pid, ok in zip(ids, corr):
            prior = seen_correct_once.get(pid, False)
            if (ok == 1.0) and (prior is False):
                aha += 1.0
                seen_correct_once[pid] = True
                if pid not in first_hit_step:
                    first_hit_step[pid] = bstep
        if len(ids) > 0:
            log_data = {"signals/aha_rate": aha / len(ids), "step": bstep}
            if len(first_hit_step) >= 3:
                log_data["signals/avg_first_hit_step"] = sum(
                    first_hit_step.values()
                ) / len(first_hit_step)
            wandb.log(log_data)

        # Log up to 10 example generations after each step
        try:
            max_rows = 10
            rows = []

            def _coerce_nums(gt):
                if not isinstance(gt, dict):
                    return []
                nums = (
                    gt.get("numbers")
                    if gt.get("numbers") is not None
                    else gt.get("nums")
                )
                if hasattr(nums, "tolist"):
                    nums = nums.tolist()
                try:
                    return list(nums) if nums is not None else []
                except Exception:
                    return []

            for j in range(min(max_rows, len(outs))):
                pid = ids[j] if j < len(ids) else None
                tgt = None
                nums = []
                if j < len(ground_truth) and isinstance(ground_truth[j], dict):
                    tgt = ground_truth[j].get("target")
                    nums = _coerce_nums(ground_truth[j])
                rows.append([pid, tgt, nums, outs[j]])
            if rows:
                table = wandb.Table(
                    columns=["id_", "target", "numbers", "generation"], rows=rows
                )
                wandb.log({"samples/generations": table, "step": bstep})
        except Exception:
            pass
        return corr  # primary reward

    # Reward #2: brevity shaping (tiny negative for verbosity)
    def reward_brevity(completions, **kw):
        outs = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
        return [-0.001 * max(0, len(s)) for s in outs]

    # --- training args & algo switch ---
    rprint("[bold green]Loading training args...[/bold green]")
    epochs = float(train_args.get("epochs", 1.0))
    per_device_bs = int(train_args.get("batch_size", 2))
    grad_accum = int(train_args.get("grad_accum", 8))
    lr = float(train_args.get("lr", 1e-6))
    logging_steps = int(train_args.get("logging_steps", 10))
    num_gen = int(train_args.get("num_generations", 4))
    max_prompt_len = int(train_args.get("max_prompt_len", 512))
    max_completion_len = int(train_args.get("max_completion_len", 128))
    loss_type = str(train_args.get("loss_type", "dapo"))  # "dapo" recommended
    scale_rewards = train_args.get(
        "scale_rewards", "group"
    )  # "group"|"batch"|"none"/False
    beta = float(train_args.get("beta", 0.0))  # set >0.0 to log KL
    optim = str(train_args.get("optim", "adamw_torch_fused"))
    num_workers = int(train_args.get("num_workers", 4))
    group_by_length = bool(train_args.get("group_by_length", True))
    dataloader_pin_memory = bool(train_args.get("dataloader_pin_memory", True))
    warmup_ratio = float(train_args.get("warmup_ratio", 0.04))
    lr_scheduler_type = str(train_args.get("lr_scheduler_type", "cosine"))
    max_steps = int(train_args.get("max_steps", -1))

    # Accelerator hints (DDP etc.). For true multi-GPU DDP, launch with accelerate/torchrun.
    accelerator_config = {
        # mixed_precision is handled by bf16/fp16 parameters directly on TrainingArguments
    }

    trainer = None
    cfg = GRPOConfig(
        output_dir=out_dir,
        logging_dir=tb_dir,
        run_name=run_name,
        report_to=["wandb"],
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=logging_steps,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        max_prompt_length=max_prompt_len,
        max_completion_length=max_completion_len,
        num_generations=num_gen,
        loss_type=loss_type,  # "dapo" or "dr_grpo" etc.
        scale_rewards=scale_rewards,  # "group"|"batch"|"none"/False
        beta=beta,  # nonzero -> logs KL
        mask_truncated_completions=True,  # generally stabilizes training
        accelerator_config=accelerator_config,
        optim=optim,
        dataloader_num_workers=num_workers,
        group_by_length=group_by_length,
        dataloader_pin_memory=dataloader_pin_memory,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        max_steps=max_steps if max_steps and max_steps > 0 else -1,
        gradient_checkpointing=True,
    )
    trainer = GRPOTrainer(
        model=base,
        processing_class=tok,  # tokenizer; left padding required
        args=cfg,
        train_dataset=ds,
        reward_funcs=[reward_correctness, reward_brevity],
        peft_config=lora_cfg,
    )

    # Train!
    rprint("[bold green]Training...[/bold green]")
    trainer.train()

    # save adapter + tokenizer
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # tarball artifact
    tar_path = os.path.join(OUT_ROOT, f"{job_id}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(out_dir, arcname=os.path.basename(out_dir))

    wandb.finish()

    # record manifest
    manifest = {
        "job_id": job_id,
        "model": model_name,
        "algo": algo,
        "tb_dir": tb_dir,
        "artifact_dir": out_dir,
        "artifact_tar": tar_path,
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
    import os
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
