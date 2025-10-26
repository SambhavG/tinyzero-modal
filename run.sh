curl -X POST "https://sambhavg-modal-workspace--tinyzero-rl-web.modal.run/v1/rl/train" \
  -H "content-type: application/json" \
  -d '{
  "parquet_glob": "/data/data/*.parquet",
  "algo": "grpo",
  "train_args": {
    "batch_size": 10,
    "grad_accum": 10,
    "num_generations": 2,
    "max_prompt_len": 384,
    "max_completion_len": 512,
    "optim": "adamw_torch_fused",
    "num_workers": 8,
    "group_by_length": true,
    "dataloader_pin_memory": true,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_train_samples": 20000,
    "max_steps": 3000,
    "use_flash_attn": true,
    "gradient_checkpointing": true,
    "compile_model": false,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05
  }
}'
