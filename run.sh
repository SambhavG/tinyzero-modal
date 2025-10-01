curl -X POST "https://sambhavg-modal-workspace--tinyzero-rl-web.modal.run/v1/rl/train" \
  -H "content-type: application/json" \
  -d '{
    "parquet_glob": "/data/*.parquet",
    "algo": "grpo",
    "train_args": {
      "epochs": 1,
      "batch_size": 2,
      "grad_accum": 8,
      "lr": 1e-6,
      "logging_steps": 10,
      "num_generations": 8,
      "max_prompt_len": 512,
      "max_completion_len": 256,
      "loss_type": "dapo",
      "scale_rewards": "group",
      "beta": 0.0,
      "wandb_entity": "sambhavg"
    }
  }'
