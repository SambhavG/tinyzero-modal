curl -X POST "https://sambhavg-modal-workspace--tinyzero-rl-web.modal.run/v1/rl/train" \
  -H "content-type: application/json" \
  -d '{
  "parquet_glob": "/data/*.parquet",
  "algo": "grpo",
  "train_args": {
    "epochs": 1,
    "batch_size": 64,
    "lr": 1e-6,
    "max_prompt_len": 256,
    "max_completion_len": 1024,
    "grpo_mini_batch_size": 32,
    "grpo_micro_batch_size": 32,
    "log_prob_micro_batch_size": 16,
    "tensor_model_parallel_size": 1,
    "gpu_memory_utilization": 0.4,
    "max_steps": 1
  }
}'
