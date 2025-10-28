# ---
# cmd: ["modal", "run", "06_gpu_and_ml/reinforcement-learning/grpo_verl.py::train"]
# ---

# # Train a model to solve math problems using GRPO and verl

# This example demonstrates how to train with [GRPO](https://arxiv.org/pdf/2402.03300) on Modal using the [verl](https://github.com/volcengine/verl) framework.
# GRPO is a reinforcement learning algorithm introduced by DeepSeek, and was used to train DeepSeek R1.
# verl is a reinforcement learning training library that is an implementation of [HybridFlow](https://arxiv.org/abs/2409.19256v2), an RLHF framework.

# The training process works as follows:
# - Each example in the dataset corresponds to a math problem.
# - In each training step, the model attempts to solve the math problems showing its steps.
# - We then compute a reward for the model's solution using the reward function defined below.
# - That reward value is then used to update the model's parameters according to the GRPO training algorithm.

# ## Setup

# Import the necessary modules for Modal deployment.
import re
import os
import time
import subprocess
from pathlib import Path
import modal
import wandb
import json
from threading import Lock

app = modal.App("example-grpo-verl-tinyzero")

VERL_REPO_PATH: Path = Path("/root/verl")
image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .pip_install("verl[vllm]==0.4.1")
    .uv_pip_install("datasets>=4.0.0")
)

DATA_PATH: Path = Path("/data")
data_volume: modal.Volume = modal.Volume.from_name(
    "countdown-data", create_if_missing=True
)


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if len(matches) == 1:
        return matches[-1].group(1).strip()
    return None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        return numbers_in_eq == available_numbers
    except Exception as _:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as _:
        return None


def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    return_score = 0.0
    equation = extract_solution(solution_str=solution_str)

    if equation is None:
        return return_score, None

    return_score += format_score

    if not validate_equation(equation, numbers):
        return return_score, None

    try:
        result = evaluate_equation(equation)
        if result is None:
            return return_score, None

        if abs(result - target) < 1e-5:
            return score, result
        else:
            return return_score, result
    except Exception as _:
        return return_score, None


_wandb_traj_table_lock = Lock()
_wandb_traj_table_value = []
_trajectories = 0


def compute_reward(
    data_source: str, solution_str: str, ground_truth: dict, extra_info: dict
) -> float:
    reward, result = compute_score(solution_str, ground_truth)
    try:
        global _wandb_traj_table_lock, _wandb_traj_table_value, _trajectories
        with _wandb_traj_table_lock:
            _wandb_traj_table_value.append(
                {
                    "time": time.time(),
                    "data_source": data_source,
                    "target": (ground_truth or {}).get("target"),
                    "result": str(result) if result is not None else "",
                    "numbers": (ground_truth or {}).get("numbers"),
                    "solution": solution_str,
                    "reward": float(reward),
                    "index": extra_info.get("index"),
                    "split": extra_info.get("split"),
                }
            )
            _trajectories += 1

            if len(_wandb_traj_table_value) >= 1000:
                # Create a new table and append the trajectories, then clear the table
                _wandb_traj_table = wandb.Table(
                    columns=[
                        "time",
                        "data_source",
                        "target",
                        "result",
                        "numbers",
                        "solution",
                        "reward",
                        "index",
                        "split",
                    ],
                    log_mode="MUTABLE",
                )
                for row in _wandb_traj_table_value:
                    _wandb_traj_table.add_data(*list(row.values()))
                _wandb_traj_table_value = []

                wandb.log(
                    {f"trajectories_{_trajectories}": _wandb_traj_table},
                    commit=False,
                )
    except Exception as e:
        print(f"Error logging trajectory: {e}")
        pass
    return float(reward)


PATH_TO_REWARD_FUNCTION: Path = Path("/root/grpo_verl_tinyzero.py")
REWARD_FUNCTION_NAME: str = "compute_reward"
MODELS_PATH: Path = Path("/models")


checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "tinyzero-checkpoints", create_if_missing=True
)


@app.function(
    image=image,
    gpu="H200:1",
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH: data_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * 60,
)
def train(*arglist) -> None:
    data_volume.reload()

    os.environ.setdefault("WANDB_ENTITY", "sambhavg")
    os.environ.setdefault("WANDB_PROJECT", "verl_countdown")
    os.environ.setdefault("WANDB_NAME", "verl_countdown_run")

    cmd: list[str] = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'test.parquet'}",
        "data.train_batch_size=32",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.n=4",
        "data.max_prompt_length=512",
        "data.max_response_length=1024",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.actor.checkpoint.save_contents='model,optimizer,extra,hf_model'",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console', 'wandb']",
        "trainer.project_name=verl_countdown",
        "trainer.experiment_name=verl_countdown_run",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.test_freq=20",
        f"trainer.default_local_dir={MODELS_PATH}",
        "trainer.resume_mode=auto",
        # Parameters chosen to ensure easy automated testing. Remove if needed.
        "trainer.save_freq=100",
        "trainer.total_training_steps=100",
        "trainer.total_epochs=1",
        # For the custom reward function.
        f"custom_reward_function.path={str(PATH_TO_REWARD_FUNCTION)}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]
    if arglist:
        cmd.extend(arglist)

    subprocess.run(cmd, check=True)


# You can now run the training using `modal run --detach grpo_verl.py::train`, or pass in any [additional args from the CLI](https://modal.com/docs/guide/apps#argument-parsing) like this `modal run --detach grpo.py::train -- trainer.total_epochs=20 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16`.

# ## Performing inference on the trained model

# We use vLLM to perform inference on the trained model.

VLLM_PORT: int = 8000


# Once you have the model checkpoints in your Modal Volume, you can load the weights and perform inference using vLLM. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).
# The weights path is as follows: `global_step_n/actor/huggingface` where n is the checkpoint you want (e.g. `global_step_5/actor/huggingface`).
# The `latest_checkpointed_iteration.txt` file stores the most recent checkpoint index.
def get_latest_checkpoint_file_path():
    with open(MODELS_PATH / "latest_checkpointed_iteration.txt") as f:
        latest_checkpoint_index = int(f.read())
    return str(
        MODELS_PATH / f"global_step_{latest_checkpoint_index}" / "actor" / "huggingface"
    )


# We provide the code for setting up an OpenAI compatible inference endpoint here. For more details re. serving models on vLLM, check out [this example.](https://modal.com/docs/examples/vllm_inference#deploy-the-server)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"VLLM_USE_V1": "1"})
)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H100:1",
    scaledown_window=2,  # How long should we stay up with no requests?
    timeout=10 * 60,  # How long should we wait for container start?
    volumes={"/root/.cache/vllm": vllm_cache_vol, MODELS_PATH: checkpoints_volume},
)
@modal.concurrent(
    max_inputs=32
)  # How many requests can one replica handle? Tune carefully!
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess

    latest_checkpoint_file_path = get_latest_checkpoint_file_path()

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        latest_checkpoint_file_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        "2",
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


# You can then deploy the server using `modal deploy grpo_verl.py`, which gives you a custom URL. You can then query it using the following curl command:

# ```bash
# curl -X POST <url>/v1/chat/completions \
#   -H 'Content-Type: application/json' \
#   -d '{
#     "messages": [
#       {"role": "system", "content": "You are a helpful assistant for solving math problems."},
#       {"role": "user", "content": "James had 4 apples. Mary gave him 2 and he ate 1. How many does he have left?"}
#     ],
#     "temperature": 0.7
#   }'
# ```

# or in the [following ways](https://modal.com/docs/examples/vllm_inference#interact-with-the-server).
