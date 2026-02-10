import logging
import random
import time
from collections.abc import Iterable

import submitit

from planit import Parallel, Plan, Chain, SlurmArgs, Step

# Submitit logs to a generic "" logger, this is not great, so it's filtered out
# like this. This doesn't matter for the functionality, but you could do
# something similar if the logs bother you.
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger().setLevel(logging.ERROR)
logger = logging.getLogger("planit")
logger.setLevel(logging.INFO)

CPU_ARGS = SlurmArgs(time="01:00:00", partition="cpu", cpus_per_task=8)
GPU_ARGS = SlurmArgs(time="01:00:00", partition="gpu", gpus_per_node=4, cpus_per_task=16)


def download_data():
    logger.info("Downloading data...")
    time.sleep(random.randint(1, 3))


def clean_data():
    logger.info("Cleaning data...")
    time.sleep(random.randint(1, 3))


def train_model(model_type: str):
    logger.info(f"Training {model_type}...")
    time.sleep(random.randint(2, 5))


def param_search(model_type: str):
    logger.info(f"Searching {model_type}...")
    time.sleep(random.randint(3, 8))


def plot_results(models: Iterable[str]):
    for model in models:
        logger.info(f"{model} is done")


steps = Chain(
    Step("download", download_data, CPU_ARGS),
    Step("clean_data", clean_data, CPU_ARGS),
    Parallel(
        Step("train_a", train_model, GPU_ARGS, "model_a"),
        Step("train_b", train_model, GPU_ARGS, "model_b"),
        Chain(
            Step("param_search", param_search, GPU_ARGS, "model_c"),
            Step("train_c", train_model, GPU_ARGS, "model_c"),
        ),
    ),
    Step("plot", plot_results, CPU_ARGS, ("model_a", "model_b", "model_c")),
)

# The 'cluster' argument here is not necessary when submitting real jobs.
# This just makes sure we have debug output and the executer runs in the
# current process instead of spawning new ones.
executor = submitit.AutoExecutor(folder="slurm_logs", cluster="debug")

plan = Plan("my_experiment", steps)
plan.describe()
plan.submit(executor)
plan.wait()
