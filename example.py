import logging
import random
import time
from collections.abc import Iterable

import submitit

from planit import Chain, Parallel, Plan, SlurmArgs, Step

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


def train_model(language: str, model_type: str, epochs=10, lr=3e-5):
    logger.info(f"Training {language} {model_type} ({epochs=}, {lr=})...")
    time.sleep(random.randint(2, 5))


def param_search(model_type: str):
    logger.info(f"Searching {model_type}...")
    time.sleep(random.randint(3, 6))


def plot_results(models: Iterable[str]):
    for model in models:
        logger.info(f"Generating plots for {model}")


steps = Chain(
    Step("download", CPU_ARGS, download_data),
    Step("clean_data", CPU_ARGS, clean_data),
    Parallel(
        Step("train_a", GPU_ARGS, train_model, "en", "model_a", epochs=2),
        Step("train_b", GPU_ARGS, train_model, "fr", "model_b", epochs=4, lr=4e-5),
        Chain(
            Step("param_search", GPU_ARGS, param_search, "model_c"),
            Step("train_c", GPU_ARGS, train_model, "am", "model_c"),
        ),
    ),
    Step("plot", CPU_ARGS, plot_results, ("model_a", "model_b", "model_c")),
)

# The 'cluster' argument here is not necessary when submitting real jobs.
# This just makes sure we have debug output and the executer runs in the
# current process instead of spawning new ones.
executor = submitit.AutoExecutor(folder="slurm_logs", cluster="debug")

plan = Plan("my_experiment", steps)
plan.describe()
plan.submit(executor)
plan.wait()
