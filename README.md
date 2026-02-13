# planit

Declaratively build workflow DAGs for SLURM clusters. Define your pipeline as a tree of `Step`, `Chain`, and `Parallel` nodes, and planit handles dependency chains (`afterok`) and submission via [submitit](https://github.com/facebookincubator/submitit).

## Installation

```bash
uv add git+https://github.com/WPoelman/planit.git
# or
pip install git+https://github.com/WPoelman/planit.git
```

## Quick start

```python
import logging

import submitit

from planit import Parallel, Plan, Chain, SlurmArgs, Step

logging.basicConfig(level=logging.INFO, format="%(message)s")

GPU = SlurmArgs(time="02:00:00", partition="gpu_a100", gpus_per_node=1, cpus_per_gpu=18)
CPU = SlurmArgs(time="01:00:00", partition="batch", cpus_per_task=8)

plan = Plan(
    "experiment",
    Chain(
        Step("download", download_data, CPU),
        Step("preprocess", preprocess, CPU),
        Parallel(
            Step("train_model_a", train, GPU, "model_a"),
            Step("train_model_b", train, GPU, "model_b"),
        ),
        Step("evaluate", evaluate, CPU),
    ),
)

plan.describe()

executor = submitit.AutoExecutor(folder="slurm_logs")
plan.submit(executor)
```

`describe()` prints the DAG and a best-case time estimate:

```
Plan: experiment
└── ▼ Chain [5:00:00]
    ├── ● download [01:00:00]
    ├── ● preprocess [01:00:00]
    ├── ⇉ Parallel [2:00:00]
    │   ├── ● train_model_a [02:00:00]
    │   └── ● train_model_b [02:00:00]
    └── ● evaluate [01:00:00]
Time Estimate (not taking queuing into account): 5:00:00
```

## Design

### Nodes

A plan is a tree built from three **`Node`** types:

- **`Step(name, func, slurm_args, *args, **kwargs)`**: a single SLURM job. `func` is called with `*args` and `**kwargs` when the job runs.
- **`Chain(*nodes)`**: runs children one after another. Each child waits for the previous one to finish (`afterok`).
- **`Parallel(*nodes)`**: runs children concurrently. All children start after the same parent finishes.

With nesting, these can generate different kinds of workflow DAGs:

```python
Chain(
    Step("setup", setup_fn, CPU), # no arguments for 'setup_fn'
    Parallel(
        Step("branch_a", work_a, GPU, "config_a"), # 'work_a' will be called as work_a("config_a")
        Chain(
            Step("branch_b_prep", prep_b, CPU, option=1), # prep_b(option=1)
            Step("branch_b_run", work_b, GPU),
        ),
    ),
    Step("aggregate", combine_results, CPU),
)
```

You can do stuff like this (not saying you *should*):

```python
Plan("thesis", Chain(
    Step("download_data", download, CPU),
    Parallel(
        Chain(
            Step("preprocess_en", preprocess, CPU, "en"),
            Parallel(
                Step("train_bert", train, GPU, "bert", epochs=10, lr=3e-5),
                Step("train_roberta", train, GPU, "roberta", epochs=5),
                Chain(
                    Step("hyperparam_search", search, GPU, "xlm-r"),
                    Step("train_xlm-r", train, GPU, "xlm-r"),
                ),
            ),
        ),
        Chain(
            Step("preprocess_nl", preprocess, CPU, "nl"),
            Step("train_nl", train, GPU, "nl"),
        ),
    ),
    Step("evaluate_all", evaluate, CPU),
    Step("generate_plots", plot, CPU),
))
```

If we assume `CPU` takes 1 hour and `GPU` 2, this generates the following DAG with `plan.describe()`:

```
Plan: thesis
└── ▼ Chain [8:00:00]
    ├── ● download_data [01:00:00]
    ├── ⇉ Parallel [5:00:00]
    │   ├── ▼ Chain [5:00:00]
    │   │   ├── ● preprocess_en [01:00:00]
    │   │   └── ⇉ Parallel [4:00:00]
    │   │       ├── ● train_bert [02:00:00]
    │   │       ├── ● train_roberta [02:00:00]
    │   │       └── ▼ Chain [4:00:00]
    │   │           ├── ● hyperparam_search [02:00:00]
    │   │           └── ● train_xlm-r [02:00:00]
    │   └── ▼ Chain [3:00:00]
    │       ├── ● preprocess_nl [01:00:00]
    │       └── ● train_nl [02:00:00]
    ├── ● evaluate_all [01:00:00]
    └── ● generate_plots [01:00:00]
Time Estimate (not taking queuing into account): 8:00:00
```

The 8 hours is the critical path:
* `download_data` (1h) +
* slowest parallel branch (`preprocess_en` 1h + `hyperparam_search` 2h + `train_xlm-r` 2h = 5h) +
* `evaluate_all` (1h) +
* `generate_plots` (1h)
* = 8h

This is of course without any potential queue time or jobs finishing early.
It's only an estimate of the requested time, taking into account parallel jobs.

### SLURM config

#### `SlurmArgs`

This is a dataclass for defining the SLURM parameters for your job:

```python
from planit import MailType, SlurmArgs

args = SlurmArgs(
    time="03:00:00",
    partition="gpu_a100",
    gpus_per_node=1,
    cpus_per_gpu=18,
    cluster="wice",
    account="my-account",
    mail_type=[MailType.BEGIN, MailType.END, MailType.FAIL],
    mail_user="me@university.edu",
)
```

This is tailored to my own work on the [VSC](https://www.vscentrum.be/) and may not cover all the configurations you might need on your cluster (see the [next section](#raw-dict) for an alternative).
CPU and GPU configurations I regularly use on the VSC are included in [example_vsc_args.py](example_vsc_args.py).

Available fields:

| Field               | Type             | Default    | Description                                         |
| ------------------- | ---------------- | ---------- | --------------------------------------------------- |
| `time`              | `str`            | *required* | Wall time (`HH:MM:SS`, `MM:SS`, or `days-HH:MM:SS`) |
| `partition`         | `str`            | *required* | SLURM partition                                     |
| `gpus_per_node`     | `int`            | `0`        | GPUs per node                                       |
| `nodes`             | `int`            | `1`        | Number of nodes                                     |
| `cpus_per_task`     | `int \| None`    | `None`     | CPUs per task                                       |
| `cpus_per_gpu`      | `int \| None`    | `None`     | CPUs per GPU                                        |
| `mem_gb`            | `int \| None`    | `None`     | Memory in GB                                        |
| `account`           | `str \| None`    | `None`     | Account name                                        |
| `cluster`           | `str \| None`    | `None`     | Cluster name                                        |
| `mail_type`         | `list[MailType]` | `[]`       | Mail notification types                             |
| `mail_user`         | `str \| None`    | `None`     | Mail recipient                                      |
| `additional_params` | `dict`           | `{}`       | Extra `slurm_additional_parameters`                 |

#### Raw dict

You can pass a raw dict directly if `SlurmArgs` doesn't fit for your cluster.
This dict should be compatible with [submitit](https://github.com/facebookincubator/submitit).
It must (at least) include `"slurm_time"` so planit can estimate durations:

```python
args = {
    "slurm_time": "02:00:00",
    "slurm_partition": "gpu_v100",
    "gpus_per_node": 1,
    "slurm_additional_parameters": {
        "clusters": "my-cluster",
        "account": "my-account",
        "cpus_per_gpu": 4,
    },
}
Step("train", train_fn, args)
```

## Errors and communication

planit only uses `afterok` dependencies: a job only starts if **all** its parents succeeded. If a job fails, SLURM automatically cancels all downstream dependents.

Communication between jobs is expected to happen through the **filesystem**.
For example, one step writes a checkpoint file and the next step reads it.
planit does not pass return values between steps; it only manages the dependency graph, slurm args, and submission.

## Debugging locally

You can use submitit's "debug" executor to run your plan locally without a cluster (and keeping everything in one process):

```python
executor = submitit.AutoExecutor(folder="slurm_logs", cluster="debug")
plan.submit(executor)
plan.wait()  # blocks until done, runs parallel branches concurrently
```

`wait()` walks the DAG structure using threads so parallel branches execute concurrently, mirroring real cluster behavior. This is intended for debugging and short jobs.
On a real cluster you should probably not use `wait()`, unless you know queue times and job durations will be short.

---
```
             _____
          .-'.  ':'-.
        .''::: .:    '.
       /   :::::'      \
      ;.    ':' `       ;
      |       '..       |
      ; '      ::::.    ;
       \       '::::   /
        '.      :::  .'
jgs        '-.___'_.-'
```
Credit: https://www.asciiart.eu/art/a5e06526e7b3ae4b
