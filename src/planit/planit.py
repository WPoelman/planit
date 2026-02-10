import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

import submitit

logger = logging.getLogger(__name__)


class MailType(StrEnum):
    NONE = "NONE"
    BEGIN = "BEGIN"
    END = "END"
    FAIL = "FAIL"
    REQUEUE = "REQUEUE"
    ALL = "ALL"


@dataclass
class SlurmArgs:
    """
    Convenience wrapper for common VSC SLURM parameters.

    These args are cluster-specific and may not cover all configurations. If your
    cluster requires different parameters, you can pass a raw dict directly
    to Step instead. The dict should contain submitit-compatible keys
    (e.g. "slurm_time", "slurm_partition", "slurm_additional_parameters")
    and must include a "slurm_time" entry for time estimation.

    Example using SlurmArgs::

        Step("train", train_fn, SlurmArgs(time="02:00:00", partition="gpu"))

    Example using a raw dict::

        Step("train", train_fn, {
            "slurm_time": "02:00:00",
            "slurm_partition": "gpu_v100",
            "slurm_additional_parameters": {"account": "my-account", ...},
        })
    """

    time: str  # 'HH:MM:SS', 'MM:SS' or 'days-HH:MM:SS'
    partition: str
    gpus_per_node: int = 0
    nodes: int = 1
    cpus_per_task: int | None = None
    cpus_per_gpu: int | None = None
    mem_gb: int | None = None
    account: str | None = None
    cluster: str | None = None
    mail_type: list[MailType] = field(default_factory=list)
    mail_user: str | None = None
    additional_params: dict[str, Any] = field(default_factory=dict)

    def to_submitit_dict(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "slurm_time": self.time,
            "slurm_partition": self.partition,
            "gpus_per_node": self.gpus_per_node,
        }
        if self.cpus_per_task is not None:
            args["cpus_per_task"] = self.cpus_per_task
        if self.mem_gb is not None:
            args["mem_gb"] = self.mem_gb

        additional: dict[str, Any] = {}
        if self.nodes != 1:
            additional["nodes"] = self.nodes
        if self.cpus_per_gpu is not None:
            additional["cpus_per_gpu"] = self.cpus_per_gpu
        if self.account is not None:
            additional["account"] = self.account
        if self.cluster is not None:
            # NOTE: this is indeed plural on VSC!
            additional["clusters"] = self.cluster
        if self.mail_type:
            additional["mail_type"] = ",".join(self.mail_type)
        if self.mail_user is not None:
            additional["mail_user"] = self.mail_user
        additional.update(self.additional_params)

        if additional:
            args["slurm_additional_parameters"] = additional
        return args


class Node(ABC):
    @abstractmethod
    def get_duration(self) -> datetime.timedelta: ...


class Parallel(Node):
    def __init__(self, *nodes: Node) -> None:
        self.nodes: list[Node] = list(nodes)

    def get_duration(self) -> datetime.timedelta:
        return max((n.get_duration() for n in self.nodes), default=datetime.timedelta(0))


class Chain(Node):
    def __init__(self, *nodes: Node) -> None:
        self.nodes: list[Node] = list(nodes)

    def get_duration(self) -> datetime.timedelta:
        return sum((n.get_duration() for n in self.nodes), datetime.timedelta(0))


class Step(Node):
    def __init__(
        self,
        name: str,
        func: Callable[..., object],
        slurm_args: SlurmArgs | dict[str, Any],
        /,
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        A Step corresponds to a single function executed as a job by SLURM.

        name:
            Name of this step.
        func:
            The function that will be executed as a SLURM job.
        slurm_args:
            SLURM parameters for the job. Either a SlurmArgs instance or a raw
            dict of submitit-compatible parameters. A raw dict must include
            "slurm_time" for time estimation.
        *args:
            Positional arguments passed to `func` when the job executes.
        **kwargs:
            Keyword arguments passed to `func` when the job executes.
        """
        self.name: str = name
        self.func: Callable[..., object] = func
        self.slurm_args: SlurmArgs | dict[str, Any] = slurm_args
        self.args: tuple[object, ...] = args
        self.kwargs: dict[str, object] = kwargs

        self.job: submitit.Job[object] | None = None
        self.duration: datetime.timedelta = _parse_slurm_time(self._get_time())

    def _get_time(self) -> str:
        if isinstance(self.slurm_args, SlurmArgs):
            return self.slurm_args.time
        time_val: Any = self.slurm_args.get("slurm_time")
        if time_val is None:
            raise ValueError(f"Step '{self.name}': raw dict must include 'slurm_time'")
        return str(time_val)

    def _to_submitit_dict(self) -> dict[str, Any]:
        if isinstance(self.slurm_args, SlurmArgs):
            return self.slurm_args.to_submitit_dict()
        return dict(self.slurm_args)

    def get_duration(self) -> datetime.timedelta:
        return self.duration


class Plan:
    def __init__(self, name: str, root: Node) -> None:
        self.name: str = name
        self.root: Node = root

    def describe(self) -> None:
        """Visualizes the DAG and calculates the best-case runtime."""
        logger.info(f"Plan: {self.name}")
        self._print_recursive(self.root, prefix="", is_last=True)
        logger.info(f"Time Estimate (not taking queuing into account): {self.root.get_duration()}")

    def _print_recursive(self, node: Node, prefix: str, is_last: bool) -> None:
        marker: str = "└── " if is_last else "├── "
        child_prefix: str = prefix + ("    " if is_last else "│   ")

        if isinstance(node, Step):
            logger.info(f"{prefix}{marker}● {node.name} [{node._get_time()}]")
        elif isinstance(node, Parallel):
            logger.info(f"{prefix}{marker}⇉ Parallel [{node.get_duration()}]")
            for i, n in enumerate(node.nodes):
                self._print_recursive(n, child_prefix, i == len(node.nodes) - 1)
        elif isinstance(node, Chain):
            logger.info(f"{prefix}{marker}▼ Chain [{node.get_duration()}]")
            for i, n in enumerate(node.nodes):
                self._print_recursive(n, child_prefix, i == len(node.nodes) - 1)

    def submit(self, executor: submitit.Executor) -> list[submitit.Job[object]]:
        """Recursively submits the plan to the cluster and returns all jobs."""
        logger.info(f"Submitting jobs for '{self.name}'...")
        all_jobs: list[submitit.Job[object]] = []

        def _walk(node: Node, parent_jobs: list[submitit.Job[object]]) -> list[submitit.Job[object]]:
            parent_ids: list[str] = [str(j.job_id) for j in parent_jobs]

            if isinstance(node, Step):
                params: dict[str, Any] = node._to_submitit_dict()
                if "slurm_job_name" not in params:
                    params["slurm_job_name"] = self.name
                if parent_ids:
                    dep_str: str = f"afterok:{':'.join(parent_ids)}"
                    params.setdefault("slurm_additional_parameters", {})["dependency"] = dep_str

                executor.update_parameters(**params)
                job: submitit.Job[object] = executor.submit(node.func, *node.args, **node.kwargs)
                node.job = job
                all_jobs.append(job)

                logger.info(f"  [Queued] {node.name} (ID: {job.job_id})")
                return [job]

            elif isinstance(node, Parallel):
                # all branches start after the same parent_jobs
                results: list[submitit.Job[object]] = []
                for n in node.nodes:
                    results.extend(_walk(n, parent_jobs))
                return results

            elif isinstance(node, Chain):
                # each node waits for the job(s) returned by the previous one
                current_deps: list[submitit.Job[object]] = parent_jobs
                for n in node.nodes:
                    current_deps = _walk(n, current_deps)
                return current_deps

            raise TypeError(f"Unknown node type: {type(node)}")

        _walk(self.root, [])
        logger.info(f"\nAll jobs for '{self.name}' have been queued.")
        return all_jobs

    def wait(self) -> None:
        """
        Wait for all jobs, running parallel branches concurrently.
        This is not recommended for actual use (unless you have short jobs).
        It's more for debugging and demo purposes.

        This (mis)uses the blocking feature of a ThreadPoolExecutor to show
        that parallel jobs are indeed running in parallel.
        """
        logger.info(f"\nWaiting for '{self.name}' to complete...")
        self._wait_node(self.root)
        logger.info(f"\nAll jobs for '{self.name}' have completed.")

    def _wait_node(self, node: Node) -> None:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import wait as wait_futures

        if isinstance(node, Step):
            if node.job is None:
                raise RuntimeError(f"Step '{node.name}' has not been submitted yet")
            node.job.result()
            logger.info(f"  [Done] {node.name}")

        elif isinstance(node, Parallel):
            with ThreadPoolExecutor(max_workers=len(node.nodes)) as pool:
                futures = [pool.submit(self._wait_node, n) for n in node.nodes]
                wait_futures(futures)
                for f in futures:
                    f.result()

        elif isinstance(node, Chain):
            for n in node.nodes:
                self._wait_node(n)


def _parse_slurm_time(time_str: str) -> datetime.timedelta:
    """Parses 'HH:MM:SS', 'MM:SS' or 'days-HH:MM:SS'."""
    try:
        if "-" in time_str:
            days, rest = time_str.split("-")
            t = datetime.datetime.strptime(rest, "%H:%M:%S")
            return datetime.timedelta(days=int(days), hours=t.hour, minutes=t.minute, seconds=t.second)
        elif time_str.count(":") == 2:
            t = datetime.datetime.strptime(time_str, "%H:%M:%S")
            return datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        elif time_str.count(":") == 1:
            t = datetime.datetime.strptime(time_str, "%M:%S")
            return datetime.timedelta(minutes=t.minute, seconds=t.second)
        else:
            raise ValueError("Unknown time format.")
    except ValueError as e:
        raise ValueError(f"Could not parse time '{time_str}'.\n{e}")
