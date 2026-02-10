"""
Example planit configurations for VSC clusters (Wice / Genius).

The cpu counts for GPU partitions are *not* arbitrary, these are required per node.
The counts for the batch jobs are the max.

See: https://docs.vscentrum.be/leuven/tier2_hardware.html
"""

from planit import MailType, SlurmArgs

MAIL = [MailType.BEGIN, MailType.END, MailType.FAIL]


WICE_BATCH = SlurmArgs(
    time="01:00:00",
    partition="batch",
    cpus_per_task=72,
    cluster="wice",
    mail_type=MAIL,
)

WICE_A100 = SlurmArgs(
    time="01:00:00",
    partition="gpu_a100",
    gpus_per_node=1,
    cpus_per_gpu=18,
    cluster="wice",
    mail_type=MAIL,
)

WICE_H100 = SlurmArgs(
    time="01:00:00",
    partition="gpu_h100",
    gpus_per_node=1,
    cpus_per_gpu=16,
    cluster="wice",
    mail_type=MAIL,
)

WICE_A100_DEBUG = SlurmArgs(
    time="01:00:00",
    partition="gpu_a100_debug",
    gpus_per_node=1,
    cpus_per_gpu=64,
    cluster="wice",
    mail_type=MAIL,
)

GENIUS_BATCH = SlurmArgs(
    time="01:00:00",
    partition="batch",
    cpus_per_task=36,
    cluster="genius",
    mail_type=MAIL,
)

GENIUS_V100 = SlurmArgs(
    time="01:00:00",
    partition="gpu_v100",
    gpus_per_node=1,
    cpus_per_gpu=4,
    cluster="genius",
    mail_type=MAIL,
)
