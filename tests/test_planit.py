import datetime

import pytest

from planit import Chain, MailType, Parallel, SlurmArgs, Step


def noop():
    pass


def test_parse_time_hh_mm_ss():
    step = Step("s", noop, SlurmArgs(time="02:30:00", partition="batch"))
    assert step.duration == datetime.timedelta(hours=2, minutes=30)


def test_parse_time_mm_ss():
    step = Step("s", noop, SlurmArgs(time="45:00", partition="batch"))
    assert step.duration == datetime.timedelta(minutes=45)


def test_parse_time_days():
    step = Step("s", noop, SlurmArgs(time="2-12:00:00", partition="batch"))
    assert step.duration == datetime.timedelta(days=2, hours=12)


def test_parse_time_invalid():
    with pytest.raises(ValueError, match="Could not parse time"):
        Step("s", noop, SlurmArgs(time="not-a-time", partition="batch"))


def test_get_time_raw_dict():
    step = Step("s", noop, {"slurm_time": "01:00:00"})
    assert step._get_time() == "01:00:00"
    assert step.duration == datetime.timedelta(hours=1)


def test_get_time_raw_dict_missing_key():
    with pytest.raises(ValueError, match="raw dict must include 'slurm_time'"):
        Step("s", noop, {"slurm_partition": "batch"})


def test_get_time_raw_dict_non_string_invalid_format():
    with pytest.raises(ValueError, match="Could not parse time"):
        Step("s", noop, {"slurm_time": 60})


def test_to_submitit_dict_minimal():
    args = SlurmArgs(time="01:00:00", partition="batch")
    d = args.to_submitit_dict()
    assert d["slurm_time"] == "01:00:00"
    assert d["slurm_partition"] == "batch"
    assert d["gpus_per_node"] == 0
    assert "slurm_additional_parameters" not in d


def test_to_submitit_dict_additional_params():
    args = SlurmArgs(
        time="01:00:00",
        partition="batch",
        account="my-account",
        cluster="wice",
        nodes=4,
        mail_type=[MailType.BEGIN, MailType.END],
        mail_user="user@example.com",
    )
    d = args.to_submitit_dict()
    additional = d["slurm_additional_parameters"]
    assert additional["account"] == "my-account"
    assert additional["clusters"] == "wice"
    assert additional["nodes"] == 4
    assert additional["mail_type"] == "BEGIN,END"
    assert additional["mail_user"] == "user@example.com"


def test_chain_duration_is_sum():
    chain = Chain(
        Step("a", noop, SlurmArgs(time="01:00:00", partition="batch")),
        Step("b", noop, SlurmArgs(time="00:30:00", partition="batch")),
    )
    assert chain.get_duration() == datetime.timedelta(hours=1, minutes=30)


def test_parallel_duration_is_max():
    par = Parallel(
        Step("a", noop, SlurmArgs(time="01:00:00", partition="batch")),
        Step("b", noop, SlurmArgs(time="02:00:00", partition="batch")),
    )
    assert par.get_duration() == datetime.timedelta(hours=2)


def test_nested_duration():
    tree = Chain(
        Step("a", noop, SlurmArgs(time="01:00:00", partition="batch")),
        Parallel(
            Step("b", noop, SlurmArgs(time="03:00:00", partition="batch")),
            Step("c", noop, SlurmArgs(time="02:00:00", partition="batch")),
        ),
        Step("d", noop, SlurmArgs(time="00:30:00", partition="batch")),
    )
    # 1h + max(3h, 2h) + 0.5h = 4.5h
    assert tree.get_duration() == datetime.timedelta(hours=4, minutes=30)


def test_empty_duration():
    assert Parallel().get_duration() == datetime.timedelta(0)
    assert Chain().get_duration() == datetime.timedelta(0)


def test_step_stores_func_args():
    def add(a, b):
        return a + b

    step = Step("s", add, SlurmArgs(time="00:10:00", partition="batch"), 1, 2)
    assert step.args == (1, 2)


def test_step_stores_args_and_kwargs():
    def train(data, epochs, lr=0.001, optim=None):
        pass

    step = Step("s", train, SlurmArgs(time="00:10:00", partition="batch"), "data.csv", 10, lr=0.01, optim="adamw")
    assert step.args == ("data.csv", 10)
    assert step.kwargs == {"lr": 0.01, "optim": "adamw"}
