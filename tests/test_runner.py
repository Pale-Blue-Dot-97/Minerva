import os

import pytest
import torch
from internet_sabotage import no_connection

from minerva.utils import CONFIG, runner


def test_wandb_connection_manager() -> None:

    with runner.WandbConnectionManager():
        assert os.environ["WANDB_MODE"] == "online"

    with no_connection():
        with runner.WandbConnectionManager():
            assert os.environ["WANDB_MODE"] == "offline"


def test_config_env_vars() -> None:
    args, _ = runner.GENERIC_PARSER.parse_known_args()

    args.ngpus_per_node = 1

    with pytest.raises(AttributeError):
        args.rank
        args.dist_url
        args.world_size

    new_args = runner.config_env_vars(args)

    if "SLURM_JOB_ID" in os.environ:
        # TODO: Simulate SLURM environment.
        pass
    else:
        assert new_args.rank == 0
        assert new_args.world_size == 1
        assert new_args.dist_url == "tcp://localhost:58472"


def test_config_args() -> None:
    args, _ = runner.GENERIC_PARSER.parse_known_args()

    args_dict = vars(args)

    # Find which CLI arguments are not in the config.
    new_args = {key: args_dict[key] for key in args_dict if key not in CONFIG}

    returned_args = runner.config_args(args)

    assert returned_args.ngpus_per_node == torch.cuda.device_count()
    assert CONFIG["seed"] is not None

    for key in new_args.keys():
        assert CONFIG[key] == new_args[key]


def test_distributed_run() -> None:
    def run(*args):
        pass

    args, _ = runner.GENERIC_PARSER.parse_known_args()

    args = runner.config_args(args)

    runner.distributed_run(run, args)

    # TODO: Simulate multiprocessing runs.
