import os
import pytest
from minerva.utils import runner


def test_config_env_vars():
    args, _ = runner.generic_parser.parse_known_args()

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
