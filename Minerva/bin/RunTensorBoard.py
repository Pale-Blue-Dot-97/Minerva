"""Script to run the TensorBoard logs from experiments.

    Copyright (C) 2021 Harry James Baker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program in LICENSE.txt. If not,
    see <https://www.gnu.org/licenses/>.

Author: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Union, Optional
from Minerva.utils import utils
import argparse


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main(path: Optional[Union[str, list]] = None, env_name: str = 'env2', exp_name: Optional[str] = None,
         host_num: int = 6006) -> None:
    if isinstance(path, list):
        if len(path) == 1:
            path = path[0]

    utils.run_tensorboard(path=path, env_name=env_name, exp_name=exp_name, host_num=host_num)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--path",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=None,  # default if nothing is provided
    )
    CLI.add_argument(
        "--env_name",
        nargs="1",
        type=str,  # any type/callable can be used here
        default=None,
    )
    CLI.add_argument(
        "--exp_name",
        nargs="1",
        type=str,  # any type/callable can be used here
        default=None,
    )
    CLI.add_argument(
        "--host_num",
        nargs="1",
        type=int,  # any type/callable can be used here
        default=None,
    )

    args = CLI.parse_args()
    main(path=args.path, env_name=args.env_name, exp_name=args.exp_name, host_num=args.host_num)
