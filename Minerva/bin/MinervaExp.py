"""Script to execute the creation, fitting and testing of a computer vision neural network model to classify land cover.

    Copyright (C) 2022 Harry James Baker

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

Attributes:
    config_path (str): Path to master config YAML file.
    config (dict): Master config defining how the experiment should be conducted.

TODO:
    * Add arg parsing from CLI
    * Add ability to conduct hyper-parameter iterative variation experimentation
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils
from Minerva.loaders import make_datasets
from Minerva.trainer import Trainer

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

config, _ = utils.load_configs(config_path)


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    datasets, n_batches, class_dist, new_config = make_datasets(**config)

    trainer = Trainer(loaders=datasets, n_batches=n_batches, class_dist=class_dist, **new_config)
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()
