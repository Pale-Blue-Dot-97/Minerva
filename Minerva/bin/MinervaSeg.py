"""Script to execute the creation, fitting and testing of an image segmentation model to classify land cover.

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

Attributes:
    config_path (str): Path to master config YAML file.
    config (dict): Master config defining how the experiment should be conducted.
    aux_configs (dict): Dict containing the auxiliary config dicts loaded from YAML.
    dataset_config (dict): Config defining the properties of the data used in the experiment.
    params (dict): Sub-dict of the master config for the model hyper-parameters.

TODO:
    * Add arg parsing from CLI
    * Add ability to conduct hyper-parameter iterative variation experimentation
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils
import Minerva.loaders as loaders
from Minerva.trainer import Trainer

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

config, aux_configs = utils.load_configs(config_path)
dataset_config = aux_configs['data_config']

# Parameters
params = config['hyperparams']['params']


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    datasets, n_batches, class_dist, ids, new_classes, new_colours = loaders.make_datasets(frac=0.1, **config)

    config['hyperparams']['model_params']['n_classes'] = len(new_classes)
    config['classes'] = new_classes
    config['colours'] = new_colours

    trainer = Trainer(loaders=datasets, n_batches=n_batches, class_dist=class_dist, **config)

    trainer.fit()

    trainer.test({'History': True, 'Pred': True, 'CM': True, 'Mask': True}, save=True)

    trainer.close()

    #trainer.run_tensorboard()


if __name__ == '__main__':
    main()
