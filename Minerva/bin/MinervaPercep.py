"""Script to create a simple MLP to classify land cover of Sentinel-2 images.

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
    imagery_config (dict): Config defining the properties of the imagery used in the experiment.
    image_size (tuple): Defines the shape of the images.
    n_pixels (int): Total number of pixels in each sample (per band).
    params (dict): Sub-dict of the master config for the model hyper-parameters.
    wheel_size: Length of each `wheel' to used in class balancing sampling. Set to n_pixels.
        This is essentially the number of pixel stacks per class to have queued at any one time.

TODO:
    * Add arg parsing from CLI
    * Add ability to conduct hyper-parameter iterative variation experimentation
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils, visutils
import Minerva.loaders as loaders
from Minerva.trainer import Trainer
from matplotlib.colors import ListedColormap
import numpy as np
import osr

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

config, aux_configs = utils.load_configs(config_path)
dataset_config = aux_configs['data_config']
imagery_config = aux_configs['imagery_config']

# Defines size of the images to determine the number of batches
image_size = imagery_config['data_specs']['image_size']

n_pixels = image_size[0] * image_size[1]

# Parameters
params = config['hyperparams']['params']

wheel_size = n_pixels


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def mlp_prediction_plot(z, y, test_ids):
    """Custom function for pre-processing the outputs from MLP testing for data visualisation.

    Args:
        z (list[float]): Predicted labels by the MLP.
        y (list[float]): Corresponding ground truth labels.
        test_ids (list[str]): Corresponding patch IDs for the test data supplied to the MLP.
    """
    # `De-interlaces' the outputs to account for the effects of multi-threaded workloads.
    z = visutils.deinterlace(z, params['num_workers'])
    y = visutils.deinterlace(y, params['num_workers'])
    test_ids = visutils.deinterlace(test_ids, params['num_workers'])

    # Extracts just a patch ID for each test patch supplied.
    test_ids = [test_ids[i] for i in np.arange(start=0, stop=len(test_ids), step=n_pixels)]

    # Create a new projection system in lat-lon
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(dataset_config['co_sys']['id'])

    # Plots the predicted versus ground truth labels for all test patches supplied.
    visutils.plot_all_pvl(predictions=z, labels=y, patch_ids=test_ids, exp_id=config['model_name'], new_cs=new_cs,
                          classes=dataset_config['classes'],
                          cmap=ListedColormap(dataset_config['colours'].values(), N=len(dataset_config['classes'])))


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    datasets, n_batches, class_dist, ids, new_classes, new_colours = loaders.make_datasets(wheel_size=wheel_size,
                                                                                           image_len=n_pixels,
                                                                                           **config)
    config['hyperparams']['model_params']['n_classes'] = len(new_classes)
    config['classes'] = new_classes
    config['colours'] = new_colours

    trainer = Trainer(loaders=datasets, n_batches=n_batches, class_dist=class_dist, **config)

    trainer.fit()

    z, y, test_ids = trainer.test({'History': True, 'Pred': True, 'CM': True}, save=True)

    mlp_prediction_plot(z, y, test_ids)

    trainer.close()

    trainer.run_tensorboard()


if __name__ == '__main__':
    main()
