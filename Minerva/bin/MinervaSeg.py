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
from Minerva.utils import utils, visutils
import Minerva.loaders as loaders
from Minerva.trainer import Trainer
from matplotlib.colors import ListedColormap
import numpy as np
import osr
from alive_progress import alive_bar

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

config, aux_configs = utils.load_configs(config_path)
dataset_config = aux_configs['data_config']

# Parameters
params = config['hyperparams']['params']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def seg_plot(z, y, test_ids: list, classes: dict, colours: dict,
             frac: float = 0.05, figdim: tuple = (9.3, 10.5)) -> None:
    """Custom function for pre-processing the outputs from image segmentation testing for data visualisation.

    Args:
        z (list[float]): Predicted segmentation masks by the network.
        y (list[float]): Corresponding ground truth masks.
        test_ids (list[str]): Corresponding patch IDs for the test data supplied to the network.
        classes (dict): Dictionary mapping class labels to class names.
        colours (dict): Dictionary mapping class labels to colours.
        frac (float): Optional; Fraction of patch samples to plot.
        figdim (tuple): Optional; Figure (height, width) in inches.

    Returns:
        None
    """
    z = np.array(z)
    y = np.array(y)

    z = np.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3]))
    y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2], y.shape[3]))
    test_ids = np.array(test_ids).flatten()

    # Create a new projection system in lat-lon
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(dataset_config['co_sys']['id'])

    print('PRODUCING PREDICTED MASKS')
    # Initialises a progress bar for the epoch.
    with alive_bar(int(frac*len(test_ids)), bar='blocks') as bar:
        # Plots the predicted versus ground truth labels for all test patches supplied.
        for i in range(int(frac*len(test_ids))):
            visutils.prediction_plot(z[i], y[i], test_ids[i],
                                     exp_id=config['model_name'],
                                     new_cs=new_cs,
                                     classes=classes,
                                     figdim=figdim,
                                     show=False,
                                     cmap_style=ListedColormap(colours.values(), N=len(colours)))

            bar()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    datasets, n_batches, class_dist, ids, new_classes, new_colours = loaders.make_datasets(**config)

    config['hyperparams']['model_params']['n_classes'] = len(new_classes)
    config['classes'] = new_classes
    config['colours'] = new_colours

    trainer = Trainer(loaders=datasets, n_batches=n_batches, class_dist=class_dist, **config)

    trainer.fit()

    z, y, test_ids = trainer.test({'History': True, 'Pred': True, 'CM': True}, save=True)

    seg_plot(z, y, test_ids, config['classes'], config['colours'])

    trainer.close()

    trainer.run_tensorboard()


if __name__ == '__main__':
    main()
