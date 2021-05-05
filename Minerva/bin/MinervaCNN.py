"""Script to create a simple CNN to classify land cover of the images in the LandCoverNet V1 dataset.

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

Created under a project funded by the Ordnance Survey Ltd


TODO:
    * Add optimiser selection logic
    * Add arg parsing from CLI
    * Add model selection logic
    * Add loss function selection logic
    * Add ability to conduct hyper-parameter iterative variation experimentation
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.models import CNN
import Minerva.loaders as loaders
from Minerva.trainer import Trainer
import yaml
import torch
from torch.backends import cudnn

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = config['hyperparams']['params']


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    model_params = config['hyperparams']['model_params']

    # Initialise model
    model = CNN(criterion, **model_params)

    # Define optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=config['hyperparams']['optimiser_params']['learning_rate'])

    datasets, n_batches, _, ids = loaders.make_datasets(cnn=True, params=params)

    trainer = Trainer(model=model, optimiser=optimiser, loaders=datasets, n_batches=n_batches, device=device, **config)

    trainer.fit()

    trainer.test({'History': True, 'Pred': True, 'CM': True}, save=False)


if __name__ == '__main__':
    main()
