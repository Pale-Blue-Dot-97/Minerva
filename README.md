# Minerva
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Pale-Blue-Dot-97/Minerva) ![GitHub](https://img.shields.io/github/license/Pale-Blue-Dot-97/Minerva)

Minerva is a package to aid in the building, fitting and testing of neural network models on land cover data.
 
## Installation

TBC

## Requirements

Currently, `Minerva` only supports the use of [Radiant MLHub LandCoverNetV1](http://registry.mlhub.earth/10.34911/rdnt.d2ce8i/) 
dataset. Included in `Minerva\bin` is `Landcovernet_Download_API.py`, a script implementing the example implementation 
of Radiant Earth's download API that can download the desired LandCoverNetV1 data. Users will require an API key that 
can be obtained from Radiant Earth upon sign-up. This key should be placed in a file named `API Key.txt` to use the API.

Required Python modules for `Minerva` are stated in `requirements.txt`.

## Usage
Minerva provides the modules to define `models` to fit and test, `loaders` to pre-process, load and parse data, 
and a `Trainer` to handle all aspects of a model fitting. 

Start by importing the desired model class defined in `models`.

```python
from Minerva.models import MLP
from Minerva.trainer import Trainer
import torch
import yaml

# Import config file
config_path = '../../config/config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

with open(config['dir']['data_config']) as file:
    dataset_config = yaml.safe_load(file)

# Defines size of the images to determine the number of batches
image_size = dataset_config['data_specs']['image_size']

image_len = image_size[0] * image_size[1]

# Parameters
params = config['hyperparams']['params']

wheel_size = image_len

# Number of epochs to train model over
max_epochs = config['hyperparams']['max_epochs']

model_params = config['hyperparams']['model_params']

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Initialise model
model = MLP(criterion, **model_params)

# Define optimiser
optimiser = torch.optim.SGD(model.parameters(), lr=config['hyperparams']['optimiser_params']['learning_rate'])
```

WIP!

See `Minerva\bin\MinervaPercep.py` as an example script implementing `Minerva`.

## License
Minerva is distributed under a [GNU GPLv3 License](https://choosealicense.com/licenses/gpl-3.0/).

## Authors

Created by Harry Baker as part of a project towards the award of a PhD Computer Science from the 
University of Southampton. Funded by the Ordnance Survey Ltd. 

I'd like to acknowledge the invaluable supervision and contributions of Dr Jonathon Hare and 
Dr Isabel Sargent towards this work.

## Project Status

This project is still very much a WIP alpha state.