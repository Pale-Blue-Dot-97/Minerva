"""MinervaPercep

Script to create a simple MLP to classify land cover of the images in the LandCoverNet V1 dataset

TODO:
    * Generalise make_loaders and transfer to trainer
    * Add optimiser selection logic
    * Add arg parsing from CLI
    * Add model selection logic
    * Add loss function selection logic
    * Add ability to conduct hyper-parameter iterative variation experimentation

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from utils import utils
from models import MLP
from loaders import BalancedBatchLoader, BatchLoader
from trainer import Trainer
import yaml
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from sklearn.model_selection import train_test_split


# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = 'config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

with open(config['dir']['data_config']) as file:
    dataset_config = yaml.safe_load(file)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Defines size of the images to determine the number of batches
image_size = dataset_config['data_specs']['image_size']

image_len = image_size[0] * image_size[1]

# Parameters
params = config['hyperparams']['params']

wheel_size = image_len

# Number of epochs to train model over
max_epochs = config['hyperparams']['max_epochs']

model_params = config['hyperparams']['model_params']


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def class_weighting(class_dist):
    # Finds total number of samples to normalise data
    n_samples = 0
    for mode in class_dist:
        n_samples += mode[1]

    return torch.tensor([(1 - (mode[1] / n_samples)) for mode in class_dist], device=device)


def make_loaders(patch_ids=None, split=(0.7, 0.15, 0.15), seed=42, shuffle=True, plot=False, balance=False,
                 p_dist=False):
    """

    Args:
        patch_ids:
        split:
        seed:
        shuffle (bool):
        plot (bool):
        balance (bool):
        p_dist (bool):

    Returns:
        loaders (dict):
        n_batches (dict):
        class_dist (Counter):

    """
    # Fetches all patch IDs in the dataset
    if patch_ids is None:
        patch_ids = utils.patch_grab()

    # Splits the dataset into train and val-test
    train_ids, val_test_ids = train_test_split(patch_ids, train_size=split[0], test_size=(split[1] + split[2]),
                                               shuffle=shuffle, random_state=seed)

    # Splits the val-test dataset into validation and test
    val_ids, test_ids = train_test_split(val_test_ids, train_size=(split[1] / (split[1] + split[2])),
                                         test_size=(split[2] / (split[1] + split[2])), shuffle=shuffle,
                                         random_state=seed)

    if p_dist:
        print('\nTrain: \n', utils.find_subpopulations(train_ids, plot=plot))
        print('\nValidation: \n', utils.find_subpopulations(val_ids, plot=plot))
        print('\nTest: \n', utils.find_subpopulations(test_ids, plot=plot))

    datasets = {}
    n_batches = {}

    if balance:
        train_stream = utils.make_sorted_streams(train_ids)
        val_stream = utils.make_sorted_streams(val_ids)

        # Define datasets for train, validation and test using BatchLoader
        datasets['train'] = BalancedBatchLoader(train_stream, batch_size=params['batch_size'],
                                                wheel_size=wheel_size, patch_len=image_len)
        datasets['val'] = BalancedBatchLoader(val_stream, batch_size=params['batch_size'],
                                              wheel_size=wheel_size, patch_len=image_len)

        n_batches['train'] = utils.num_batches(len(train_stream.columns) * len(train_stream))
        n_batches['val'] = utils.num_batches(len(val_stream.columns) * len(val_stream))

    if not balance:
        # Define datasets for train, validation and test using BatchLoader
        datasets['train'] = BatchLoader(train_ids, batch_size=params['batch_size'])
        datasets['val'] = BatchLoader(val_ids, batch_size=params['batch_size'])

        n_batches['train'] = utils.num_batches(len(train_ids))
        n_batches['val'] = utils.num_batches(len(val_ids))

    datasets['test'] = BatchLoader(test_ids, batch_size=params['batch_size'])
    n_batches['test'] = utils.num_batches(len(test_ids))

    # Create train, validation and test batch loaders and pack into dict
    loaders = {'train': DataLoader(datasets['train'], **params),
               'val': DataLoader(datasets['val'], **params),
               'test': DataLoader(datasets['test'], **params)}

    class_dist = utils.find_subpopulations(patch_ids, plot=False)

    return loaders, n_batches, class_dist


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Initialise model
    model = MLP(criterion, **model_params)

    # Define optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=config['hyperparams']['optimiser_params']['learning_rate'])

    loaders, n_batches, _ = make_loaders(balance=True)

    trainer = Trainer(model=model, max_epochs=max_epochs, batch_size=params['batch_size'], optimiser=optimiser,
                      loaders=loaders, n_batches=n_batches, device=device)
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()
