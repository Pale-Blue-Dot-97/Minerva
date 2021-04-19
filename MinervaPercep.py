"""MinervaPercep

Script to create a simple MLP to classify land cover of the images in the LandCoverNet V1 dataset

TODO:
    * Add method to save models to file

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import utils
from models import MLP
import yaml
import random
from abc import ABC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.backends import cudnn
from torchsummary import summary
from itertools import cycle, chain, islice
from alive_progress import alive_bar
from collections import deque

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = 'config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

with open(config['dir']['data_config']) as file:
    dataset_config = yaml.safe_load(file)

# Defines size of the images to determine the number of batches
image_size = dataset_config['data_specs']['image_size']

flattened_image_size = image_size[0] * image_size[1]

classes = dataset_config['classes']

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = config['hyperparams']['params']

wheel_size = flattened_image_size

# Number of epochs to train model over
max_epochs = config['hyperparams']['max_epochs']

model_params = config['hyperparams']['model_params']


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class BalancedBatchLoader(IterableDataset, ABC):
    """Adaptation of BatchLoader to load data with perfect class balance
    """

    def __init__(self, class_streams, batch_size=32):
        """Initialisation

        Args:
            class_streams (pandas.DataFrame): DataFrame with a column of patch IDs for each class
            batch_size (int): Sets the number of samples in each batch
        """
        self.streams_df = class_streams
        self.batch_size = batch_size

        # Dict to hold a `wheel' for each class
        self.wheels = {}

        # Initialise each wheel with a maximum length of wheel_size global parameter
        for cls in self.streams_df.columns.to_list():
            self.wheels[cls] = deque(maxlen=wheel_size)

        # Loads the patches from the row of IDs supplied into a pandas.Series of pandas.DataFrames
        patches = pd.Series([self.load_patch_df(patch_id) for patch_id in self.streams_df.sample(frac=1).iloc[0]])
        patches.apply(self.refresh_wheels)

        # Checks if wheel is empty after adding to wheel
        for cls in self.streams_df.columns.to_list():
            if not self.wheels[cls]:
                print('EMPTY WHEEL {}!'.format(cls))
                self.emergency_fill(cls)

    def load_patch_df(self, patch_id):
        """Loads a patch using patch ID from disk into a Pandas.DataFrame and returns

        Args:
            patch_id (str): ID for patch to be loaded

        Returns:
            df (pandas.DataFrame): Patch loaded into a DataFrame
        """
        # Initialise DataFrame object
        df = pd.DataFrame()

        # Load patch from disk and create time-series pixel stacks
        patch = utils.make_time_series(patch_id)

        # Reshape patch
        patch = patch.reshape((patch.shape[0] * patch.shape[1], patch.shape[2] * patch.shape[3]))

        # Loads accompanying labels from file and flattens
        labels = utils.lc_load(patch_id).flatten()

        # Wraps each pixel stack in an numpy.array, appends to a list and adds as a column to df
        df['PATCH'] = [np.array(pixel) for pixel in patch]

        # Adds labels as a column to df
        df['LABELS'] = labels

        return df

    def load_patches(self, row):
        """ Loads the patches associated with the patch IDs in row as pandas.DataFrames nested into a pandas.Series
        object

        Args:
            row (pandas.Series): A row of patch IDs

        Returns:
            (pandas.Series): Series of DataFrames of patches
        """
        return pd.Series([self.load_patch_df(row[1][cls]) for cls in self.streams_df.columns.to_list()])

    def refresh_wheels(self, patch_df):
        for cls in self.streams_df.columns.to_list():
            for pixel in patch_df['PATCH'].loc[patch_df['LABELS'] == cls]:
                self.wheels[cls].appendleft(pixel.flatten())

    # THIS IS BODGY AF
    def emergency_fill(self, cls):
        print('EMERGENCY FILL INITIATED')
        patches = pd.Series([self.load_patch_df(patch_id) for patch_id in self.streams_df[cls].sample(frac=1)])

        for patch in patches:
            print('ATTEMPTING TO INIT WHEEL')
            for pixel in patch['PATCH'].loc[patch['LABELS'] == cls]:
                self.wheels[cls].appendleft(pixel.flatten())

            if self.wheels[cls]:
                print('CRISIS OVER')
                return

    def process_data(self, row):
        """Loads and processes patches into wheels for each class and yields from them,
        periodically refreshing the wheels with new data

        Args:
            row (pandas.Series): Randomly selected row of patch IDs, one for each class

        Yields:
            x (torch.Tensor): A data sample as tensor
            y (torch.Tensor): Corresponding label as int tensor
        """
        # Iterates for the flattened length of a patch and yields x and y for each class from their respective wheels
        for i in range(flattened_image_size):
            if i == 0:
                # Loads the patches from the row of IDs supplied into a pandas.Series of pandas.DataFrames
                patches = self.load_patches(row)
                patches.apply(self.refresh_wheels)

            # For every class in the dataset, rotate the corresponding wheel and yield the pixel stack from position [0]
            for cls in self.streams_df.columns.to_list():
                # Rotate current class's wheel 1 turn
                self.wheels[cls].rotate(1)

                # Yield pixel stack at position [0] for this class's wheel and the corresponding class label
                # i.e this class number as a tensor int
                yield torch.tensor(self.wheels[cls][0].flatten(), dtype=torch.float), \
                      torch.tensor(cls, dtype=torch.long)

    def get_stream(self, streams_df):
        return chain.from_iterable(map(self.process_data, streams_df.iterrows()))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return full ID stream
        if worker_info is None:
            return self.get_stream(self.streams_df)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate fraction of dataset per worker
            per_worker = int(np.math.ceil(1.0 / float(worker_info.num_workers)))

            # Return a random sample of the patch IDs of fractional size per worker
            # and using random seed modulated by the worker ID
            return self.get_stream(self.streams_df.sample(frac=per_worker, random_state=42 * worker_info.id,
                                                          replace=False, axis=0))


class BatchLoader(IterableDataset, ABC):
    """
    Source: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """

    def __init__(self, patch_ids, batch_size):
        self.patch_ids = patch_ids
        self.batch_size = batch_size

    def process_data(self, patch_id):
        patch = utils.make_time_series(patch_id)

        x = torch.tensor([pixel.flatten() for pixel in patch.reshape(-1, *patch.shape[-2:])], dtype=torch.float)
        y = torch.tensor(np.array(utils.lc_load(patch_id), dtype=np.int64).flatten(), dtype=torch.long)

        for i in range(len(y)):
            yield x[i], y[i]

    def get_stream(self, patch_ids):
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # If single threaded process, return all patch IDs
        if worker_info is None:
            return self.get_stream(self.patch_ids)

        # If multi-threaded, split patch IDs between workers
        else:
            # Calculate number of patch IDs of the dataset per worker
            per_worker = int(np.math.ceil(len(self.patch_ids) / float(worker_info.num_workers)))

            # Set random seed modulated by the worker ID
            random.seed(42 * worker_info.id)

            # Return a random sample of the patch IDs of size per worker
            return self.get_stream(random.sample(self.patch_ids, per_worker))


class Trainer:
    def __init__(self, model, max_epochs, batch_size, optimiser):
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size

        loaders, n_batches, _ = make_loaders(balance=True)
        self.loaders = loaders
        self.n_batches = n_batches
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],

            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }

        self.model.set_optimiser(optimiser)

        # Transfer to GPU
        model.to(device)

        # Print model summary
        summary(model, (1, self.model.input_size))

    def epoch(self, mode):
        total_loss = 0.0
        total_correct = 0.0
        test_labels = []
        test_predictions = []
        with alive_bar(self.n_batches[mode], bar='blocks') as bar:
            if mode is 'train':
                self.model.train()
            else:
                self.model.eval()

            for x_batch, y_batch in islice(self.loaders[mode], self.n_batches[mode]):
                # Transfer to GPU
                x, y = x_batch.to(device), y_batch.to(device)

                if mode is 'train':
                    loss, z = self.model.training_step(x, y)

                    total_loss += loss.item()
                    total_correct += (torch.argmax(z, 1) == y).sum().item()

                elif mode is 'val':
                    loss, z = self.model.validation_step(x, y)

                    total_loss += loss.item()
                    total_correct += (torch.argmax(z, 1) == y).sum().item()

                elif mode is 'test':
                    loss, z = self.model.testing_step(x, y)

                    total_loss += loss.item()
                    total_correct += (torch.argmax(z, 1) == y).sum().item()
                    test_predictions.append(np.array(torch.argmax(z, 1).cpu()))
                    test_labels.append(np.array(y.cpu()))
                bar()

        self.metrics['{}_loss'.format(mode)].append(total_loss / self.n_batches[mode])
        self.metrics['{}_acc'.format(mode)].append(total_correct / (self.n_batches[mode] * self.batch_size))

        if mode is 'test':
            return test_predictions, test_labels
        else:
            return

    def fit(self):
        for epoch in range(self.max_epochs):
            print(f'Epoch: {epoch + 1}/{self.max_epochs}')

            self.epoch('train')
            self.epoch('val')

            print('Training loss: {} | Validation Loss: {}'.format(self.metrics['train_loss'][epoch],
                                                                   self.metrics['val_loss'][epoch]))
            print('Train Accuracy: {}% | Validation Accuracy: {}% \n'.format(self.metrics['train_acc'][epoch] * 100.0,
                                                                             self.metrics['val_acc'][epoch] * 100.0))

    def test(self):
        print('\r\nTESTING')
        predictions, labels = self.epoch('test')

        print('Test Loss: {} | Test Accuracy: {}% \n'.format(self.metrics['test_loss'][0],
                                                             self.metrics['test_acc'][0] * 100.0))

        submetrics = {k: self.metrics[k] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}
        utils.plot_results(submetrics, np.array(predictions).flatten(), np.array(labels).flatten(),
                           save=True, show=False)


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
        datasets['train'] = BalancedBatchLoader(train_stream, batch_size=params['batch_size'])
        datasets['val'] = BalancedBatchLoader(val_stream, batch_size=params['batch_size'])

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

    trainer = Trainer(model=model, max_epochs=max_epochs, batch_size=params['batch_size'], optimiser=optimiser)
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()
