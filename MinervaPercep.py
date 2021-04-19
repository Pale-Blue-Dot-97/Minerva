"""MinervaPercep

Script to create a simple MLP to classify land cover of the images in the LandCoverNet V1 dataset

TODO:
    * Add method to save models to file

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from utils import utils
from models import MLP
from loaders import BalancedBatchLoader, BatchLoader
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torchsummary import summary
from itertools import islice
from alive_progress import alive_bar

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

image_len = image_size[0] * image_size[1]

classes = dataset_config['classes']

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = config['hyperparams']['params']

wheel_size = image_len

# Number of epochs to train model over
max_epochs = config['hyperparams']['max_epochs']

model_params = config['hyperparams']['model_params']


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
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

    trainer = Trainer(model=model, max_epochs=max_epochs, batch_size=params['batch_size'], optimiser=optimiser)
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()
