"""Module containing the class Trainer to handle the fitting of neural networks.

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
    * Add method to save models to file
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import importlib
from Minerva.utils import visutils, utils
import numpy as np
import torch
from torchinfo import summary
from itertools import islice
from alive_progress import alive_bar


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class Trainer:
    """Helper class to handle the entire fitting and evaluation of a model.

    Attributes:
        params (dict): Dictionary describing all the parameters that define how the model will be constructed, trained
            and evaluated. These should be defined via config YAML files.
        model: Model to be fitted of a class contained within Minerva.models.
        max_epochs (int): Number of epochs to train the model for.
        batch_size (int): Size of each batch of samples supplied to the model.
        loaders (dict[DataLoader]): Dictionary containing DataLoaders for each dataset.
        n_batches (dict[int]): Dictionary of the number of batches to supply to the model for train, validation and
            testing.
        metrics (dict): Dictionary to hold the loss and accuracy results from training, validation and testing.
        device: The CUDA device on which to fit the model.
    """

    def __init__(self, loaders, n_batches: dict, **params):
        """Initialises the Trainer.

        Args:
            loaders (dict[DataLoader]): Dictionary containing DataLoaders for each dataset.
            n_batches (dict): Dictionary of the number of batches to supply to the model for train, validation and
                testing.
        Keyword Args:
            results (list[str]): Path to the results directory to save plots to.
            model_name (str): Name of the model to be used in filenames of results.
            batch_size (int): Size of each batch of samples supplied to the model.
            max_epochs (int): Number of epochs to train the model for.
        """
        self.params = params

        # Creates model (and loss function) from specified parameters in params.
        self.model = self.make_model()

        self.max_epochs = params['hyperparams']['max_epochs']
        self.batch_size = params['hyperparams']['params']['batch_size']
        self.loaders = loaders
        self.n_batches = n_batches

        # Creates a dict to hold the loss and accuracy results from training, validation and testing.
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],

            'train_acc': [],
            'val_acc': [],
            'test_acc': []
        }

        # Creates and sets the optimiser for the model.
        self.make_optimiser()

        self.device = utils.get_cuda_device()

        # Transfer to GPU
        self.model.to(self.device)

        # Print model summary
        summary(self.model, input_size=(self.batch_size, *self.model.input_shape))

    def make_model(self):
        """Creates a model from the parameters specified by config.

        Returns:
            Initialised model.
        """
        model_params = self.params['hyperparams']['model_params']

        # Gets the torch optimiser library.
        models_module = importlib.import_module('Minerva.models')

        # Gets the optimiser requested by config parameters.
        model = getattr(models_module, self.params['model_name'].split('-')[0])

        # Initialise model
        return model(self.make_criterion(), **model_params)

    def make_criterion(self):
        """Creates a PyTorch loss function based on config parameters.

        Returns:
            Initialised PyTorch loss function specified by config parameters.
        """
        # Gets the torch neural network library.
        module = importlib.import_module('torch.nn')

        # Gets the loss function requested by config parameters.
        criterion = getattr(module, self.params['hyperparams']['loss_params'].pop('name'))
        
        return criterion()

    def make_optimiser(self):
        """Creates a PyTorch optimiser based on config parameters and sets optimiser."""

        # Gets the torch optimiser library.
        opt_module = importlib.import_module('torch.optim')

        # Gets the optimiser requested by config parameters.
        optimiser = getattr(opt_module, self.params['hyperparams']['optim_params'].pop('name'))

        # Constructs and sets the optimiser for the model based on supplied config parameters.
        self.model.set_optimiser(optimiser(self.model.parameters(), **self.params['hyperparams']['optim_params']))

    def epoch(self, mode):
        """All encompassing function for any type of epoch, be that train, validation or testing.

        Args:
            mode (str): Either train, val or test. Defines the type of epoch to run on the model.

        Returns:
            If a test epoch, returns the predicted and ground truth labels and the patch IDs supplied to the model.
        """
        # Initialises variables to hold overall epoch results.
        total_loss = 0.0
        total_correct = 0.0
        test_labels = []
        test_predictions = []
        test_ids = []

        # Initialises a progress bar for the epoch.
        with alive_bar(self.n_batches[mode], bar='blocks') as bar:
            # Sets the model up for training or evaluation modes
            if mode is 'train':
                self.model.train()
            else:
                self.model.eval()

            # Core of the epoch. Gets batches from the appropriate loader.
            for x_batch, y_batch, patch_id in islice(self.loaders[mode], self.n_batches[mode]):

                # Transfer to GPU.
                x, y = x_batch.to(self.device), y_batch.to(self.device)

                # Runs a training epoch.
                if mode is 'train':
                    loss, z = self.model.training_step(x, y)

                    total_loss += loss.item()
                    total_correct += (torch.argmax(z, 1) == y).sum().item()

                # Runs a validation epoch.
                elif mode is 'val':
                    loss, z = self.model.validation_step(x, y)

                    total_loss += loss.item()
                    total_correct += (torch.argmax(z, 1) == y).sum().item()

                # Runs a testing epoch.
                elif mode is 'test':
                    loss, z = self.model.testing_step(x, y)

                    total_loss += loss.item()
                    total_correct += (torch.argmax(z, 1) == y).sum().item()
                    test_predictions.append(np.array(torch.argmax(z, 1).cpu()))
                    test_labels.append(np.array(y.cpu()))
                    test_ids.append(patch_id)

                # Updates progress bar that sample has been processed.
                bar()

        # Updates metrics with epoch results.
        self.metrics['{}_loss'.format(mode)].append(total_loss / self.n_batches[mode])
        self.metrics['{}_acc'.format(mode)].append(total_correct / (self.n_batches[mode] * self.batch_size))

        if mode is 'test':
            return test_predictions, test_labels, test_ids
        else:
            return

    def fit(self):
        """Fits the model by running max_epochs number of training and validation epochs."""
        for epoch in range(self.max_epochs):
            print(f'Epoch: {epoch + 1}/{self.max_epochs}')

            self.epoch('train')
            self.epoch('val')

            print('Training loss: {} | Validation Loss: {}'.format(self.metrics['train_loss'][epoch],
                                                                   self.metrics['val_loss'][epoch]))
            print('Train Accuracy: {}% | Validation Accuracy: {}% \n'.format(self.metrics['train_acc'][epoch] * 100.0,
                                                                             self.metrics['val_acc'][epoch] * 100.0))

    def test(self, plots, save=True):
        """Tests the model by running a testing epoch then taking the results and orchestrating the plotting and
        analysis of them.

        Args:
            plots (dict): Dictionary defining which plots of the test results to create.
            save (bool): Optional; Determines whether or not to save the plots created to file.

        Returns:
            Test predicted and ground truth labels along with the patch IDs supplied to the model during testing.
        """
        print('\r\nTESTING')
        predictions, labels, ids = self.epoch('test')

        print('Test Loss: {} | Test Accuracy: {}% \n'.format(self.metrics['test_loss'][0],
                                                             self.metrics['test_acc'][0] * 100.0))

        submetrics = {k: self.metrics[k] for k in ('train_loss', 'val_loss', 'train_acc', 'val_acc')}

        visutils.plot_results(submetrics, plots, np.array(predictions).flatten(), np.array(labels).flatten(),
                              save=save, show=False, model_name=self.params['model_name'],
                              results_dir=self.params['dir']['results'])

        return predictions, labels, ids
