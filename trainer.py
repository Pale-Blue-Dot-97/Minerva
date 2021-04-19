"""trainer

Module containing the class Trainer to handle the fitting of neural networks

TODO:
    * Fully document
    * Add method to save models to file

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from utils import utils
import numpy as np
import torch
from torchsummary import summary
from torch.backends import cudnn
from itertools import islice
from alive_progress import alive_bar


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class Trainer:
    def __init__(self, model, max_epochs, batch_size, optimiser, loaders, n_batches, device=None):
        self.model = model
        self.max_epochs = max_epochs
        self.batch_size = batch_size

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

        if device is None:
            # CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            cudnn.benchmark = True

        self.device = device

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
                x, y = x_batch.to(self.device), y_batch.to(self.device)

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
