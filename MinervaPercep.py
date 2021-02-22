"""MinervaPercep

Script to create a simple MLP to classify land cover of the images in the LandCoverNet V1 dataset

TODO:
    * Construct a simple MLP
    * Add methods to visualise and analyse results of training and testing
    * Add method to save models to file

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import glob
import random
from abc import ABC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.backends import cudnn
from torchsummary import summary
from itertools import cycle, chain, islice
import Radiant_MLHub_DataVis as rdv
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from collections import Counter, deque
import seaborn as sns
import tensorflow as tf
# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# Path to directory holding dataset
data_dir = 'landcovernet'

# Prefix to every patch ID in every patch directory name
patch_dir_prefix = 'ref_landcovernet_v1_labels_'

# Band IDs of SENTINEL-2 images contained in the LandCoverNet dataset
band_ids = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# Defines size of the images to determine the number of batches
image_size = (256, 256)

classes = rdv.RE_classes

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

# Parameters
params = {'batch_size': 16,
          'num_workers': 1}

wheel_size = 2 * 32

# Number of epochs to train model over
max_epochs = 1


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
class MLP(torch.nn.Module, ABC):
    """
    Simple class to construct a Multi-Layer Perceptron (MLP)
    """
    def __init__(self, input_size, n_classes, hidden_sizes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = n_classes
        self.hidden_sizes = hidden_sizes
        self.layers = torch.nn.ModuleList()

        for i in range(len(hidden_sizes)):
            if i is 0:
                self.layers.append(torch.nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(hidden_sizes[-1], n_classes))

    def forward(self, x):
        """Performs a forward pass of the network

        Args:
            self
            x (torch.Tensor):

        Returns:

        """

        y = x

        for layer in self.layers:
            y = layer(y)

        return y


class BalancedBatchLoader(IterableDataset, ABC):
    """
    Source: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """
    def __init__(self, class_streams, batch_size=32, wheel_size=10*32):
        self.streams_df = class_streams
        self.batch_size = batch_size
        self.wheels = {}

        for cls in self.streams_df.columns.to_list():
            self.wheels[cls] = deque(maxlen=wheel_size)

        for cls in self.streams_df.columns.to_list():
            while len(self.wheels[cls]) is not self.wheels[cls].maxlen:
                patch_id = self.streams_df.sample(frac=1).reset_index(drop=True)[cls][0]

                patch = make_time_series(patch_id)
                patch = patch.reshape(-1, *patch.shape[-2:])
                patch = patch.reshape(patch.shape[0], -1)

                labels = lc_load(patch_id).flatten()

                self.add_to_wheels(patch, labels)

    @property
    def shuffled_data_list(self):
        return self.streams_df.sample(frac=1).reset_index(drop=True)

    def add_to_wheels(self, patch, labels):
        for cls in self.streams_df.columns.to_list():
            for pixel in patch[np.where(labels == cls)]:
                self.wheels[cls].appendleft(pixel.flatten())

    def process_data(self, row):
        print('processing')
        for cls in self.streams_df.columns.to_list():
            patch_id = row[1][cls]

            patch = make_time_series(patch_id)
            patch = patch.reshape(-1, *patch.shape[-2:])
            patch = patch.reshape(patch.shape[0], -1)

            labels = lc_load(patch_id).flatten()

            self.add_to_wheels(patch, labels)

            self.wheels[cls].rotate(1)

            yield torch.tensor(self.wheels[cls][0].flatten(), dtype=torch.float), torch.tensor(cls, dtype=torch.long)

    def get_stream(self, streams_df):
        print('get stream')
        return chain.from_iterable(map(self.process_data, streams_df.iterrows()))

    def get_streams(self):
        print('get streams')
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        print('iter')
        return self.get_stream(self.streams_df)


class BatchLoader(IterableDataset, ABC):
    """
    Source: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
    """
    def __init__(self, patch_ids, batch_size):
        self.patch_ids = patch_ids
        self.batch_size = batch_size

    @property
    def shuffled_data_list(self):
        return random.sample(self.patch_ids, len(self.patch_ids))

    def process_data(self, patch_id):
        print('processing')
        patch = make_time_series(patch_id)

        x = torch.tensor([pixel.flatten() for pixel in patch.reshape(-1, *patch.shape[-2:])], dtype=torch.float)
        y = torch.tensor(np.array(lc_load(patch_id), dtype=np.int64).flatten(), dtype=torch.long)

        for i in range(len(y)):
            yield x[i], y[i]

    def get_stream(self, patch_ids):
        print('get stream')
        return chain.from_iterable(map(self.process_data, cycle(patch_ids)))

    def get_streams(self):
        print('get streams')
        return zip(*[self.get_stream(self.shuffled_data_list) for _ in range(self.batch_size)])

    def __iter__(self):
        print('iter')
        return self.get_stream(self.patch_ids)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def prefix_format(patch_id, scene):
    """Formats a string representing the prefix of a path to any file in a scene

    Args:
        patch_id (str): Unique patch ID
        scene (str): Date of scene in YY_MM_DD format

    Returns:
        prefix (str): Prefix of path to any file in a given scene
    """
    return '%s/%s%s/%s/%s_%s' % (data_dir, patch_dir_prefix, patch_id, scene, patch_id,
                                 rdv.datetime_reformat(scene, '%Y_%m_%d', '%Y%m%d'))


def scene_grab(patch_id):
    """Finds and loads all CLDs for a given patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        scenes (list): List of CLD masks for each scene
        scene_names (list): List of scene dates in YY_MM_DD

    """
    # Get the name of all the directories for this patch
    scene_dirs = glob.glob('%s/%s%s/*/' % (data_dir, patch_dir_prefix, patch_id))

    # Extract the scene names (i.e the dates) from the paths
    scene_names = [(scene.partition('\\')[2])[:-1] for scene in scene_dirs]

    # List to hold scenes
    scenes = []

    # Finds and appends each CLD of each scene of a patch to scenes
    for date in scene_names:
        scenes.append(rdv.load_array('%s_CLD_10m.tif' % prefix_format(patch_id, date), 1))

    return scenes, scene_names


def lc_load(patch_id):
    """Loads the LC labels for a given patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        LC_label (list): 2D array containing LC labels for each pixel of a patch

    """
    return rdv.load_array('%s/%s%s/%s_2018_LC_10m.tif' % (data_dir, patch_dir_prefix, patch_id, patch_id), 1)


def labels_to_ohe(labels, n_classes):
    """Convert an iterable of indices to one-hot encoded labels.

    Args:
        labels (list[int]): List of class number labels to be converted to OHE
        n_classes (int): Number of classes to determine length of OHE label

    Returns:
        Labels in OHE form

    """
    targets = np.array(labels).reshape(-1)
    return np.eye(n_classes)[targets]


def find_patch_modes(patch_id):
    """Finds the distribution of the classes within this patch

    Args:
        patch_id (str): Unique patch ID

    Returns:
        (Counter): Modal distribution of classes in the patch provided
    """
    return Counter(np.array(lc_load(patch_id)).flatten()).most_common()


def class_frac(patch):
    """Computes the fractional sizes of the classes of the given patch and returns a dict of the results

    Args:
        patch (pandas.Series): Row of DataFrame representing the entry for a patch

    Returns:
        new_columns (dict): Dictionary with keys as class numbers and associated values of fractional size of class
                            plus a key-value pair for the patch ID
    """
    new_columns = {'PATCH': patch['PATCH']}
    for mode in patch['MODES']:
        new_columns[mode[0]] = mode[1]/(image_size[0] * image_size[1])

    return new_columns


def make_sorted_streams(patch_ids):
    """Creates a DataFrame with columns of patch IDs sorted for each class by class size in those patches

    Args:
        patch_ids (list[str]):

    Returns:
        streams_df (pandas.DataFrame): Database of list of patch IDs sorted by fractional sizes of class labels

    """
    df = pd.DataFrame()
    df['PATCH'] = patch_ids

    df['MODES'] = df['PATCH'].apply(find_patch_modes)

    df = pd.DataFrame([row for row in df.apply(class_frac, axis=1)])

    df.fillna(0, inplace=True)

    class_dist = find_subpopulations(df['PATCH'], plot=False)

    stream_size = int(len(df['PATCH'])/len(classes))

    streams = {}

    for mode in reversed(class_dist):
        stream = df.sort_values(by=mode[0], ascending=False)['PATCH'][:stream_size]
        streams[mode[0]] = stream.tolist()
        df.drop(stream.index, inplace=True)

        print('{} Stream Dist:'.format(mode[0]), find_subpopulations(stream.tolist()))

        if mode[0] == 4:
            for patch in stream.tolist():
                modes2 = Counter(lc_load(patch).flatten()).most_common()
                for mode2 in modes2:
                    if mode2[0] == 4:
                        print(patch)

    streams_df = pd.DataFrame(streams)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(streams_df)

    print(len(set(streams_df.to_numpy().flatten())) == len(streams_df.to_numpy().flatten()))

    return streams_df


def cloud_cover(scene):
    """Calculates percentage cloud cover for a given scene based on its scene CLD

    Args:
        scene (numpy.ndarray):

    Returns:
        (float): Percentage cloud cover
    """
    return np.sum(scene) / scene.size


def month_sort(df, month):
    """Finds the the scene with the lowest cloud cover in a given month

    Args:
        df (pandas.DataFrame): Dataframe containing all scenes and their cloud cover percentages
        month (str): Month of a year to sort

    Returns:
        (str): Date of the scene with the lowest cloud cover percentage for the given month
    """
    return df.loc[month].sort_values(by='COVER')['DATE'][0]


def scene_selection(df):
    """Selects the 24 best scenes of a patch based on REF's 2-step selection criteria

    Args:
        df (pandas.DataFrame): Dataframe containing all scenes and their cloud cover percentages

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format
    """
    # Step 1: Find scene with lowest cloud cover percentage in each month
    step1 = []
    for month in range(1, 13):
        step1.append(month_sort(df, '%d-2018' % month))

    # Step 2: Find the 12 scenes with the lowest cloud cover percentage of the remaining scenes
    df.drop(index=pd.to_datetime(step1, format='%Y_%m_%d'), inplace=True)
    step2 = df.sort_values(by='COVER')['DATE'][:12].tolist()

    # Return 24 scenes selected by the 2-step REF criteria
    return step1 + step2


def find_best_of(patch_id):
    """Finds the 24 scenes sorted by cloud cover according to REF's 2-step criteria using scene_selection()

    Args:
        patch_id (str): Unique patch ID

    Returns:
        scene_names (list): List of 24 strings representing dates of the 24 selected scenes in YY_MM_DD format
    """
    # Creates a DataFrame
    patch = pd.DataFrame()

    # Using scene_grab(), gets all the scene CLDs and dates for the given patch and adds to DataFrame
    patch['SCENE'], patch['DATE'] = scene_grab(patch_id)

    # Calculates the cloud cover percentage for every scene and adds to DataFrame
    patch['COVER'] = patch['SCENE'].apply(cloud_cover)

    # Removes unneeded scene column
    del patch['SCENE']

    # Re-indexes the DataFrame to datetime
    patch.set_index(pd.to_datetime(patch['DATE'], format='%Y_%m_%d'), drop=True, inplace=True)

    # Sends DataFrame to scene_selection() and returns the 24 selected scenes
    return scene_selection(patch)


def stack_bands(patch_id, scene):
    """Stacks together all the bands of the SENTINEL-2 images in a given scene of a patch

    Args:
        patch_id (str): Unique patch ID
        scene (str): Date of scene in YY_MM_DD format to stack bands in

    Returns:
        Normalised and stacked red, green, blue arrays into RGB array
    """
    bands = []
    # Load R, G, B images from file and normalise
    for band in band_ids:
        bands.append(normalize(rdv.load_array('%s_%s_10m.tif' % (prefix_format(patch_id, scene), band), 1)))

    # Stack together RGB bands
    # Note that it has to be order BGR not RGB due to the order numpy stacks arrays
    return np.dstack(bands)


def make_time_series(patch_id):
    """Makes a time-series of each pixel of a patch across 24 scenes selected by REF's criteria using scene_selection().
     All the bands in the chosen scene are stacked using stack_bands()

    Args:
        patch_id (str): Unique patch ID

    Returns:
        (numpy.ndarray): Array of shape(rows, columns, 24, 12) holding all x for a patch
    """
    # List of scene dates found by REF's selection criteria
    scenes = find_best_of(patch_id)

    # Loads all pixels in a patch across the 24 scenes and 12 bands
    x = []
    for scene in scenes:
        x.append(stack_bands(patch_id, scene))

    # Returns a reordered numpy.ndarray holding all x for the given patch
    return np.moveaxis(np.array(x), 0, 2)


def make_loaders(patch_ids=None, split=(0.7, 0.15, 0.15), seed=42, shuffle=True, plot=False, balance=False, p_dist=False):
    """

    Args:
        patch_ids:
        split:
        seed:
        shuffle (bool):
        plot (bool):
        balance (bool):

    Returns:
        loaders (dict):
        n_batches (dict):
        class_dist (Counter):

    """
    # Fetches all patch IDs in the dataset
    if patch_ids is None:
        patch_ids = rdv.patch_grab()

    # Splits the dataset into train and val-test
    train_ids, val_test_ids = train_test_split(patch_ids, train_size=split[0], test_size=(split[1]+split[2]),
                                               shuffle=shuffle, random_state=seed)

    # Splits the val-test dataset into validation and test
    val_ids, test_ids = train_test_split(val_test_ids, train_size=(split[1]/(split[1]+split[2])),
                                         test_size=(split[2]/(split[1]+split[2])), shuffle=shuffle, random_state=seed)

    if p_dist:
        print('\nTrain: \n', find_subpopulations(train_ids, plot=plot))
        print('\nValidation: \n', find_subpopulations(val_ids, plot=plot))
        print('\nTest: \n', find_subpopulations(test_ids, plot=plot))

    datasets = {}

    if balance:
        train_stream = make_sorted_streams(train_ids)
        val_stream = make_sorted_streams(val_ids)

        # Define datasets for train, validation and test using BatchLoader
        datasets['train'] = BalancedBatchLoader(train_stream, batch_size=params['batch_size'], wheel_size=wheel_size)
        datasets['val'] = BalancedBatchLoader(val_stream, batch_size=params['batch_size'], wheel_size=wheel_size)

    if not balance:
        # Define datasets for train, validation and test using BatchLoader
        datasets['train'] = BatchLoader(train_ids, batch_size=params['batch_size'])
        datasets['val'] = BatchLoader(val_ids, batch_size=params['batch_size'])

    datasets['test'] = BatchLoader(test_ids, batch_size=params['batch_size'])

    # Create train, validation and test batch loaders and pack into dict
    loaders = {'train': DataLoader(datasets['train'], **params),
               'val': DataLoader(datasets['val'], **params),
               'test': DataLoader(datasets['test'], **params)}

    # Create a dict to hold number of batches for train, validation and test
    n_batches = {'train': num_batches(train_ids, balanced=balance),
                 'val': num_batches(val_ids, balanced=balance),
                 'test': num_batches(test_ids)}

    class_dist = find_subpopulations(patch_ids, plot=False)

    return loaders, n_batches, class_dist


def find_subpopulations(ids, plot=False):
    """Loads all LC labels for the given patches using lc_load() then finds the number of samples for each class

    Args:
        ids (list): List of patch IDs to analyse
        plot (bool): Plots distribution of subpopulations if True

    Returns:
        class_dist (Counter): Modal distribution of classes in the dataset provided
    """
    # Loads all LC label masks for the given patch IDs
    labels = []
    for patch_id in ids:
        labels.append(lc_load(patch_id))

    if plot:
        # Plots a pie chart of the distribution of the classes within the given list of patches
        plot_subpopulations(np.array(labels).flatten(), class_names=rdv.RE_classes, cmap=rdv.RE_cmap_dict)

    # Finds the distribution of the classes within the data
    return Counter(np.array(labels).flatten()).most_common()


def plot_subpopulations(class_labels, class_names=None, cmap=None):
    """Creates a pie chart of the distribution of the classes within the data

    Args:
        class_labels (np.array[int]): List of class labels
        class_names (dict): Dictionary mapping class labels to class names
        cmap (dict): Dictionary mapping class labels to class colours

    Returns:
        None
    """

    # Finds the distribution of the classes within the data
    class_dist = Counter(class_labels).most_common()

    # List to hold the name and percentage distribution of each class in the data as str
    classes = []

    # List to hold the total counts of each class
    counts = []

    # List to hold colours of classes in the correct order
    colours = []

    # Finds total number of samples to normalise data
    n_samples = 0
    for mode in class_dist:
        n_samples += mode[1]

    # For each class, find the percentage of data that is that class and the total counts for that class
    for label in class_dist:
        # Sets percentage label to <0.01% for classes matching that equality
        if (label[1] * 100.0 / n_samples) > 0.01:
            classes.append('{} \n{:.2f}%'.format(class_names[label[0]], (label[1] * 100.0 / n_samples)))
        else:
            classes.append('{} \n<0.01%'.format(class_names[label[0]]))
        counts.append(label[1])
        colours.append(cmap[label[0]])

    # Locks figure size
    plt.figure(figsize=(6, 5))

    # Plot a pie chart of the data distribution amongst the classes
    patches, text = plt.pie(counts, colors=colours, explode=[i * 0.05 for i in range(len(classes))])

    # Adds legend
    plt.legend(patches, classes, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    # Show plot for review
    plt.show()


def num_batches(ids, balanced=False):
    """Determines the number of batches needed to cover the dataset across ids

    Args:
        ids (list): List of patch IDs in the dataset to be loaded in by batches
        balanced (bool): If false, divide by batch_size. If true, do not divide by batch_size

    Returns:
        num_batches (int): Number of batches needed to cover the whole dataset
    """
    if balanced:
        return int((len(ids) * image_size[0] * image_size[1]))
    if not balanced:
        return int((len(ids) * image_size[0] * image_size[1]) / params['batch_size'])


def plot_history(metrics):
    """Plots model history based on metrics supplied
    
    Args:
        metrics (dict): Dictionary containing the names and results of the metrics by which model was assessed

    Returns:
        None
    """""
    # Initialise figure
    plt.figure()

    # Plots each metric in metrics, appending their artist handles
    handles = []
    for metric in metrics.values():
        handles.append(plt.plot(metric)[0])

    # Creates legend from plot artist handles and names of metrics
    plt.legend(handles=handles, labels=metrics.keys())

    # Adds axis labels
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')

    # Show figure
    plt.show()


def make_confusion_matrix(model, test_images, test_labels, batch_size, classes, filename, show=True, save=False):
    """Creates a heat-map of the confusion matrix of the given model

    Args:
        model (keras.Model):
        test_images ([[[float]]]): Images for testing model post-fitting
        test_labels ([[int]]): Accompanying labels for testing images
        batch_size (int): Number of images in each batch for network input
        classes ([str]): List of all class names
        filename (str): Name of file to save plot to
        show (bool): Whether to show plot
        save (bool): Whether to save plot to file

    Returns:
        None
    """

    # Uses model to make predictions on the images supplied
    pred_labels = model.predict_classes(test_images, batch_size=batch_size)

    # Creates the confusion matrix based on these predictions and the corresponding ground truth labels
    multi_class_cm = tf.math.confusion_matrix(labels=np.argmax(test_labels, axis=1), predictions=pred_labels).numpy()

    # Normalises confusion matrix
    multi_class_cm_norm = np.around(multi_class_cm.astype('float') / multi_class_cm.sum(axis=1)[:, np.newaxis],
                                    decimals=2)

    # Converts confusion matrix to Pandas.DataFrame
    multi_class_cm_df = pd.DataFrame(multi_class_cm_norm, index=classes, columns=classes)

    # Plots figure
    plt.figure()
    sns.heatmap(multi_class_cm_df, annot=True, square=True, cmap=plt.cm.get_cmap('Blues'), vmin=0.0, vmax=1.0)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')

    # Shows and/or saves plot
    if show:
        plt.show()
    if save:
        plt.savefig('%s_multi.png' % filename)
        plt.close()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    loaders, n_batches, class_dist = make_loaders(balance=True, p_dist=True)

    # Finds total number of samples to normalise data
    n_samples = 0
    for mode in class_dist:
        n_samples += mode[1]

    #find_subpopulations(rdv.patch_grab(), plot=True)

    #class_weights = torch.tensor([(1 - (mode[1]/n_samples)) for mode in class_dist], device=device)

    # Initialise model
    model = MLP(288, 8, [288, 144, 72, 36])

    # Transfer to GPU
    model.to(device)

    # Print model summary
    summary(model, (1, 288))

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()#weight=class_weights)

    # Define optimiser
    optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.90)
    #optimiser = torch.optim.Adam(model.parameters())#, lr=1e-3)#, amsgrad=True)
    #optimiser = torch.optim.Adadelta(model.parameters())

    train_loss_history = []
    val_loss_history = []

    train_acc_history = []
    val_acc_history = []

    # Iterates through epochs of training and validation
    for epoch in range(max_epochs):
        # TRAIN =================================================================

        train_loss = 0
        train_correct = 0

        # Batch trains model for this epoch
        with alive_bar(n_batches['train'], bar='blocks') as bar:
            model.train()
            for x_batch, y_batch in islice(loaders['train'], n_batches['train']):
                # Transfer to GPU
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                print(x_batch.size())
                print(y_batch.size())

                optimiser.zero_grad()

                # Forward pass
                y_pred = model(x_batch)

                # Compute Loss
                loss = criterion(y_pred, y_batch)

                # Backward pass
                loss.backward()
                optimiser.step()

                train_loss += loss.item()

                # calculate the accuracy
                predicted = torch.argmax(y_pred, 1)
                train_correct += (predicted == y_batch).sum().item()

                bar()

        val_loss = 0
        val_correct = 0
        # VALIDATION ===============================================================
        with alive_bar(n_batches['val'], bar='blocks') as bar, torch.no_grad():
            # Set the model to eval mode
            model.eval()
            for x_batch, y_batch in islice(loaders['val'], n_batches['val']):
                # Transfer to GPU
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Forward pass
                y_pred = model(x_batch)

                # Compute Loss
                loss = criterion(y_pred.squeeze(), y_batch)

                val_loss += loss.item()

                # calculate the accuracy
                predicted = torch.argmax(y_pred, 1)
                val_correct += (predicted == y_batch).sum().item()

                bar()

        # Output epoch results
        train_loss /= n_batches['train']
        val_loss /= n_batches['val']
        train_accuracy = train_correct / (n_batches['train'] * params['batch_size'])
        val_accuracy = val_correct / (n_batches['val'] * params['batch_size'])
        print(f'Epoch: {epoch + 1}/{max_epochs}| Training loss: {train_loss} | Validation Loss: {val_loss} | '
              f'Train Accuracy: {train_accuracy * 100}% | Validation Accuracy: {val_accuracy * 100}% \n')

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)

    # TEST ======================================================================
    model.eval()
    test_loss = 0
    test_correct = 0
    predictions = []
    with alive_bar(n_batches['test'], bar='blocks') as bar, torch.no_grad():
        for x_batch, y_batch in islice(loaders['test'], n_batches['test']):
            # Transfer to GPU
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            y_pred = model(x_batch)

            # Compute Loss
            loss = criterion(y_pred.squeeze(), y_batch)

            test_loss += loss.item()

            # calculate the accuracy
            predicted = torch.argmax(y_pred, 1)
            test_correct += (predicted == y_batch).sum().item()
            predictions.append(np.array(predicted.cpu()))

            bar()

    test_loss /= n_batches['test']
    test_accuracy = test_correct / (n_batches['test'] * params['batch_size'])
    print(f'Test loss: {test_loss}.. Test Accuracy: {test_accuracy * 100}%')

    metrics = {'Train Loss': train_loss_history,
               'Validation Loss': val_loss_history,
               'Train Accuracy': train_acc_history,
               'Validation Accuracy': val_acc_history}

    plot_history(metrics)
    plot_subpopulations(np.array(predictions).flatten(), rdv.RE_classes, rdv.RE_cmap_dict)


if __name__ == '__main__':
    main()
