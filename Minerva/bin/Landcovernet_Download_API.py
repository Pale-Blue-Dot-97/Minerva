"""API for downloading the LandCoverNet dataset.

Based on the old version of the API tutorial from LandCoverNet but is still functional.

Notes:
    Requires a Radiant Earth API key. With a Radiant MLHub account, your API key can be created within the
    `Settings & Keys' section of your profile. This key should then be copied into a file named `API Key.txt'
    within the package directory.

Source: https://github.com/radiantearth/mlhub-tutorials/blob/main/notebooks/radiant-mlhub-landcovernet.ipynb

Edited by: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.

Attributes:
    API_KEY (str): API key loaded in from API Key.txt. Should be the user key from dashboard.mlhub.earth.
    API_BASE (str): URL to the API.
    COLLECTION_ID (str): The ID for the LandCoverNetV1 dataset.
    s3:
    p: Thread pool to use for multi-threaded downloading of files.

TODO:
    * Fully document
"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Optional, Union, List
import argparse
import requests
import boto3  # Required to download assets hosted on S3
import os
from urllib.parse import urlparse
import arrow
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
# copy your API key from dashboard.mlhub.earth and paste it in the following
API_KEY = open('../../API Key', 'r').read()
API_BASE = 'https://api.radiant.earth/mlhub/v1'

COLLECTION_ID = 'ref_landcovernet_v1_labels'

s3 = boto3.client('s3')

p = ThreadPool(20)

data_dir = os.sep.join(('..', '..', 'data'))

# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def print_credentials():
    """Prints the credentials of the API from the returned response."""
    r = requests.get(f'{API_BASE}/collections/{COLLECTION_ID}', params={'key': API_KEY})
    print(f'\nDescription: {r.json()["description"]}')
    print(f'License: {r.json()["license"]}')
    print(f'DOI: {r.json()["sci:doi"]}')
    print(f'Citation: {r.json()["sci:citation"]}\n')


def get_classes() -> list:
    """Fetches and prints all possible classes in the dataset

    Returns:
        List of all possible label classes.
    """

    # Get response from API.
    r = requests.get(f'{API_BASE}/collections/{COLLECTION_ID}/items', params={'key': API_KEY})

    # Get the JSON for the classes in the dataset.
    label_classes = r.json()['features'][0]['properties']['label:classes']

    # Print the classes in the dataset in alphabetical order.
    for label_class in label_classes:
        print(f'Classes for {label_class["name"]}')
        for c in sorted(label_class['classes']):
            print(f'- {c}')

    return label_classes[0]['classes']


def download_s3(uri: str, path: str) -> None:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path[1:]
    s3.download_file(bucket, key, os.path.join(path, key.split('/')[-1]))


def download_http(uri: str, path: str) -> None:
    parsed = urlparse(uri)
    r = requests.get(uri)
    f = open(os.path.join(path, parsed.path.split('/')[-1]), 'wb')
    for chunk in r.iter_content(chunk_size=512 * 1024):
        if chunk:
            f.write(chunk)
    f.close()


def get_download_uri(uri: str):
    r = requests.get(uri, allow_redirects=False)
    return r.headers['Location']


def download(d: List[str]) -> None:
    href = d[0]
    path = d[1]
    download_uri = get_download_uri(href)
    parsed = urlparse(download_uri)

    if parsed.scheme in ['s3']:
        download_s3(download_uri, path)
    elif parsed.scheme in ['http', 'https']:
        download_http(download_uri, path)


def get_source_item_assets(path_args) -> list:
    path = path_args[0]
    href = path_args[1]
    asset_downloads = []

    try:
        r = requests.get(href, params={'key': API_KEY})
    except requests.exceptions.RequestException:
        print('ERROR: Could Not Load', href)
        return []

    dt = arrow.get(r.json()['properties']['datetime']).format('YYYY_MM_DD')
    asset_path = os.path.join(path, dt)

    if not os.path.exists(asset_path):
        os.makedirs(asset_path)

    for key, asset in r.json()['assets'].items():
        asset_downloads.append((asset['href'], asset_path))

    return asset_downloads


def download_source_and_labels(item) -> list:
    labels = item.get('assets').get('labels')
    links = item.get('links')

    # Make the directory to download the files to
    path = f'landcovernet/{item["id"]}/'
    if not os.path.exists(path):
        os.makedirs(path)

    source_items = []

    # Download the source imagery
    for link in links:
        if link['rel'] != 'source':
            continue
        source_items.append((path, link['href']))

    results = p.map(get_source_item_assets, source_items)
    results.append([(labels['href'], path)])

    return results


def get_items(uri: str, classes: Optional[Union[List[str], str]] = None, max_items_downloaded: Optional[int] = None,
              items_downloaded: int = 0, downloads: Optional[list] = None) -> list:
    """Loops over the dataset selecting items to download matching criteria given until limit is reached.

    Args:
        uri (str): URI to the dataset collection.
        classes (list[str], str): Optional; Classes to prioritise in the selection of items to download.
        max_items_downloaded (int): Optional; Maximum number of items to download.
        items_downloaded (int): Optional; Number of items downloaded thus far.
        downloads (list): Optional; Cumulative items selected for download from previous loop.

    Returns:
        downloads (list): Items selected for download from dataset.
    """

    if downloads is None:
        downloads = []

    print('Loading', uri, '...')

    r = requests.get(uri, params={'key': API_KEY})
    collection = r.json()

    for feature in collection.get('features', []):
        # Check if the item has one of the label classes we're interested in.
        matches_class = True
        if classes is not None:
            matches_class = False
            for label_class in feature['properties'].get('labels', []):
                if label_class in classes:
                    matches_class = True
                    break

        # If the item does not match all of the criteria we specify, skip it.
        if not matches_class:
            continue

        # Skips if patch already exists in current working directory.
        if os.path.isdir(os.path.join('landcovernet', feature['id'])):
            print('Skip ', feature['id'])
            continue

        print('Getting Source Imagery Assets for', feature['id'])
        # Download the label and source imagery for the item.
        downloads.extend(download_source_and_labels(feature))

        # Stop downloaded items if we reached the maximum we specify.
        items_downloaded += 1
        if max_items_downloaded is not None and items_downloaded >= max_items_downloaded:
            return downloads

    # Get the next page if results, if available.
    for link in collection.get('links', []):
        if link['rel'] == 'next' and link['href'] is not None:
            get_items(link['href'], classes=classes, max_items_downloaded=max_items_downloaded,
                      items_downloaded=items_downloaded, downloads=downloads)

    return downloads


def download_request(classes: Optional[Union[List[str], str]] = None,
                     max_items_downloaded: Optional[int] = None) -> None:
    """Given a request, downloads the items matching the query from the Radiant MLHub LandCoverNet API.

    Args:
        classes (list[str], str): Optional; Classes to prioritise in the selection of items from the dataset.
        max_items_downloaded (int): Optional; Maximum number of items to download from request.

    Returns:
        None
    """

    # Creates a list of items to download from the collection based on the request query submitted
    to_download = get_items(f'{API_BASE}/collections/{COLLECTION_ID}/items',
                            classes=classes,
                            max_items_downloaded=max_items_downloaded,
                            downloads=[])

    # Downloads the requested list of items from the collection
    print('Downloading Assets')
    for d in tqdm(to_download):
        p.map(download, d)


def format_download_request(classes: Optional[Union[List[str], str]] = None,
                            items: Optional[Union[List[int], int]] = None,
                            download_dir: str = data_dir) -> None:
    """Formats the correct arguments for download_request based on the types of input given for classes and items.

    Args:
        classes (list[str], str): Optional; The classes from which to prioritise in the selection of items
            to download.
        items (list[int], int): Optional; The number of items to download, either per class or as a whole.
        download_dir (str): Optional; The path to the directory to download data to.

    Returns:
        None
    """
    def str_reformat(string: Optional[str] = None) -> Optional[str]:
        """Removes any hyphens or underscores from the provided string. Takes None argument to allow pass-through.

        Args:
            string (str): Optional; String for hyphens and underscores to be removed from.

        Returns:
            string (str): String with hyphens and underscores removed or None if None parsed.
        """

        if string is None:
            return
        else:
            string.replace('_', '')
            string.replace('-', '')
            return string

    # Change directory to specified location for downloads.
    try:
        os.chdir(download_dir)
    except FileNotFoundError:
        os.mkdir(download_dir)
        os.chdir(download_dir)

    # If certain classes from which to download from was specified:
    if classes is not None:
        # If multiple classes were specified:
        if type(items) is list:
            for i in range(len(classes)):
                print('{} - {}'.format(classes[i], items[i]))
                download_request(classes=str_reformat(classes[i]), max_items_downloaded=items[i])
        else:
            download_request(classes=str_reformat(classes), max_items_downloaded=items)

    # If classes were not specified:
    else:
        # Ensure only a int is sent to specify the maximum number of items of the dataset to download.
        if type(items) is list:
            download_request(max_items_downloaded=items[0])

        else:
            download_request(max_items_downloaded=items)


def main(classes: Optional[Union[list, str]] = None, items: Optional[Union[list, int]] = None) -> None:

    # Fetches and prints API credentials
    print_credentials()

    # Print all possible classes in dataset.
    get_classes()

    format_download_request(classes=classes, items=items)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--classes",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=None,  # default if nothing is provided
    )
    CLI.add_argument(
        "--items",
        nargs="*",
        type=int,  # any type/callable can be used here
        default=None,
    )

    args = CLI.parse_args()

    print('Classes: {}'.format(args.classes))
    print('Items: {}'.format(args.items))

    command = input('Proceed? (Y/N): \n')

    if command in ['Y', 'y' 'yes', 'Yes', 'YES']:
        main(classes=args.classes, items=args.items)

    else:
        print('ABORT')
