"""Landcovernet_Download_API.

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


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def get_classes():
    # Fetches and prints all possible classes in the dataset
    r = requests.get(f'{API_BASE}/collections/{COLLECTION_ID}/items', params={'key': API_KEY})
    label_classes = r.json()['features'][0]['properties']['label:classes']
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


def download(d: List[str, str]) -> None:
    href = d[0]
    path = d[1]
    download_uri = get_download_uri(href)
    parsed = urlparse(download_uri)

    if parsed.scheme in ['s3']:
        download_s3(download_uri, path)
    elif parsed.scheme in ['http', 'https']:
        download_http(download_uri, path)


def get_source_item_assets(args) -> list:
    path = args[0]
    href = args[1]
    asset_downloads = []
    try:
        r = requests.get(href, params={'key': API_KEY})
    except:
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


def get_items(uri: str, classes: Optional[list, str] = None, max_items_downloaded: Optional[int] = None,
              items_downloaded: int = 0, downloads: Optional[list] = None) -> list:
    if downloads is None:
        downloads = []
    print('Loading', uri, '...')
    r = requests.get(uri, params={'key': API_KEY})
    collection = r.json()
    for feature in collection.get('features', []):
        # Check if the item has one of the label classes we're interested in
        matches_class = True
        if classes is not None:
            matches_class = False
            for label_class in feature['properties'].get('labels', []):
                if label_class in classes:
                    matches_class = True
                    break

        # If the item does not match all of the criteria we specify, skip it
        if not matches_class:
            continue

        # Skips if patch already exists in current working directory.
        if os.path.isdir(os.path.join('landcovernet', feature['id'])):
            print('Skip ', feature['id'])
            continue

        print('Getting Source Imagery Assets for', feature['id'])
        # Download the label and source imagery for the item
        downloads.extend(download_source_and_labels(feature))

        # Stop downloaded items if we reached the maximum we specify
        items_downloaded += 1
        if max_items_downloaded is not None and items_downloaded >= max_items_downloaded:
            return downloads

    # Get the next page if results, if available
    for link in collection.get('links', []):
        if link['rel'] == 'next' and link['href'] is not None:
            get_items(link['href'], classes=classes, max_items_downloaded=max_items_downloaded,
                      items_downloaded=items_downloaded, downloads=downloads)

    return downloads


def download_request(classes: Optional[list, str] = None, max_items_downloaded: Optional[int] = None) -> None:
    """Given a request, downloads the items matching the query from the Radiant MLHub LandCoverNet API.

    Args:
        classes (list):
        max_items_downloaded (int): Maximum number of items downloaded from request

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


def main(classes: Optional[list, str] = None, items: Optional[Union[list, int]] = None) -> None:
    # Fetches and prints API credentials
    r = requests.get(f'{API_BASE}/collections/{COLLECTION_ID}', params={'key': API_KEY})
    print(f'Description: {r.json()["description"]}')
    print(f'License: {r.json()["license"]}')
    print(f'DOI: {r.json()["sci:doi"]}')
    print(f'Citation: {r.json()["sci:citation"]}')

    def str_reformat(string: str) -> str:
        string.replace('_', '')
        string.replace('-', '')

        return string

    os.chdir('../../data')

    if classes is not None:
        if type(items) is list:
            for i in range(len(classes)):
                print('{} - {}'.format(classes[i], items[i]))
                download_request(classes=str_reformat(classes[i]), max_items_downloaded=items[i])

        elif type(items) is int:
            download_request(classes=str_reformat(classes), max_items_downloaded=items)

        else:
            download_request(classes=str_reformat(classes), max_items_downloaded=10)

    else:
        if type(items) is list:
            download_request(max_items_downloaded=items[0])

        elif items is int:
            download_request(max_items_downloaded=items)

        else:
            download_request()


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
