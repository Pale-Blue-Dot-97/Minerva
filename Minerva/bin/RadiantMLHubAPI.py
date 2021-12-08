"""API for downloading the LandCoverNet dataset.

Based on the API notebook tutorial of the Radiant MLHub API for downloading LandCoverNet assets.

Notes:
    Requires a Radiant Earth API key. With a Radiant MLHub account, your API key can be created within the
    `Settings & Keys' section of your profile. This key should then set to the environment by
    calling `mlhub configure'. See https://mlhub.earth/docs for more details.

Source: https://github.com/radiantearth/mlhub-tutorials/blob/main/notebooks/radiant-mlhub-landcovernet.ipynb

Edited by: Harry James Baker

Email: hjb1d20@soton.ac.uk or hjbaker97@gmail.com

Institution: University of Southampton

Created under a project funded by the Ordnance Survey Ltd.

Attributes:
    collection_id (str): The ID for the LandCoverNetV1 dataset.
    config_path (str): Path to master config YAML file.
    config (dict): Master config defining how the experiment should be conducted.
    data_dir (list): Path to directory holding dataset.

TODO:
    * Add missing docstrings
    * Add type-hinting
"""

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils
from typing import Optional, List
import os
import argparse
import urllib.parse
import re
from pathlib import Path
import itertools as it
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from radiant_mlhub import client, get_session

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
collection_id = 'ref_landcovernet_v1_labels'

items_pattern = re.compile(r'^/mlhub/v1/collections/(\w+)/items/(\w+)$')

config_path = '../../config/config.yml'

config, _ = utils.load_configs(config_path)

data_dir = os.sep.join(config['dir']['data'][:-1])


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def print_credentials():
    collection = client.get_collection(collection_id)
    print(f'Description: {collection["description"]}')
    print(f'License: {collection["license"]}')
    print(f'DOI: {collection["sci:doi"]}')
    print(f'Citation: {collection["sci:citation"]}')


def get_classes():
    items = client.list_collection_items(collection_id, limit=1)

    first_item = next(items)

    label_classes = first_item['properties']['label:classes']
    for label_class in label_classes:
        print(f'Classes for {label_class["name"]}')
        for c in sorted(label_class['classes']):
            print(f'- {c}')


def filter_item(item, classes=None, cloud_and_shadow=None, seasonal_snow=None):
    """Function to be used as an argument to Python's built-in filter function that filters out any items that
    do not match the given classes, cloud_and_shadow, and/or seasonal_snow values.

    If any of these filter arguments are set to None, they will be ignored. For instance, using
    filter_item(item, cloud_and_shadow=True) will only return items
    where item['properties']['cloud_and_shadow'] == 'true',
    and will not filter based on classes/labels, or seasonal_snow.
    """
    # Match classes, if provided

    item_labels = item['properties'].get('labels', [])
    if classes is not None and not any(label in classes for label in item_labels):
        return False

    # Match cloud_and_shadow, if provided
    item_cloud_and_shadow = item['properties'].get('cloud_and_shadow', 'false') == 'true'
    if cloud_and_shadow is not None and item_cloud_and_shadow != cloud_and_shadow:
        return False

    # Match seasonal_snow, if provided
    item_seasonal_snow = item['properties'].get('seasonal_snow', 'false') == 'true'
    if seasonal_snow is not None and item_seasonal_snow != seasonal_snow:
        return False

    return True


def get_items(cid: str, classes=None, cloud_and_shadow=None, seasonal_snow=None, max_items=1):
    """Generator that yields up to max_items items that match the given classes, cloud_and_shadow, and seasonal_snow
    values. Setting one of these filter arguments to None will cause that filter to be ignored (e.g. classes=None
    means that items will not be filtered by class/label).
    """
    filter_fn = partial(
        filter_item,
        classes=classes,
        cloud_and_shadow=cloud_and_shadow,
        seasonal_snow=seasonal_snow
    )
    filtered = filter(
        filter_fn,

        client.list_collection_items(cid, limit=max_items)
    )
    yield from it.islice(filtered, max_items)


def download(item, asset_key, output_dir: str = data_dir) -> None:
    """Downloads the given item asset by looking up that asset and then following the "href" URL."""

    # Try to get the given asset and return None if it does not exist
    asset = item.get('assets', {}).get(asset_key)
    if asset is None:
        print(f'Asset "{asset_key}" does not exist in this item')
        return None

    # Try to get the download URL from the asset and return None if it does not exist
    download_url = asset.get('href')
    if download_url is None:
        print(f'Asset {asset_key} does not have an "href" property, cannot download.')
        return None

    session = get_session()
    r = session.get(download_url, allow_redirects=True, stream=True)

    filename = urllib.parse.urlsplit(r.url).path.split('/')[-1]
    output_path = Path(output_dir) / filename

    with output_path.open('wb') as dst:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            if chunk:
                dst.write(chunk)


def download_labels_and_source(item, assets=None, output_dir: str = data_dir) -> None:
    """Downloads all label and source imagery assets associated with a label item that match the given asset types.
    """

    # Follow all source links and add all assets from those
    def _get_download_args(link):
        # Get the item ID (last part of the link path)
        source_item_path = urllib.parse.urlsplit(link['href']).path
        source_item_collection, source_item_id = items_pattern.fullmatch(source_item_path).groups()
        source_item = client.get_collection_item(source_item_collection, source_item_id)

        source_download_dir = download_dir / 'source'
        source_download_dir.mkdir(exist_ok=True)

        matching_source_assets = [
            asset
            for asset in source_item.get('assets', {})
            if assets is None or asset in assets
        ]
        return [
            (source_item, asset, source_download_dir)
            for asset in matching_source_assets
        ]

    download_args = []

    download_dir = Path(output_dir) / item['id']
    download_dir.mkdir(parents=True, exist_ok=True)

    labels_download_dir = download_dir / 'labels'
    labels_download_dir.mkdir(exist_ok=True)

    # Download the labels assets
    matching_assets = [
        asset
        for asset in item.get('assets', {})
        if assets is None or asset in assets
    ]

    for asset in matching_assets:
        download_args.append((item, asset, labels_download_dir))

    source_links = [link for link in item['links'] if link['rel'] == 'source']

    with ThreadPoolExecutor(max_workers=16) as executor:
        for argument_batch in executor.map(_get_download_args, source_links):
            download_args += argument_batch

    print(f'Downloading {len(download_args)} assets...')
    with ThreadPoolExecutor(max_workers=16) as executor:
        with tqdm(total=len(download_args)) as pbar:
            for _ in executor.map(lambda triplet: download(*triplet), download_args):
                pbar.update(1)


def main(classes: Optional[List[str]] = None, max_items: Optional[int] = None,
         assets: Optional[List[str]] = None, download_dir: str = data_dir,
         archive_download: bool = False) -> None:
    print_credentials()

    get_classes()

    if archive_download:
        client.download_archive(collection_id, output_dir=download_dir)

    else:
        items = get_items(
            collection_id,
            classes=classes,
            max_items=max_items,
        )
        for item in items:
            download_labels_and_source(item, assets=assets, output_dir=download_dir)


if __name__ == '__main__':
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--classes",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=None,  # default if nothing is provided
    )
    CLI.add_argument(
        "--assets",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=str,
        default=None,  # default if nothing is provided
    )
    CLI.add_argument(
        "--items",
        type=int,  # any type/callable can be used here
        default=None,
    )
    CLI.add_argument(
        "--d",
        type=str,  # any type/callable can be used here
        default=data_dir,
    )
    CLI.add_argument(
        "--archive_download",
        action='store_true',
        default=False
    )

    args = CLI.parse_args()

    print(f'Classes: {args.classes}')
    print(f'Assets: {args.assets}')
    print(f'Max items: {args.items}')
    print(f'Output path: {args.d}')

    command = input('Proceed? (Y/N): \n')

    if command in ['Y', 'y' 'yes', 'Yes', 'YES']:
        main(classes=args.classes, assets=args.assets, max_items=args.items, download_dir=args.d,
             archive_download=args.archive_download)

    else:
        print('ABORT')
