"""Script to cut GEOTiffs into patches from larger tiles.

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

Attributes:
    tile_dir (str): Path to directory holding the tiles to be cut from.
    tile_suffix (str): Suffix in tile filenames identifying them as belonging to the dataset desired.

TODO:

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from typing import Union, Iterable
import os
from Minerva.utils import utils
from alive_progress import alive_bar

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
tile_dir = os.sep.join(['G:', 'GitHub', 'Minerva', 'Minerva', 'data', 'ESRI2020'])
tile_suffix = '20200101-20210101'


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def cut_patch_labels(tile_ids: Union[list, tuple, Iterable]) -> None:
    """Finds all patches within each tile and cuts the matching patches from the tiles.

    Args:
        tile_ids (list[str]): List of unique tile IDs to cut patches from.

    Returns:
        None
    """
    # Gets the IDs for all patches in the dataset.
    patch_ids = utils.patch_grab()

    # Progress bar.
    with alive_bar(len(patch_ids), bar='blocks') as bar:
        # Iterates through all tile IDs and finds all patches belonging to each tile.
        for tile_id in tile_ids:
            tile_patches = utils.get_patches_in_tile(tile_id, patch_ids)

            # Iterates through all patches within the tile and cuts the matching patches
            # from the tile and saves to file.
            for patch_id in tile_patches:
                utils.cut_to_extents(patch_id, tile_id, tile_dir)

                # Updates progress bar.
                bar()


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    # Finds and returns all unique tile IDs within the specified directory with the matching suffixes.
    tile_ids = utils.tile_grab(tile_dir, tile_suffix)

    # Cuts all the patches from each tile and saves to file.
    print('CUTTING PATCH LABELS FROM TILES')
    cut_patch_labels(tile_ids)


if __name__ == '__main__':
    main()
