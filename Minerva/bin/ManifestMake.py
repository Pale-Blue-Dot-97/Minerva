"""Script to create manifests of data for use in Minerva pre-processing to reduce computation time.

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
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils
import pandas as pd
import yaml
import os

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

# Path to directory holding dataset
data_dir = os.sep.join(config['dir']['data'])


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def make_manifest():
    print('GETTING PATCH IDs')
    patch_ids = utils.patch_grab()
    scenes = []
    clouds = []

    print('GETTING CLDs and SCENES')
    for patch_id in patch_ids:
        clds, dates = utils.cloud_grab(patch_id)
        scenes += [(patch_id, scene) for scene in dates]
        clouds += clds

    print('CONSTRUCTING DATAFRAME')
    df = pd.DataFrame()
    df['SCENE'] = scenes
    df['PATCH'] = df['SCENE'].apply(utils.extract_patch_ids)
    df['DATE'] = df['SCENE'].apply(utils.extract_dates)
    scene_tags = utils.scene_tag(scenes)
    df['SCENE'] = scene_tags

    print('CALCULATING CLASS MODES')
    # Calculates the class modes of each patch.
    df['MODES'] = df['PATCH'].apply(utils.find_patch_modes)

    print('CALCULATING CLASS FRACTIONS')
    # Calculates the fractional size of each class in each patch.
    df = pd.DataFrame([row for row in df.apply(utils.class_frac, axis=1)])
    df.fillna(0, inplace=True)

    print('CALCULATING CLOUD COVER')
    df['CLD'] = clouds
    # Calculates the cloud cover percentage for every scene and adds to DataFrame.
    df['COVER'] = df['CLD'].apply(utils.cloud_cover)

    # Removes unneeded CLD and MODES columns.
    del df['CLD']
    del df['MODES']

    print('FINDING CENTRE LABELS')
    df['CPL'] = utils.dataset_lc_load(df['PATCH'], utils.find_centre_label)

    return df


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    manifest = make_manifest()

    print(manifest)

    fn = os.sep.join([data_dir, '{}_Manifest.csv'.format(utils.get_dataset_name())])

    print('MANIFEST TO FILE -----> {}'.format(fn))
    manifest.to_csv(fn)


if __name__ == '__main__':
    main()
