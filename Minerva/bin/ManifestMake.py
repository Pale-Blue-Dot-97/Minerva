"""Script to create manifests of data for use in Minerva pre-processing to reduce computation time.

    Copyright (C) 2022 Harry James Baker

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

Created under a project funded by the Ordnance Survey Ltd.

Attributes:
    config_path (str): Path to master config YAML file.
    config (dict): Master config defining how the experiment should be conducted.

TODO:
    * Re-engineer for use with torchvision style datasets
    * Consider use of parquet format rather than csv
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import utils
from Minerva.loaders import construct_dataloader, load_all_samples
import pandas as pd
import os

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/mf_config.yml'

config, _ = utils.load_configs(config_path)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def make_manifest() -> pd.DataFrame:
    """Constructs a manifest of the dataset detailing each sample therein.

    The dataset to construct a manifest of is defined by the 'data_config' value in the config.

    Returns:
        df (pd.DataFrame): The completed manifest as a DataFrame.
    """
    dataloader_params = config['dataloader_params']
    dataset_params = config['dataset_params']
    sampler_params = config['sampler_params']
    collator_params = config['collator']

    print('CONSTRUCTING DATASET')
    loader = construct_dataloader(config['dir']['data'], dataset_params, sampler_params, 
                                  dataloader_params, collator_params=collator_params)

    print('FETCHING SAMPLES')
    df = pd.DataFrame()
    df['MODES'] = load_all_samples(loader)

    print('CALCULATING CLASS FRACTIONS')
    # Calculates the fractional size of each class in each patch.
    df = pd.DataFrame([row for row in df.apply(utils.class_frac, axis=1)])
    df.fillna(0, inplace=True)
    
    # Delete redundant MODES column.
    del df['MODES']

    return df


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    manifest = make_manifest()

    print(manifest)

    output_dir = os.sep.join(config['dir']['output'])

    fn = os.sep.join([output_dir, f'{utils.get_dataset_name()}_Manifest.csv'])

    print(f'MANIFEST TO FILE -----> {fn}')
    manifest.to_csv(fn)


if __name__ == '__main__':
    main()
