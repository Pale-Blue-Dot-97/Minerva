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
import pandas as pd
import os
from torch.utils.data import DataLoader

# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/mf_config.yml'

config, _ = utils.load_configs(config_path)


# =====================================================================================================================
#                                                     METHODS
# =====================================================================================================================
def load_all_samples(dataloader):
    samples = {}
    for i, sample in enumerate(dataloader):
        samples[i] = sample['mask']

    return samples


def make_manifest() -> pd.DataFrame:
    """Constructs a manifest of the dataset detailing each sample therein.

    The dataset to construct a manifest of is defined by the 'data_config' value in the config.

    Returns:
        df (pd.DataFrame): The completed manifest as a DataFrame.
    """
    dataloader_params = config['dataloader_params']
    dataset_params = config['dataset_params']
    sampler_params = config['sampler_params']

    print('CONSTRUCTING DATASET')
    _image_dataset = utils.func_by_str(module=dataset_params['imagery']['module'],
                                       func=dataset_params['imagery']['name'])

    _label_dataset = utils.func_by_str(module=dataset_params['labels']['module'],
                                       func=dataset_params['labels']['name'])

    imagery_root = os.sep.join((*config['dir']['data'], dataset_params['imagery']['root']))
    labels_root = os.sep.join((*config['dir']['data'], dataset_params['labels']['root']))

    image_dataset = _image_dataset(root=imagery_root, **dataset_params['imagery']['params'])
    label_dataset = _label_dataset(root=labels_root, **dataset_params['labels']['params'])

    dataset = image_dataset & label_dataset

    # --+ MAKE SAMPLERS +=========================================================================================+
    sampler = utils.func_by_str(module=sampler_params['module'], func=sampler_params['name'])
    sampler = sampler(dataset=image_dataset, **sampler_params['params'])

    # --+ MAKE DATALOADERS +======================================================================================+
    collator = utils.func_by_str(config['collator']['module'], config['collator']['name'])
    loader = DataLoader(dataset, sampler=sampler, collate_fn=collator, **dataloader_params)

    print('FETCHING SAMPLES')
    df = pd.DataFrame()
    df['PATCHES'] = load_all_samples(loader)

    print('CALCULATING CLASS MODES')
    # Calculates the class modes of each patch.
    df['MODES'] = df['PATCH'].apply(utils.find_patch_modes)

    print('CALCULATING CLASS FRACTIONS')
    # Calculates the fractional size of each class in each patch.
    df = pd.DataFrame([row for row in df.apply(utils.class_frac, axis=1)])
    df.fillna(0, inplace=True)

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
