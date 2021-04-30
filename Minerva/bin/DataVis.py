"""DataVis

Example script using visutils to create GIFs of the SCL for all patches in dataset

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

"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from Minerva.utils import visutils
import yaml
from matplotlib.colors import ListedColormap
from osgeo import osr


# =====================================================================================================================
#                                                     GLOBALS
# =====================================================================================================================
config_path = '../../config/config.yml'
lcn_config_path = '../../config/landcovernet.yml'
s2_config_path = '../../config/S2.yml'

with open(config_path) as file:
    config = yaml.safe_load(file)

with open(lcn_config_path) as file:
    lcn_config = yaml.safe_load(file)

with open(s2_config_path) as file:
    s2_config = yaml.safe_load(file)

# Create a new projection system in lat-lon
WGS84_4326 = osr.SpatialReference()
WGS84_4326.ImportFromEPSG(lcn_config['co_sys']['id'])

# ======= RADIANT MLHUB PRESETS =======================================================================================
# Radiant Earth land cover classes reformatted to split across two lines for neater plots
RE_classes = lcn_config['classes']

# Custom cmap matching the Radiant Earth Foundation specifications
RE_cmap = ListedColormap(lcn_config['colours'].values(), N=len(RE_classes))

# Pre-set RE figure height and width (in inches)
RE_figdim = (8.02, 10.32)

# ======= SENTINEL-2 L2A SCL PRESETS ==================================================================================
# SCL land cover classes reformatted to split across two lines for neater plots
S2_SCL_classes = s2_config['classes']

# Custom colour mapping from class definitions in the SENTINEL-2 L2A MSI
S2_SCL_cmap_dict = s2_config['colours']

# Custom cmap matching the SENTINEL-2 L2A SCL classes
S2_SCL_cmap = ListedColormap(S2_SCL_cmap_dict.values(), N=len(S2_SCL_classes))

# Preset SCL figure height and width (in inches)
S2_SCL_figdim = (8, 10.44)


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
if __name__ == '__main__':
    # Additional options for names dictionary:
    #           'patch_ID': '31PGS_15',     Five char alpha-numeric SENTINEL tile ID and
    #                                       2 digit int REF MLHub chip (patch) ID ranging from 0-29
    #           'date': '16.04.2018',       Date of scene in DD.MM.YYYY format
    my_names = {'band_ID': 'SCL',           # 3 char alpha-numeric Band ID
                'R_band': 'B02',            # Red, Green, Blue band IDs for RGB images
                'G_band': 'B03',
                'B_band': 'B04'}

    visutils.make_all_the_gifs(my_names, frame_length=0.5, data_band=1, classes=S2_SCL_classes, cmap_style=S2_SCL_cmap,
                               new_cs=WGS84_4326, alpha=0.3, figdim=S2_SCL_figdim)
