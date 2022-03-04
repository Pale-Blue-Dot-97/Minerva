import yaml
import os, sys, getopt, shutil
from typing import Tuple, Dict, Any

config_dir_path = '../../config/'
default_config_name = 'config.yml'


def load_configs(master_config_path: str) -> Tuple[Dict[str, Any], ...]:
    """Loads the master config from YAML. Finds other config paths within and loads them.

    Args:
        master_config_path (str): Path to the master config YAML file.

    Returns:
        Master config and any other configs found from paths in the master config.
    """

    def yaml_load(path: str) -> Any:
        """Loads YAML file from path as dict.
        Args:
            path(str): Path to YAML file.

        Returns:
            yml_file (dict): YAML file loaded as dict.
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def aux_config_load(paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Loads and returns config files from YAML as dicts.

        Args:
            paths (dict): Dictionary mapping config names to paths to their YAML files.

        Returns:
            Config dictionaries loaded from YAML from paths.
        """
        configs = {}
        for _config_name in paths.keys():
            # Loads config from YAML as dict.
            configs[_config_name] = yaml_load(paths[_config_name])
        return configs

    # First loads the master config.
    master_config = yaml_load(master_config_path)

    # Gets the paths for the other configs from master config.
    config_paths = master_config['dir']['configs']

    # Loads and returns the other configs along with master config.
    return master_config, aux_config_load(config_paths)

# Get sys.argv and extract options and arguments.
try:
    opts, args = getopt.getopt(sys.argv[1:], "c:")
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

# Set the config path from the option found from sys.argv.
config_name = None
for o, a in opts:
    if o == "-c":
        config_name = a

cwd = os.getcwd()
path = os.path.abspath(os.path.dirname(__file__))

os.chdir(os.sep.join((path, config_dir_path)))

try:
    if not os.path.exists(config_name):
        config_name = default_config_name
except TypeError:
    config_name = default_config_name

if not os.path.exists(default_config_name):
    shutil.copy("example_config.yml", default_config_name)

config, aux_configs = load_configs(config_name)

os.chdir(cwd)
