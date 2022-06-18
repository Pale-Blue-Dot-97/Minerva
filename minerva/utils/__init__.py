import argparse
import ntpath
import os
from typing import Any, Dict, Tuple, Optional

import yaml

# Default values for the path to the config directory and config name.
config_dir_path: str = "../../inbuilt_cfgs/"
default_config_name: str = "example_config.yml"

# Objects to hold the config name and path.
config_name: Optional[str] = None
config_path: Optional[str] = None


def chdir_to_default(conf_name: Optional[str] = None) -> str:
    this_abs_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.sep.join((this_abs_path, config_dir_path)))

    if conf_name is None:
        return default_config_name
    elif not os.path.exists(conf_name):
        return default_config_name
    else:
        return conf_name


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
    config_paths = master_config["dir"]["configs"]

    # Loads and returns the other configs along with master config.
    return master_config, aux_config_load(config_paths)


master_parser = argparse.ArgumentParser(add_help=False)
master_parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="Path to the config file defining experiment",
)
master_parser.add_argument(
    "--default-config-dir",
    dest="default_config_dir",
    action="store_true",
    help="Set config path to default",
)
args, _ = master_parser.parse_known_args()

# Set the config path from the option found from args.
if args.config is not None:
    head, tail = ntpath.split(args.config)
    if head != "" or head is not None:
        config_path = head
    elif head == "" or head is None:
        config_path = ""
    config_name = tail

# Overwrites the config path if option found in args regardless of -c args.
if args.default_config_dir:
    if config_path is not None:
        print(
            "Warning: Config path specified with `--default_config_dir` option."
            + "\nDefault config directory path will be used."
        )
    config_path = None

# Store the current working directory (i.e where script is being run from).
cwd = os.getcwd()

# If no config_path, change directory to the default config directory.
if config_path is None:
    config_name = chdir_to_default(config_name)

# Check the config specified exists at the path given. If not, assume its in the default directory.
else:
    if config_name is None:
        config_name = chdir_to_default(config_name)
    elif not os.path.exists(os.sep.join((config_path, config_name))):
        config_name = chdir_to_default(config_name)
    else:
        pass

path = config_name
if config_path is not None:
    path = os.sep.join((config_path, config_name))

# Loads the configs from file using paths found in sys.args.
config, aux_configs = load_configs(path)

# Change the working directory back to script location.
os.chdir(cwd)
