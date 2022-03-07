import yaml
import os, sys, getopt, shutil, ntpath
from typing import Tuple, Dict, Any, List, Optional

# Default values for the path to the config directory and config name.
config_dir_path = '../../inbuilt_cfgs/'
default_config_name = 'config.yml'

# Objects to hold the config name and path.
config_name = None
config_path = None


def get_sys_args(flags: str, long_options: Optional[List[str]] = None) -> Optional[Tuple[List[Tuple[str, str]], 
                                                                                   List[Tuple[str, ...]]]]:
    """Get sys.argv and extract options and arguments."""
    try:
        opts, args = getopt.getopt(sys.argv[1:], flags, long_options)
        return opts, args
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)


def chdir_to_default(config_name: str = config_name) -> None:
    this_abs_path = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.sep.join((this_abs_path, config_dir_path)))

    try:
        if not os.path.exists(config_name):
            config_name = default_config_name
    except TypeError:
        config_name = default_config_name


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


# Gets the sys.args formatted using flags and options.
opts, args = get_sys_args("c:", ["default_config_dir"])

# Set the config path from the option found from sys.argv.
for o, a in opts:
    if o == "-c":
        head, tail = ntpath.split(a)
        if head !=  "" or head is not None:
            config_path = head
        elif head == "" or head is None:
            config_path = ""
        config_name = tail

# Overwrites the config path if option found in sys.args regardless of -c args.
for o, a in opts:
    if o == "--default_config_dir":
        if config_path is not None:
            print("Warning: Config path specified with `--default_config_dir` option." +
                  "\nDefault config directory path will be used.")
        config_path = None

# Store the current working directory (i.e where script is being run from).
cwd = os.getcwd()

# If no config_path, change directory to the default config directory.
if config_path is None:
    chdir_to_default(config_name)

# Check the config specified exists at the path given. If not, assume its in the default directory.
else:
    try:
        if not os.path.exists(os.sep.join((config_path, config_name))):
            chdir_to_default(config_name)
    except TypeError:
        chdir_to_default(config_name)

# Ensures there is a config.yml to act as default for testing on GitHub etc. 
if not os.path.exists(default_config_name):
    shutil.copy("example_config.yml", default_config_name)

path = config_name
if config_path is not None:
    path = os.sep.join((config_path, config_name))

# Loads the configs from file using paths found in sys.args.
config, aux_configs = load_configs(path)

# Change the working directory back to script location.
os.chdir(cwd)
