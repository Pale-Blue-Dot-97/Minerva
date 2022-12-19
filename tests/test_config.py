import os
from pathlib import Path
from minerva.utils.config_load import (
    load_configs,
    universal_path,
    check_paths,
    chdir_to_default,
    DEFAULT_CONFIG_NAME,
)


def test_universal_path():
    path1 = "one/two/three/file.txt"
    path2 = ["one", "two", "three", "file.txt"]

    correct = Path(path1)

    assert universal_path(path1) == correct
    assert universal_path(path2) == correct


def test_config_path(config_root, config_here):
    assert "tmp/config" in str(config_root)

    # Still works because we are relative to inbuilt_cfgs here
    base, aux = load_configs(config_root / "exp_mf_config.yml")
    assert base
    assert aux

    base, aux = load_configs(config_here / "exp_mf_config.yml")
    assert base
    assert aux


def test_check_paths(config_root: Path):
    path, config_name, config_path = check_paths(
        config_root, use_default_conf_dir=False
    )

    assert path == str(config_root)
    assert str(config_name) == config_root.name
    assert config_path == config_root.parent

    # Store the current working directory (i.e where script is being run from).
    cwd = os.getcwd()

    exp_config = "example_GeoCLR_config.yml"
    path2, config_name2, config_path2 = check_paths(
        exp_config, use_default_conf_dir=True
    )

    assert path2 == exp_config
    assert config_name2 == exp_config
    assert config_path2 == None

    # Change the working directory back to script location.
    os.chdir(cwd)


def test_chdir_to_default():
    def run_chdir(input, output):
        assert output == chdir_to_default(input)
        assert Path(os.getcwd()) == this_abs_path
        os.chdir(cwd)

    cwd = os.getcwd()
    this_abs_path = (Path(__file__).parent / ".." / "inbuilt_cfgs").resolve()

    config_name1 = "example_GeoCLR_config.yml"

    run_chdir(config_name1, config_name1)
    run_chdir(None, DEFAULT_CONFIG_NAME)

    config_name2 = "wrong_config.yml"

    run_chdir(config_name2, DEFAULT_CONFIG_NAME)
