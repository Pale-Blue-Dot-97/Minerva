---
title: 'Minerva: The Remote Sensing Machine Learning Framework'
tags:
  - Python
  - machine learning
  - artificial intelligence
  - remote sensing
authors:
  - name: Harry Baker
    orcid: 0000-0002-4382-8196
    equal-contrib: true
    affiliation: "1, 2"
    corresponding: true
  - name: Jonathon Hare
    orcid: 0000-0003-2921-4283
    equal-contrib: false
    affiliation: "1"
    corresponding: false
  - name: Isabel Sargent
    orcid: 0000-0002-3982-7318
    equal-contrib: false
    affiliation: "1"
    corresponding: false
  - name: Adam Prugel Bennett
    # orcid:
    equal-contrib: false
    affiliation: "1"
    corresponding: false
  - name: Steve Coupland
    # orcid:
    equal-contrib: false
    affiliation: "2"
    corresponding: false
affiliations:
 - name: University of Southampton, University Road, Southampton, UK, SO17 1BJ
   index: 1
 - name: Ordnance Survey, Explorer House, Adanac Drive, Southampton, UK, S016 0AS
   index: 2
date: 06 January 2025
bibliography: paper.bib

---

# Summary

Remote-sensing and earth observation requires its own distictive formats for data from number of bands in imagery, geo-spatial co-ordinates and applications. Most computer vision machine learning research is performed in domains outside of remote sensing and thus, many of the existing libraries in Python for ML lack the features we need. Thankfully, the `torchgeo` [@Stewart_TorchGeo_2022] library extends the popular `torch` library for remote sensing researchers. `minerva` takes this functionality further introducing a framework for researchers to design and execute remote sensing focussed machine learning experiments at scale. `minerva` includes support for `hydra` for experiment configuration, `wandb` logging support, use of high performance computing via `SLURM`, along with numerous other QoL utilities.

# Statement of need

The `minerva` package is primarily designed for use in training, validating and testing machine learning models in the remote sensing domain. Using `hydra`, experiments are configured via `YAML` files that `minerva` interprets, allowing users a great degree of flexibility. It was orginally conceived in 2021 when it was found that `pytorch-lightning` did not offer the level of flexibility required for our particular use-case. Since then, `minerva` has grown from a PhD research repositry into a fully-fledged package -- albeit still in its beta infancy -- with regular users.

# Package Structure

<!---
+-------------------+--------------+-----------------------------------------------+
| Sub-Package       | Module       | Description                                   |
|                   |              |                                               |
+:=================:+:============:+:=============================================:+
| datasets          | collators    | Collation functions designed for `minerva`    |
|                   +--------------+-----------------------------------------------+
|                   | dfc          | Implementation of DFC2020 competition dataset |
|                   +--------------+-----------------------------------------------+
|                   | factory      | Functionality for constructing                |
|                   |              | datasets and `DataLoader` in `minerva`        |
|                   +--------------+-----------------------------------------------+
|                   | naip         | Adapted version of `torchgeo` `NAIP` dataset  |
|                   |              | that works with the NAIP files provided in    |
|                   |              | the ChesapeakeCVPR dataset                    |
|                   +--------------+-----------------------------------------------+
|                   | paired       | Datasets to handle paired sampling            |
|                   |              | for use in Siamese learning                   |
|                   +--------------+-----------------------------------------------+
|                   | ssl4eos12    | Simple adaption of the `torchgeo` Sentinel2   |
|                   |              | dataset for use with the SSL4EO-S12 dataset   |
|                   +--------------+-----------------------------------------------+
|                   | utils        | Utility functions for datasets in `minerva`   |
+-------------------+--------------+-----------------------------------------------+
| logger            | steplog      | Loggers to handle the logging                 |
|                   |              | from each step of a task                      |
|                   +--------------+-----------------------------------------------+
|                   | tasklog      | Loggers designed to handle the logging        |
|                   |              | and analysis for a whole task                 |
+-------------------+--------------+-----------------------------------------------+
| models            | core         | Core utility functions and abstract classes   |
|                   |              | underpinning `models`                         |
|                   +--------------+-----------------------------------------------+
|                   | fcn          | Fully Convolutional Network (FCN) models      |
|                   +--------------+-----------------------------------------------+
|                   | psp          | Pyramid Spatial Pooling Net (PSPNet) adapted  |
|                   |              | for use in `minerva`                          |
|                   +--------------+-----------------------------------------------+
|                   | resnet       | ResNets adapted for use in `minerva`          |
|                   +--------------+-----------------------------------------------+
|                   | siamese      | Siamese models adapted for use in `minerva`   |
|                   +--------------+-----------------------------------------------+
|                   | unet         | Module containing UNet models. Most code      |
|                   |              | from https://github.com/milesial/Pytorch-UNet |
+-------------------+--------------+-----------------------------------------------+
| tasks             | core         | Core functionality of `tasks`, defining the   |
|                   |              | abstract `MinervaTask` class                  |
|                   +--------------+-----------------------------------------------+
|                   | epoch        | Standard epoch style for use with generic     | 
|                   |              | model fitting                                 |
|                   +--------------+-----------------------------------------------+
|                   | knn          | K-Nearest Neighbour (KNN) validation task     |
|                   +--------------+-----------------------------------------------+
|                   | tsne         | TSNE clustering task                          |
+-------------------+--------------+-----------------------------------------------+
| utils             | config_load  | Handles the loading of config files           | 
|                   |              | and checking paths                            |
|                   +--------------+-----------------------------------------------+
|                   | runner       | Generic functionality for running `minerva`   |
|                   |              | scripts, setting up distributed computing,    |
|                   |              | handling SLURM variables and                  |
|                   |              | Weights and Biases logging                    |
|                   +--------------+-----------------------------------------------+
|                   | utils        | Utility functions                             |
|                   +--------------+-----------------------------------------------+
|                   | visutils     | Visualisation utility functionality           |
+-------------------+--------------+-----------------------------------------------+
| _root_            | loss         | Specialised loss functions for `minerva`      |
|                   +--------------+-----------------------------------------------+
|                   | modelio      | Standarised functions to handle various IO    |
|                   |              | structures from `dataloaders` and to models   |
|                   +--------------+-----------------------------------------------+
|                   | optimisers   | Custom `torch` optimisers. Consists soley of  |
|                   |              | adapted LARS optimiser from Noah Golmant      |
|                   +--------------+-----------------------------------------------+
|                   | pytorchtools | `EarlyStopping` functionality to track when   |
|                   |              | the training of a model should stop. By       |
|                   |              | Bjarte Mehus Sunde                            |
|                   +--------------+-----------------------------------------------+
|                   | samplers     | Custom samplers for `torchgeo` datasets       |
|                   +--------------+-----------------------------------------------+
|                   | trainer      | Module containing `Trainer` that handles      | 
|                   |              | the fitting of models                         |
|                   +--------------+-----------------------------------------------+
|                   | transforms   | Custom transforms to handle multi-spectral    | 
|                   |              | imagery and geospatial data                   |
+===================+==============+===============================================+
| Footer                                                                           |
+===================+==============+===============================================+
--->

# Comparison to Similar Projects

Given the rapid expansion and advancement of machine learning research since 2014, it will not surprise the reader that there are a wide variety of open-source libraries that support ML practitioners. However, with regards to the remote-sensing focussed researcher, there is a far smaller selection. The stand out package, which `minerva` heavily relies on, is `torchgeo` [@Stewart_TorchGeo_2022]. Like `minerva`, `torchgeo` has matured significantly over the last few years to become an invaulable tool for remote-sensing AI researchers. Its stand out features include its native support for handling GeoTiffs and geospatial information, making it effortless for a user to currate and manipulate datasets to train a remote-sensing focussed model on. `torchgeo.datamodule` also offers much of the same framework features `minerva` does but takes a slightly different approach as to how an experiment is defined.

`minerva` also bears similarities to `pytorch-lightning` in its internal structure. Like `pytorch-lightning`, the internal workings of performing each step of model training is abstracted away in `minerva` from a user. The major difference between the libraries (other than the former's far superior stability and maturity) is `minerva`'s focus on configuring experiments via `YAML` configuration. This stems largely from `minerva`'s raison d'etre -- to act as a framework to facilitate research experiments. As such, `minerva` does lack the same flexibility that `pytorch-lightning` offers its users.

# User Guide

The core functionality of `minerva` provides the modules to define `models` to fit and test, `loaders` to pre-process,
load and parse data, and a `Trainer` to handle all aspects of a model fitting. Below is a MWE of creating datasets,
initialising a Trainer and model, and fitting and testing that model then outputting the results:

## MWE Driver Script

```python
import hydra
from omegaconf import DictConfig

from minerva.trainer import Trainer  # Class designed to handle fitting of model.


@hydra.main(version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Initialise a Trainer. Also creates the model.
    trainer = Trainer(**cfg)

    # Run the fitting (train and validation epochs).
    trainer.fit()

    # Run the testing epoch and output results.
    trainer.test()
```

See `scripts\MinervaExp.py` as an example script implementing `minerva`.

## Config Structure

`minerva` uses `hydra` and `omegaconf` to handle config files and the CLI for users. This allows for hyper-parameters for experiments to be easily reproducible and flexible. `hydra` also provides an adaptable CLI that integrates with the config seemlessly. It also provides functionality for hyper-parameter sweeps.

See `minerva\inbuilt_cfgs\example_config.yml` as an example config file.

## Distributed Computing

`minerva` fully supports the `SLURM` protocol for high performance and distributed computing by utilising the `submitit-slurm` plugin for `hydra`. Configs containing the `SLURM` variables for a user's job should be contained in a `hydra\launcher` sub-directory and contain the SLURM variables the user wishes to request for their job, similar to how they would be specified in a SBATCH file. Reference also needs to be made to `submitit-slurm` to activate the plugin, as shown in the example below: 

```yaml
# configs/hydra/launcher/swarm_h100_2gpu.yaml
defaults:
  - submitit_slurm

_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
submitit_folder: /.submitit/%j  # Where to send the output logs of the job to. %j is the job ID

# SLURM variables
partition: swarm_h100
timeout_min: 1000
tasks_per_node: 1
nodes: 1
gres: gpu:2
cpus_per_task: 10
mem_gb: 312
```

Then this config should be referenced in the main experiment config, as shown below:

```yaml
# configs/experiment_config.yaml
# === HYDRA DEFAULTS ==========================================================
defaults:
   - override hydra/launcher: swarm_h100_2gpu
   - _self_

task: 1
```

The `task` number should match the number of nodes the job is to be submitted to.

### Caveat

Unfortunately, it was found that due to pickling issues with `multiprocessing` used by `pytorch`, coupled with the requirements of `hydra` and `submitit-slurm`, the `SLURM` distributed computing and hyper-parameter sweep functionality requires the driver code to be located within the `runner` module rather than in a user defined script, like shown in the MWE. Therefore, `scripts/MinervaExp.py` is actually structured as such:

```python
from typing import Optional

import hydra
from omegaconf import DictConfig
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from minerva.utils import DEFAULT_CONF_DIR_PATH, DEFAULT_CONFIG_NAME, runner, utils


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
@hydra.main(
    version_base="1.3",
    config_path=str(DEFAULT_CONF_DIR_PATH),
    config_name=DEFAULT_CONFIG_NAME,
)
@runner.distributed_run
def main(gpu: int, wandb_run: Optional[Run | RunDisabled], cfg: DictConfig) -> None:
    # Due to the nature of multiprocessing and its interaction with hydra, wandb and SLURM,
    # the actual code excuted in the job is contained in `run_trainer` in `runner`.
    #
    # Any code placed here will not be executed with multiprocessing!

    pass


if __name__ == "__main__":
    # Print Minerva banner.
    utils._print_banner()

    with runner.WandbConnectionManager():
        # Run the specified main with distributed computing and the arguments provided.
        main()
```

where `minerva.utils.runner.run_trainer` `minerva.utils.runner.distributed_run` is:

```python
# minerva/utils/runner.py

def distributed_run(
    run: Callable[[int, Optional[Run | RunDisabled], DictConfig], Any],
) -> Callable[..., Any]:
    """Runs the supplied function and arguments with distributed computing according to arguments.

    :func:`_run_preamble` adds some additional commands to initialise the process group for each run
    and allocating the GPU device number to use before running the supplied function.

    Note:
        ``args`` must contain the attributes ``rank``, ``world_size`` and ``dist_url``. These can be
        configured using :func:`config_env_vars` or :func:`config_args`.

    Args:
        run (~typing.Callable[[int, ~argparse.Namespace], ~typing.Any]): Function to run with distributed computing.
        args (~argparse.Namespace): Arguments for the run and to specify the variables for distributed computing.
    """

    OmegaConf.register_new_resolver("cfg_load", _config_load_resolver, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "to_patch_size", _construct_patch_size, replace=True
    )

    @functools.wraps(run)
    def inner_decorator(cfg: DictConfig):
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)

        cfg = config_args(cfg)

        if cfg.world_size <= 1:
            # Setups up the `wandb` run.
            wandb_run, cfg = setup_wandb_run(0, cfg)

            # Run the experiment.
            run_trainer(0, wandb_run, cfg)

        else:  # pragma: no cover
            try:
                mp.spawn(_run_preamble, (run_trainer, cfg), cfg.ngpus_per_node)  # type: ignore[attr-defined]
            except KeyboardInterrupt:
                dist.destroy_process_group()  # type: ignore[attr-defined]

    return inner_decorator

def run_trainer(
    gpu: int, wandb_run: Optional[Run | RunDisabled], cfg: DictConfig
) -> None:
    trainer = Trainer(
        gpu=gpu,
        wandb_run=wandb_run,
        **cfg,  # type: ignore[misc]
    )

    if not cfg.get("eval", False):
        trainer.fit()

    if cfg.get("pre_train", False) and gpu == 0:
        trainer.save_backbone()
        trainer.close()

    if not cfg.get("pre_train", False):
        trainer.test()
```

The flexibility of config files and `minerva` should, however, allow this simple driver code to provide all the functionality a user may require. In the unlikely event a user requires custom driver code, they would be required to copy the above example and adapt to their needs. It was found that adding a `callable` with the driver code in as an arg to `distributed_run` did not work -- it must be defined in the same scope. Further debugging of this peculilar pickling error is required to provide a far more satisfactory user experience. 

# Conclusion

# Acknowledgements

This work was possible thanks to a PhD funded by the Ordnance Survey. Thanks must also go to my supervisors, Prof. Jonathon Hare, Dr. Isabel Sargent, Prof. Adam Prugel-Bennett and Steve Coupland, whose guidance and support have contributed to the creation of `minerva` and its associated work. Contributions to `minerva` were also made by several people past and present at the Ordnance Survey not listed above as authors:

* Jo Walsh
* Navid Rahimi
* Joe Guyatt
* Ben Dickens
* Kitty Varghese

# References
