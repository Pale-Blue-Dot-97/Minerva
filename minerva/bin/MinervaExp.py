"""Script to execute the creation, fitting and testing of a computer vision neural network model to classify land cover.

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

TODO:
    * Add arg parsing from CLI
    * Add ability to conduct hyper-parameter iterative variation experimentation
"""
# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
from minerva.utils import config
from minerva.trainer import Trainer


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main():
    trainer = Trainer(**config)
    trainer.fit()

    if config["pre_train"]:
        trainer.save_backbone()

    else:
        trainer.test()


if __name__ == "__main__":
    main()
