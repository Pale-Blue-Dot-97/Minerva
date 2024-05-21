# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2024 Harry Baker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @org: University of Southampton
# Created under a project funded by the Ordnance Survey Ltd.
r"""Script to make a LMBD file of the SSL4EO-S12 dataset.
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = "Harry Baker"
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "MIT License"
__copyright__ = "Copyright (C) 2024 Harry Baker"

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import argparse
import os
import shutil

from torch.utils.data import DataLoader
from tqdm import tqdm

from minerva.datasets.ssl4eos12 import MinervaSSL4EO, make_lmdb, random_subset


# =====================================================================================================================
#                                                      MAIN
# =====================================================================================================================
def main(args) -> None:

    # Make lmdb dataset.
    if args.make_lmdb_file:
        if os.path.isdir(args.save_path):
            shutil.rmtree(args.save_path)

        train_dataset = MinervaSSL4EO(
            root=args.root,
            normalize=args.normalize,
            mode=args.mode,
            bands=args.bands,
            dtype=args.dtype,
        )
        train_subset = random_subset(train_dataset, frac=args.frac, seed=42)

        make_lmdb(
            train_subset, args.save_path, num_workers=args.num_workers, mode=args.mode
        )

    # Check dataset class.
    else:
        train_dataset = MinervaSSL4EO(root=args.root, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)

        for idx, (s1, s2a, s2c) in tqdm(
            enumerate(train_loader), total=len(train_dataset)
        ):
            if idx > 0:
                break

            print(
                f"{s1.shape=}\n{s1.dtype=}\n{s2a.shape=}\n{s2a.dtype=}\n{s2c.shape=}\n{s2c.dtype=}"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--make_lmdb_file", action="store_true", default=False)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="s2a")
    parser.add_argument("--bands", nargs="*", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="uint8")
    args = parser.parse_args()

    main(args)
