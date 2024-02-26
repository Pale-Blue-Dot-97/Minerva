# -*- coding: utf-8 -*-
# Copyright 2023 Zhitong Xiong, Fahong Zhang, Yi Wang, Yilei Shi, Xiao Xiang Zhu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Implementation for the DFC2020 competition dataset.

.. note::
    Adapted from Dataset4EO https://github.com/EarthNets/Dataset4EO/tree/main
"""
# =====================================================================================================================
#                                                    METADATA
# =====================================================================================================================
__author__ = [
    "Zhitong Xiong",
    "Fahong Zhang",
    "Yi Wang",
    "Yilei Shi",
    "Xiao Xiang Zhu",
    "Harry Baker",
]
__contact__ = "hjb1d20@soton.ac.uk"
__license__ = "Apache 2.0 License"
__copyright__ = (
    "Copyright (C) 2023 Zhitong Xiong, Fahong Zhang, Yi Wang, Yilei Shi, Xiao Xiang Zhu"
)
__all__ = [
    "DFC2020",
    # "SEN12MS",
]

# =====================================================================================================================
#                                                     IMPORTS
# =====================================================================================================================
import abc
import ast
import csv
import enum
import hashlib
import importlib
import itertools
import os
import pathlib
import tarfile
from typing import (
    IO,
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    IterableWrapper,
    IterDataPipe,
    Mapper,
    RarArchiveLoader,
    ShardingFilter,
    Shuffler,
    TarArchiveLoader,
    ZipArchiveLoader,
)
from torchvision.datasets.utils import (
    _decompress,
    _detect_file_type,
    _get_google_drive_file_id,
    _get_redirect_url,
    download_file_from_google_drive,
    download_url,
    extract_archive,
    verify_str_arg,
)
from typing_extensions import Literal

NAME = "dfc2020"
FNAME = "DFC2020"
_TRAIN_LEN = 5128
_VAL_LEN = 0
_TEST_LEN = 986

D = TypeVar("D")

# pseudo-infinite until a true infinite buffer is supported by all datapipes
INFINITE_BUFFER_SIZE = 1_000_000_000

BUILTIN_DIR = pathlib.Path(__file__).parent


def _info() -> Dict[str, Any]:
    return dict(categories=read_categories_file(NAME))


# =====================================================================================================================
#                                                     CLASSES
# =====================================================================================================================
def hint_sharding(datapipe: IterDataPipe) -> ShardingFilter:
    return ShardingFilter(datapipe)


def hint_shuffling(datapipe: IterDataPipe[D]) -> Shuffler[D]:
    return Shuffler(datapipe, buffer_size=INFINITE_BUFFER_SIZE).set_shuffle(False)


def read_categories_file(name: str) -> List[Union[str, Sequence[str]]]:
    path = BUILTIN_DIR / f"{name}.categories"
    with open(path, newline="") as file:
        rows = list(csv.reader(file))
        rows = [row[0] if len(row) == 1 else row for row in rows]
        return rows


class OnlineResource(abc.ABC):
    def __init__(
        self,
        *,
        file_name: str,
        sha256: Optional[str] = None,
        preprocess: Optional[
            Union[
                Literal["decompress", "extract"], Callable[[pathlib.Path], pathlib.Path]
            ]
        ] = None,
    ) -> None:
        self.file_name = file_name
        self.sha256 = sha256

        if isinstance(preprocess, str):
            if preprocess == "decompress":
                preprocess = self._decompress
            elif preprocess == "extract":
                preprocess = self._extract
            else:
                raise ValueError(
                    f"Only `'decompress'` or `'extract'` are valid if `preprocess` is passed as string,"
                    f"but got {preprocess} instead."
                )
        self._preprocess = preprocess

    @staticmethod
    def _extract(file: pathlib.Path) -> pathlib.Path:
        return pathlib.Path(
            extract_archive(
                str(file),
                to_path=str(file).replace("".join(file.suffixes), ""),
                remove_finished=False,
            )
        )

    @staticmethod
    def _decompress(file: pathlib.Path) -> pathlib.Path:
        return pathlib.Path(_decompress(str(file), remove_finished=True))

    def _loader(self, path: pathlib.Path) -> IterDataPipe[Tuple[str, IO]]:
        if path.is_dir():
            return FileOpener(FileLister(str(path), recursive=True), mode="rb")

        dp = FileOpener(IterableWrapper((str(path),)), mode="rb")

        archive_loader = self._guess_archive_loader(path)

        if archive_loader:
            dp = archive_loader(dp)

        return dp

    _ARCHIVE_LOADERS = {
        ".tar": TarArchiveLoader,
        ".zip": ZipArchiveLoader,
        ".rar": RarArchiveLoader,
    }

    def _guess_archive_loader(
        self, path: pathlib.Path
    ) -> Optional[
        Callable[[IterDataPipe[Tuple[str, IO]]], IterDataPipe[Tuple[str, IO]]]
    ]:
        try:
            _, archive_type, _ = _detect_file_type(path.name)
        except RuntimeError:
            if path.name.endswith(".rar"):
                return self._ARCHIVE_LOADERS.get(".rar")
            return None
        return self._ARCHIVE_LOADERS.get(archive_type)  # type: ignore[arg-type]

    def load(
        self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False
    ) -> IterDataPipe[Tuple[str, IO]]:
        root = pathlib.Path(root)
        path = root / self.file_name
        # Instead of the raw file, there might also be files with fewer suffixes after decompression or directories
        # with no suffixes at all.
        stem = path.name.replace("".join(path.suffixes), "")

        # In a first step, we check for a folder with the same stem as the raw file. If it exists, we use it since
        # extracted files give the best I/O performance. Note that OnlineResource._extract() makes sure that an archive
        # is always extracted in a folder with the corresponding file name.
        folder_candidate = path.parent / stem
        if folder_candidate.exists() and folder_candidate.is_dir():
            return self._loader(folder_candidate)

        # If there is no folder, we look for all files that share the same stem as the raw file, but might have a
        # different suffix.
        file_candidates = {file for file in path.parent.glob(stem + ".*")}
        # If we don't find anything, we download the raw file.
        if not file_candidates:
            file_candidates = {
                self.download(root, skip_integrity_check=skip_integrity_check)
            }
        # If the only thing we find is the raw file, we use it and optionally perform some preprocessing steps.
        # Apply integrity check to this raw file as well
        if file_candidates == {path}:
            if not skip_integrity_check:
                self._check_sha256(path)
            if self._preprocess is not None:
                path = self._preprocess(path)
        # Otherwise, we use the path with the fewest suffixes. This gives us the decompressed > raw priority that we
        # want for the best I/O performance.
        else:
            path = min(file_candidates, key=lambda path: len(path.suffixes))
        return self._loader(path)

    @abc.abstractmethod
    def _download(self, root: pathlib.Path) -> None:
        pass

    def download(
        self, root: Union[str, pathlib.Path], *, skip_integrity_check: bool = False
    ) -> pathlib.Path:
        root = pathlib.Path(root)
        self._download(root)
        path = root / self.file_name
        if self.sha256 and not skip_integrity_check:
            self._check_sha256(path)
        return path

    def _check_sha256(
        self, path: pathlib.Path, *, chunk_size: int = 1024 * 1024
    ) -> None:
        hash = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(chunk_size), b""):
                hash.update(chunk)
        sha256 = hash.hexdigest()
        if sha256 != self.sha256:
            raise RuntimeError(
                f"After the download, the SHA256 checksum of {path} didn't match the expected one: "
                f"{sha256} != {self.sha256}"
            )


class HttpResource(OnlineResource):
    def __init__(
        self,
        url: str,
        *,
        file_name: Optional[str] = None,
        mirrors: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            file_name=file_name or pathlib.Path(urlparse(url).path).name, **kwargs
        )
        self.url = url
        self.mirrors = mirrors
        self._resolved = False

    def resolve(self) -> OnlineResource:
        if self._resolved:
            return self

        redirect_url = _get_redirect_url(self.url)
        if redirect_url == self.url:
            self._resolved = True
            return self

        meta = {
            attr.lstrip("_"): getattr(self, attr)
            for attr in (
                "file_name",
                "sha256",
                "_preprocess",
            )
        }

        gdrive_id = _get_google_drive_file_id(redirect_url)
        if gdrive_id:
            return GDriveResource(gdrive_id, **meta)

        http_resource = HttpResource(redirect_url, **meta)
        http_resource._resolved = True
        return http_resource

    def _download(self, root: pathlib.Path) -> None:
        if not self._resolved:
            return self.resolve()._download(root)

        for url in itertools.chain((self.url,), self.mirrors):
            try:
                download_url(url, str(root), filename=self.file_name, md5=None)
            # TODO: make this more precise
            except Exception:  # nosec: B112
                continue

            return
        else:
            # TODO: make this more informative
            raise RuntimeError("Download failed!")


class GDriveResource(OnlineResource):
    def __init__(self, id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.id = id

    def _download(self, root: pathlib.Path) -> None:
        download_file_from_google_drive(
            self.id, root=str(root), filename=self.file_name, md5=None
        )


class Dataset(IterDataPipe[Dict[str, Any]], abc.ABC):
    @staticmethod
    def _verify_str_arg(
        value: str,
        arg: Optional[str] = None,
        valid_values: Optional[Collection[str]] = None,
        *,
        custom_msg: Optional[str] = None,
    ) -> str:
        return verify_str_arg(value, arg, valid_values, custom_msg=custom_msg)

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        skip_integrity_check: bool = False,
        dependencies: Collection[str] = (),
    ) -> None:
        for dependency in dependencies:
            try:
                importlib.import_module(dependency)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"{type(self).__name__}() depends on the third-party package '{dependency}'. "
                    f"Please install it, for example with `pip install {dependency}`."
                ) from None

        self._root = pathlib.Path(root).expanduser().resolve()
        resources = [
            resource.load(self._root, skip_integrity_check=skip_integrity_check)
            for resource in self._resources()
        ]
        self._dp = self._datapipe(resources)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self._dp

    @abc.abstractmethod
    def _resources(self) -> List[OnlineResource]:
        pass

    @abc.abstractmethod
    def _datapipe(
        self, resource_dps: List[IterDataPipe]
    ) -> IterDataPipe[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def _generate_categories(self) -> Sequence[Union[str, Sequence[str]]]:
        raise NotImplementedError


class DFC2020(Dataset):
    """
    - **homepage**: https://www.iarai.ac.at/rsbenchmark4uss/
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        *,
        split: str = "train",
        skip_integrity_check: bool = False,
    ) -> None:
        self._split = self._verify_str_arg(split, "split", ("train", "test"))
        self.root = root
        self._categories = _info()["categories"]
        self.CLASSES = (
            "Forest",
            "Shrubland",
            "Grassland",
            "Wetland",
            "Cropland",
            "Urban/Built-up",
            "Barren",
            "Water",
        )
        self.PALETTE = [
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
        ]

        super().__init__(root, skip_integrity_check=skip_integrity_check)

    _TRAIN_VAL_ARCHIVES = {
        "trainval": (
            "landslide4sense.tar",
            "c7f6678d50c7003eba47b3cace8053c9bfa6b4692cd1630fe2d6b7bec11ccc77",
        ),
    }

    def decompress_integrity_check(self, decom_dir):
        train_img_dir = os.path.join(decom_dir, "train", "img")
        train_mask_dir = os.path.join(decom_dir, "train", "mask")
        val_img_dir = os.path.join(decom_dir, "val", "img")

        if (
            not os.path.exists(train_img_dir)
            or not os.path.exists(train_mask_dir)
            or not os.path.exists(val_img_dir)
        ):
            return False

        # num_train_img = len(os.listdir(train_img_dir))
        # num_train_mask = len(os.listdir(train_mask_dir))
        # num_val_img = len(os.listdir(val_img_dir))

        return True
        # return (num_train_img == _TRAIN_LEN) and \
        #         (num_train_mask == _TRAIN_LEN) and \
        #         (num_val_img == _VAL_LEN)

    def _resources(self) -> List[OnlineResource]:
        file_name, sha256 = self._TRAIN_VAL_ARCHIVES["trainval"]
        decom_dir = os.path.join(self.root, "landslide4sense")
        self.decom_dir = decom_dir
        archive = HttpResource(
            "https://syncandshare.lrz.de/dl/fiLurHQ9Cy4NwvmPGYQe7RWM/{}".format(
                file_name
            ),
            sha256=sha256,
        )

        if not self.decompress_integrity_check(decom_dir):
            print("Decompressing the tar file...")
            with tarfile.open(os.path.join(self.root, file_name), "r:gz") as tar:
                tar.extractall(decom_dir)  # nosec: B202
                tar.close()

        return [archive]

    def _is_in_folder(
        self, data: Tuple[str, Any], *, name: str, depth: int = 1
    ) -> bool:
        path = pathlib.Path(data)
        in_folder = name in str(path.parent)
        return in_folder

    def _prepare_sample(self, data):
        # label_path, label = None, None
        image_path, label_path = data
        """
        img = h5py.File(image_path, 'r')['image'][()]
        img = torch.tensor(img.astype(np.uint8)).permute(2, 0, 1)
        label = h5py.File(label_path, 'r')['image'][()]
        label = torch.tensor(label.astype(np.uint8))
        """
        img_info = dict({"filename": image_path, "ann": dict({"seg_map": label_path})})
        return img_info

    class _Demux(enum.IntEnum):
        TRAIN = 0
        TEST = 1
        VAL = 2

    def _classify_archive(self, data: Tuple[str, Any]) -> Optional[int]:
        if self._is_in_folder(data, name="train", depth=2):
            return self._Demux.TRAIN
        if self._is_in_folder(data, name="val", depth=2):
            return self._Demux.VAL
        elif self._is_in_folder(data, name="test", depth=2):
            return self._Demux.TEST
        else:
            return None

    def _datapipe(self, res):
        # image_dp = FileLister(root=os.path.join(self.root, FNAME, 'images'), recursive=True)
        # train_img_dp, test_img_dp = image_dp.demux(
        #     num_instances=2,
        #     classifier_fn=self._classify_archive,
        #     drop_none=True,
        #     buffer_size=INFINITE_BUFFER_SIZE
        # )

        # label_dp = FileLister(root=os.path.join(self.root, FNAME, 'classes'), recursive=True)
        # train_label_dp, test_label_dp = label_dp.demux(
        #     num_instances=2,
        #     classifier_fn=self._classify_archive,
        #     drop_none=True,
        #     buffer_size=INFINITE_BUFFER_SIZE
        # )

        # train_dp = train_img_dp.zip(train_label_dp)
        # test_dp = test_img_dp.zip(test_label_dp)

        """tfs = transforms.Compose(transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop((224, 224), scale=[0.5, 1]))"""

        ndp = ast.literal_eval(self._split + "_dp")
        ndp = hint_shuffling(ndp)
        ndp = hint_sharding(ndp)
        ndp = Mapper(ndp, self._prepare_sample)
        # ndp = ndp.map(tfs)
        return ndp

    def __len__(self) -> int:
        return {"train": _TRAIN_LEN, "val": _VAL_LEN, "test": _TEST_LEN}[self._split]


# class BaseSenS12MS(NonGeoDataset):
#     # Mapping from IGBP to DFC2020 classes.
#     DFC2020_CLASSES = [
#         0,  # Class 0 unused in both schemes.
#         1,
#         1,
#         1,
#         1,
#         1,
#         2,
#         2,
#         3,  # --> will be masked if no_savanna == True
#         3,  # --> will be masked if no_savanna == True
#         4,
#         5,
#         6,  # 12 --> 6
#         7,  # 13 --> 7
#         6,  # 14 --> 6
#         8,
#         9,
#         10,
#     ]

#     # Indices of sentinel-2 high-/medium-/low-resolution bands
#     S2_BANDS_HR = [2, 3, 4, 8]
#     S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
#     S2_BANDS_LR = [1, 10, 11]

#     splits: List[str] = []

#     igbp = False

#     def __init__(
#         self,
#         root: str,
#         split="val",
#         no_savanna=False,
#         use_s2hr=False,
#         use_s2mr=False,
#         use_s2lr=False,
#         use_s1=False,
#         labels=False,
#     ) -> None:
#         super().__init__()

#         # Make sure at least one of the band sets are requested.
#         if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
#             raise ValueError(
#                 "No input specified, set at least one of "
#                 + "use_[s2hr, s2mr, s2lr, s1] to True!"
#             )

#         self.use_s2hr = use_s2hr
#         self.use_s2mr = use_s2mr
#         self.use_s2lr = use_s2lr
#         self.use_s1 = use_s1

#         assert split in self.splits
#         self.no_savanna = no_savanna

#         # Provide number of input channels
#         self.n_inputs = self.get_ninputs()

#         # Provide index of channel(s) suitable for previewing the input
#         self.display_channels, self.brightness_factor = self.get_display_channels()

#         # Provide number of classes.
#         if no_savanna:
#             self.n_classes = max(self.DFC2020_CLASSES) - 1
#         else:
#             self.n_classes = max(self.DFC2020_CLASSES)

#         self.labels = labels

#         self.root = Path(root)

#         # Make sure parent dir exists.
#         assert self.root.exists()

#         self.samples: List[Dict[str, str]]

#     def load_sample(
#         self,
#         sample: Dict[str, str],
#     ) -> Dict[str, Any]:
#         """Util function for reading data from single sample.

#         Args:
#             sample (dict[str, str]): Dictionary defining the paths to the files for each modality of the sample.

#         Returns:
#             dict[str, ~typing.Any]: Dictionary defining the sample in each modality.
#         """

#         use_s2 = self.use_s2hr or self.use_s2mr or self.use_s2lr

#         # Load just S2 data.
#         if use_s2 and not self.use_s1:
#             img = self.load_s2(sample["s2"])

#         # Load S1 and S2 data.
#         elif self.use_s1 and use_s2:
#             img = torch.concatenate(
#                 (self.load_s2(sample["s2"]), self.load_s1(sample["s1"])), dim=0
#             )  # type: ignore[assignment]

#         # Load just S1 data.
#         elif self.use_s1 and not use_s2:
#             img = self.load_s1(sample["s1"])

#         else:
#             raise ValueError("No data selected")

#         # Load labels.
#         if self.labels:
#             lc = self.load_lc(sample["lc"])
#             return {"image": img, "mask": lc, "id": sample["id"]}
#         else:
#             return {"image": img, "id": sample["id"]}

#     def get_ninputs(self) -> int:
#         """Calculate number of input channels.

#         Returns:
#             int: Number of input channels.
#         """
#         n_inputs = 0
#         if self.use_s2hr:
#             n_inputs += len(self.S2_BANDS_HR)
#         if self.use_s2mr:
#             n_inputs += len(self.S2_BANDS_MR)
#         if self.use_s2lr:
#             n_inputs += len(self.S2_BANDS_LR)
#         if self.use_s1:
#             n_inputs += 2
#         return n_inputs

#     def get_display_channels(self) -> Tuple[List[int], int]:
#         """Select channels for preview images.

#         Returns:
#             tuple[list[int] | int, int]: Tuple of the index of display channels and the brightness factor.
#         """
#         if self.use_s2hr and self.use_s2lr:
#             display_channels = [3, 2, 1]
#             brightness_factor = 3
#         elif self.use_s2hr:
#             display_channels = [2, 1, 0]
#             brightness_factor = 3
#         elif not (self.use_s2hr or self.use_s2mr or self.use_s2lr):
#             display_channels = [0]
#             brightness_factor = 1
#         else:
#             display_channels = [0]
#             brightness_factor = 3

#         return display_channels, brightness_factor

#     def load_s2(self, path: str) -> FloatTensor:
#         """Util function for reading and cleaning Sentinel2 data.

#         Args:
#             path (str): Path to patch of Sentinel2 data.

#         Returns:
#             FloatTensor: Patch of Sentinel2 data.
#         """
#         bands_selected = []
#         if self.use_s2hr:
#             bands_selected.extend(self.S2_BANDS_HR)
#         if self.use_s2mr:
#             bands_selected.extend(self.S2_BANDS_MR)
#         if self.use_s2lr:
#             bands_selected.extend(self.S2_BANDS_LR)

#         bands_selected = sorted(bands_selected)

#         # Read data out of the TIFF file.
#         with rasterio.open(path) as data:
#             s2 = data.read(bands_selected)

#         s2 = s2.astype(np.float32)
#         s2 = np.clip(s2, 0, 10000)
#         s2 /= 10000

#         # Cast to 32-bit float.
#         s2 = s2.astype(np.float32)

#         # Convert to Tensor.
#         s2 = torch.tensor(s2)
#         assert isinstance(s2, FloatTensor)

#         return s2

#     @staticmethod
#     def load_s1(path: str) -> FloatTensor:
#         """Util function for reading and cleaning Sentinel1 data.

#         Args:
#             path (str): Path to patch of Sentinel1 data.

#         Returns:
#             FloatTensor: Patch of Sentinel1 data.
#         """
#         with rasterio.open(path) as data:
#             s1 = data.read()

#         s1 = s1.astype(np.float32)
#         s1 = np.nan_to_num(s1)
#         s1 = np.clip(s1, -25, 0)
#         s1 /= 25
#         s1 += 1

#         # Cast to 32-bit float.
#         s1 = s1.astype(np.float32)

#         # Convert to Tensor.
#         s1 = torch.tensor(s1)
#         assert isinstance(s1, FloatTensor)

#         return s1

#     def load_lc(self, path: str) -> Tensor:
#         """Util function for reading Sentinel land cover data.

#         Args:
#             path (str): Path to the land cover labels for the patch.

#         Returns:
#             Tensor: Label mask for the patch.
#         """

#         # Load labels.
#         with rasterio.open(path) as data:
#             lc = data.read(1)

#         # Convert IGBP to dfc2020 classes.
#         if self.igbp:
#             lc = np.take(self.DFC2020_CLASSES, lc)
#         else:
#             lc = lc.astype(np.int64)

#         # Adjust class scheme to ignore class savanna.
#         if self.no_savanna:
#             lc[lc == 3] = 0
#             lc[lc > 3] -= 1

#         # Convert to zero-based labels and set ignore mask.
#         lc -= 1
#         lc[lc == -1] = 255

#         # Convert to tensor.
#         lc = torch.tensor(lc, dtype=torch.int8)
#         assert isinstance(lc, Tensor)

#         return lc

#     def __getitem__(self, index: int) -> Dict[str, Any]:
#         """Get a single example from the dataset"""

#         # Get and load sample from index file.
#         sample = self.samples[index]
#         return self.load_sample(sample)

#     def __len__(self) -> int:
#         """Get number of samples in the dataset"""
#         return len(self.samples)


# class DFC2020(BaseSenS12MS):
#     """PyTorch dataset class for the DFC2020 dataset"""

#     splits = ["val", "test"]
#     igbp = False

#     def __init__(
#         self,
#         root: str,
#         split="val",
#         no_savanna=False,
#         use_s2hr=False,
#         use_s2mr=False,
#         use_s2lr=False,
#         use_s1=False,
#         labels=False,
#     ) -> None:
#         super(DFC2020, self).__init__(
#             root, split, no_savanna, use_s2hr, use_s2mr, use_s2lr, use_s1, labels
#         )

#         # Build list of sample paths.
#         if split == "val":
#             path = self.root / "ROIs0000_validation" / "s2_validation"
#         else:
#             path = self.root / "ROIs0000_test" / "s2_0"

#         s2_locations = glob(str(path / "*.tif"), recursive=True)
#         self.samples = []
#         for s2_loc in tqdm(s2_locations, desc="[Load]"):
#             s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
#             lc_loc = s2_loc.replace("_dfc_", "_lc_").replace("s2_", "dfc_")
#             self.samples.append(
#                 {"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": Path(s2_loc).name}
#             )

#         # Sort list of samples.
#         self.samples = sorted(self.samples, key=lambda i: i["id"])

#         print(f"Loaded {len(self.samples)} samples from the DFC2020 {split} subset")


# # TODO: Add tests to cover SEN12MS dataset
# class SEN12MS(BaseSenS12MS):  # pragma: no cover
#     """PyTorch dataset class for the SEN12MS dataset

#     Expects dataset dir as:
#     >>> - SEN12MS_holdOutScenes.txt
#     >>> - ROIsxxxx_y
#     >>>     - lc_n
#     >>>     - s1_n
#     >>>     - s2_n

#     SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
#     train/val split and can be obtained from:
#     https://github.com/MSchmitt1984/SEN12MS/blob/master/splits
#     """

#     splits = ["train", "holdout"]
#     igbp = True

#     def __init__(
#         self,
#         root: str,
#         split="train",
#         no_savanna=False,
#         use_s2hr=False,
#         use_s2mr=False,
#         use_s2lr=False,
#         use_s1=False,
#         labels=False,
#     ) -> None:
#         super(SEN12MS, self).__init__(
#             root, split, no_savanna, use_s2hr, use_s2mr, use_s2lr, use_s1, labels
#         )

#         # Find and index samples.
#         self.samples = []
#         if split == "train":
#             pbar = tqdm(total=162556)  # we expect 541,986 / 3 * 0.9 samples
#         else:
#             pbar = tqdm(total=18106)  # we expect 541,986 / 3 * 0.1 samples
#         pbar.set_description("[Load]")

#         val_list = list(
#             pd.read_csv(self.root / "SEN12MS_holdOutScenes.txt", header=None)[0]
#         )
#         val_list = [x.replace("s1_", "s2_") for x in val_list]

#         # Compile a list of paths to all samples
#         if split == "train":
#             train_list = []
#             for seasonfolder in [
#                 "ROIs1970_fall",
#                 "ROIs1158_spring",
#                 "ROIs2017_winter",
#                 "ROIs1868_summer",
#             ]:
#                 train_list += [
#                     str(Path(seasonfolder) / x)
#                     for x in (self.root / seasonfolder).iterdir()
#                 ]
#             train_list = [x for x in train_list if "s2_" in x]
#             train_list = [x for x in train_list if x not in val_list]
#             sample_dirs = train_list
#         else:
#             sample_dirs = val_list

#         for folder in sample_dirs:
#             s2_locations = glob(str(self.root / f"{folder}/*.tif"), recursive=True)

#             # INFO there is one "broken" file in the sen12ms dataset with nan
#             #      values in the s1 data. we simply ignore this specific sample
#             #      at this point. id: ROIs1868_summer_xx_146_p202
#             if folder == "ROIs1868_summer/s2_146":
#                 broken_file = str(
#                     self.root
#                     / "ROIs1868_summer"
#                     / "s2_146"
#                     / "ROIs1868_summer_s2_146_p202.tif"
#                 )
#                 s2_locations.remove(broken_file)
#                 pbar.write("ignored one sample because of nan values in the s1 data")

#             for s2_loc in s2_locations:
#                 s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
#                 lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")

#                 pbar.update()
#                 self.samples.append(
#                     {"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": Path(s2_loc).name}
#                 )

#         pbar.close()

#         # Sort list of samples
#         self.samples = sorted(self.samples, key=lambda i: i["id"])

#         print(f"Loaded {len(self.samples)} samples from the SEN12MS {split} subset")
