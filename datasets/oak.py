import json
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from pathlib import Path

from torchvision.datasets.folder import default_loader

from data.transforms.base import IODTransform
from src.data.dataset import IODSplit
from src.data.split_loader import SplitLoader, _construct_transforms


class OAKSplit(IODSplit):
    def make_targets(self, annotation_path):
        with open(str(annotation_path), "r") as annotation_file:
            annotations = json.load(annotation_file)

            boxes = []
            areas = []
            labels = []
            crowds = []
            image_id = int(annotation_path.stem.replace("_", ""))
            for annotation in annotations:
                b = annotation["box2d"]
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                areas.append((b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))

                l = self._category_id_map[annotation["id"]]
                labels.append(l)
                crowds.append(0)

            step = torch.scalar_tensor(int(annotation_path.parent.stem.split("_")[-1]))

            box_tensor = torch.Tensor(boxes).to(torch.int64)
            label_tensor = torch.Tensor(labels).to(torch.int64)

            assert len(labels) == len(boxes)
            if len(labels) == 0:
                box_tensor = torch.empty((0, 4), dtype=torch.float32)
                label_tensor = torch.empty(0, dtype=torch.int64)
            return {
                "boxes": box_tensor,
                "labels": label_tensor,
                "image_id": torch.Tensor([image_id]).to(torch.int64),
                "step": step
            }

    def __init__(self, frames: List[Path], annotations: List[Path], root, split_id, c_id_map,
                 transforms=None,
                 loader=default_loader):
        super().__init__(split_id, root, None)

        if transforms is None:
            transforms = list()
        self._transforms: List[IODTransform] = transforms
        self.loader = loader
        self.frames = frames
        self._annotations = annotations
        self._category_id_map: Dict = c_id_map
        self.__targets = None
        self.__labels = set(self._category_id_map.values())

        self.steps = torch.unique(torch.stack([x["step"] for x in self.targets]))

    @property
    def targets(self) -> List[Dict[str, torch.Tensor]]:
        if self.__targets is None:
            self.__targets = [self.make_targets(self._annotations[i]) for i in range(len(self))]
        return self.__targets

    @property
    def labels(self) -> Set[int]:
        return self.__labels

    def __getitem__(self, idx):
        target = self.targets[idx]
        img_path = self.frames[idx]
        img = self.loader(str(img_path))

        for t in self._transforms:
            if t.needs_targets:
                img, target = t(img, target)
            else:
                img = t(img)

        return img, target

    def __len__(self):
        return len(self.frames)


_OAK_REQS = {"base_path": str, "frame_path": str, "label_path": str, "num_splits": int}


class OAKLoader(SplitLoader):
    def __init__(self, **kwargs):
        self._validate_cfg(kwargs, _OAK_REQS)
        self._root = Path(kwargs["base_path"])
        self._frames_path: Path = Path(kwargs["frame_path"])
        self._annotations_path: Path = Path(kwargs["label_path"])
        self._n_frames = kwargs.get("n_frames", -1)
        self._train_test_split = kwargs.get("train_test_split", 0.75)
        self._train_ts, self._eval_ts = _construct_transforms(kwargs.get("transforms", dict()))

        assert self._root.exists(), self._root
        assert self._frames_path.exists(), self._frames_path
        assert self._annotations_path.exists(), self._annotations_path

        prelim_frames = list(self._frames_path.rglob("*.jpg"))

        self._frames = []
        self._annotations = []
        for frame_path in prelim_frames:
            a_path = self._annotations_path / (str(frame_path.relative_to(self._frames_path)).split(".")[0] + ".json")
            if a_path.exists():
                self._frames.append(frame_path)
                self._annotations.append(a_path)

        if self._n_frames > 0:
            self._frames = self._frames[:self._n_frames]
            self._annotations = self._annotations[:self._n_frames]

        self._steps = np.unique([int(x.parent.stem.split("_")[-1]) for x in self._annotations])

        self._num_splits: int = kwargs["num_splits"]

        self._class_names, self._id_map = self._preproc_annots(self._annotations)

        # Maps each scenario within the dataset to a split
        self._split_map = self._create_split_map()
        self._splits = self._create_splits(self._split_map)

    def _preproc_annots(self, annotations_paths):
        max_idx = 0
        c_names = dict()
        id_map = dict()
        for annot_path in annotations_paths:
            with open(annot_path, "r") as annot_file:
                annots = json.load(annot_file)
                for annot in annots:
                    c = annot["category"]
                    l = annot["id"]

                    if l not in id_map:
                        id_map[l] = max_idx
                        c_names[max_idx] = c
                        max_idx += 1

        return c_names, id_map

    def _create_split_map(self) -> Dict[int, int]:
        task_splits = np.array_split(self._steps, self._num_splits)

        e_map = {}
        for split_idx, split in enumerate(task_splits):
            for task_idx in split:
                e_map[int(task_idx.item())] = split_idx

        return e_map

    def _create_splits(self, e_map: Dict[int, int]) -> Dict[int, List[int]]:
        splits = {v: [] for v in e_map.values()}
        for frame, annot in zip(self._frames, self._annotations):
            step = int(annot.parent.stem.split("_")[-1])
            split_id = e_map[step]
            assert frame.stem == annot.stem, print(frame.stem, annot.stem)
            splits[split_id].append((frame, annot))

        for split_id in splits.keys():
            splits[split_id] = list(zip(*splits[split_id]))
        return splits

    def construct_splits(self) -> Tuple[List[OAKSplit], List[OAKSplit], int]:
        train_splits = []
        test_splits = []

        for split_id, split in self._splits.items():
            indices = torch.randperm(len(split[0]))

            s_new = list(zip(*split))

            train_sz = int(len(indices) * self._train_test_split)
            train_data = list(zip(*s_new[:train_sz]))
            eval_data = list(zip(*s_new[train_sz:]))

            train_split = OAKSplit(*train_data, root=str(self._root), split_id=split_id, transforms=self._train_ts,
                                   c_id_map=self._id_map)
            eval_split = OAKSplit(*eval_data, root=str(self._root), split_id=split_id, transforms=self._eval_ts,
                                  c_id_map=self._id_map)

            train_splits.append(train_split)
            test_splits.append(eval_split)
        return train_splits, test_splits, len(self._splits)

    @property
    def class_names(self):
        return self._class_names
