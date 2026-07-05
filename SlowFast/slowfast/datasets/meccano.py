#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os

import numpy as np
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
from PIL import Image

import slowfast.utils.logging as logging

from . import sampling
from . import transform
from . import utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Meccano(torch.utils.data.Dataset):
    """
    MECCANO frame-based dataset loader for the local fixed layout.
    """

    SPLIT_FILES = {
        "train": "MECCANO_train_actions.csv",
        "val": "MECCANO_val_actions.csv",
        "test": "MECCANO_test_actions.csv",
    }
    FRAME_DIRS = {
        "train": "Train",
        "val": "Val",
        "test": "Test",
    }

    def __init__(self, cfg, mode, num_retries=10):
        assert mode in ["train", "val", "test"], (
            "Split '{}' not supported for MECCANO".format(mode)
        )
        self.mode = mode
        self.cfg = cfg
        self._video_meta = {}
        self._num_retries = num_retries
        self._data_root = os.path.abspath(self.cfg.DATA.PATH_TO_DATA_DIR)
        self._frames_root = os.path.join(self._data_root, "RGB_frames")
        self._split_file = os.path.join(self._data_root, self.SPLIT_FILES[self.mode])
        self._split_frames_root = os.path.join(
            self._frames_root, self.FRAME_DIRS[self.mode]
        )

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        else:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing MECCANO %s...", mode)
        self._construct_loader()

    def _construct_loader(self):
        path_to_file = self._split_file
        assert PathManager.exists(path_to_file), "{} not found".format(path_to_file)
        assert os.path.isdir(self._split_frames_root), (
            "{} not found".format(self._split_frames_root)
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_start = []
        self._frame_end = []

        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                if not path_label.strip():
                    continue
                if clip_idx == 0 and path_label.lower().startswith("video_id,"):
                    continue

                parts = [part.strip() for part in path_label.split(",")]
                assert len(parts) == 5, "Unexpected MECCANO row: {}".format(path_label)
                video_path, action_label, _action_name, frame_start, frame_end = parts

                for idx in range(self._num_clips):
                    self._path_to_videos.append(video_path)
                    self._frame_start.append(frame_start)
                    self._frame_end.append(frame_end)
                    self._labels.append(int(action_label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        assert len(self._path_to_videos) > 0, (
            "Failed to load MECCANO split {} from {}".format(self.mode, path_to_file)
        )
        self.num_videos = len(self._path_to_videos)
        logger.info(
            "Constructed MECCANO dataloader (size: %s) from %s",
            self.num_videos,
            path_to_file,
        )

    def __getitem__(self, index):
        if self.mode in ["train", "val"]:
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode == "test":
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        frame_start = int(self._frame_start[index][:-4])
        frame_end = int(self._frame_end[index][:-4])

        frames = []
        frame_dir = os.path.join(self._split_frames_root, self._path_to_videos[index])
        for frame_count in range(frame_start, frame_end + 1):
            frame_name = f"{frame_count:05d}.jpg"
            frame_path = os.path.join(frame_dir, frame_name)
            with Image.open(frame_path) as image:
                image = np.array(image.convert("RGB"), copy=True)
            frames.append(torch.from_numpy(image))

        frames = torch.stack(frames)
        frames = sampling.temporal_sampling(
            frames, frame_start, frame_end, self.cfg.DATA.NUM_FRAMES
        )

        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        frames = frames.permute(3, 0, 1, 2)

        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
        )
        frames = utils.pack_pathway_output(self.cfg, frames)

        label = self._labels[index]
        time_idx = np.array([frame_start, frame_end], dtype=np.int64)
        return frames, label, index, time_idx, {}

    def __len__(self):
        return self.num_videos

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
    ):
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            frames, _ = transform.random_crop(frames, crop_size)
            if random_horizontal_flip:
                frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
