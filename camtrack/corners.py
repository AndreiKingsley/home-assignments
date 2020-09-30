#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


class ShiTomasiParams:

    def __init__(self, max_corners=7000, block_size=7, min_distance=4, quality_level=0.01):
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.max_corners = max_corners


class LucasKanadeParams:

    def __init__(self,
                 win_size=(21, 21),
                 max_level=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                 flags=0
                 ):
        self.criteria = criteria
        self.max_level = max_level
        self.win_size = win_size


class CornerHandler:

    def __init__(self, init_img, shi_tomasi_params=ShiTomasiParams(), lucas_kanade_params=LucasKanadeParams()):
        self.lucas_kanade_params = lucas_kanade_params
        self.shi_tomasi_params = shi_tomasi_params
        self.prev_img = None
        self.cur_img = init_img
        self.img_shape = init_img.shape
        self.corners = np.empty((0, 2))
        self.corner_ids = np.empty((0, 1))
        self.corner_sizes = np.empty((0, 1))

        _, self.frame_pyramid = cv2.buildOpticalFlowPyramid(
            (self.cur_img * 255).astype(np.uint8),
            self.lucas_kanade_params.win_size,
            self.lucas_kanade_params.max_level,
            None,
            False
        )

        self.detect_corners_pyramidal(self.create_filter_mask())

        self.prev_img = self.cur_img
        self.prev_frame_pyramid = self.frame_pyramid

    def get_corners(self) -> FrameCorners:
        return FrameCorners(
            self.corner_ids,
            self.corners,
            self.corner_sizes
        )

    def add_next_img(self, new_img):
        self.prev_img = self.cur_img
        self.cur_img = new_img

        levels_number, self.frame_pyramid = cv2.buildOpticalFlowPyramid(
            (self.cur_img * 255).astype(np.uint8),
            self.lucas_kanade_params.win_size,
            self.lucas_kanade_params.max_level,
            None,
            False
        )

        next_pts, status, err = None, None, None
        for level in range(levels_number, -1, -1):
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_pyramid[level],
                self.frame_pyramid[level],
                self.corners.astype(np.float32) / 2 ** level,
                (next_pts * 2) if next_pts is not None else None,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW if next_pts is not None else 0,
                winSize=self.lucas_kanade_params.win_size,
                maxLevel=self.lucas_kanade_params.max_level,
                criteria=self.lucas_kanade_params.criteria
            )

        mask = ((status == 1) & (err < np.quantile(err, 0.9))).reshape(-1)
        self.corners = next_pts[mask]
        self.corner_sizes = self.corner_sizes[mask]
        self.corner_ids = self.corner_ids[mask]

        self.detect_corners_pyramidal(self.create_filter_mask())

        self.prev_frame_pyramid = self.frame_pyramid
        self.prev_img = self.cur_img

    def create_filter_mask(self):
        mask = np.full(self.img_shape, 255, dtype=np.uint8)
        for (x, y), size in zip(self.corners, self.corner_sizes):
            mask = cv2.circle(mask, (np.round(x).astype(int), np.round(y).astype(int)), int(size), thickness=-1,
                              color=0)
        return mask

    def detect_features_shi_tomasi(self, image, mask):
        features = cv2.goodFeaturesToTrack(
            image=image,
            maxCorners=self.shi_tomasi_params.max_corners,
            qualityLevel=self.shi_tomasi_params.quality_level,
            minDistance=self.shi_tomasi_params.min_distance,
            mask=mask,
            blockSize=self.shi_tomasi_params.block_size,
            useHarrisDetector=False
        )
        if features is None:
            return np.empty((0, 2))
        return features.reshape((-1, 2))

    def detect_corners_pyramidal(self, mask):
        corner_mask = mask.copy()
        corners = np.empty((0, 2))
        corner_sizes = np.empty((0, 1))
        k = 1
        for frame in self.frame_pyramid:
            new_corners = self.detect_features_shi_tomasi(frame, corner_mask)
            corners = np.concatenate((corners, new_corners * k), axis=0)
            corner_sizes = np.concatenate(
                (corner_sizes,
                 np.full((new_corners.shape[0], 1), self.shi_tomasi_params.block_size * k)),
                axis=0
            )
            corner_mask = corner_mask[::2, ::2]
            k *= 2

        if self.corner_ids.shape[0] != 0:
            self.corner_ids = np.concatenate(
                (
                self.corner_ids, np.arange(self.corner_ids[-1], self.corner_ids[-1] + corners.shape[0]).reshape(-1, 1)),
                axis=0
            )
        else:
            self.corner_ids = np.arange(corners.shape[0]).reshape(-1, 1)

        self.corners = np.concatenate((self.corners, corners), axis=0)
        self.corner_sizes = np.concatenate((self.corner_sizes, corner_sizes), axis=0)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    init_img = frame_sequence[0]
    handler = CornerHandler(init_img)
    corners = handler.get_corners()
    builder.set_corners_at_frame(0, corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        handler.add_next_img(image_1)
        corners = handler.get_corners()
        builder.set_corners_at_frame(frame, corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
