#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage
from _corners import filter_frame_corners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *

import sortednp as snp
import cv2
import pims

proj_err = 4.0
triang_params = TriangulationParameters(proj_err, 3.0, 0.1)

init_pair_threshold = 1000


def init_tracking(
        intrinsic_mat,
        corner_storage,
        rgb_sequence: pims.FramesSequence
):
    best_points_cnt = -1
    best_pair = ((-1, None), (-1, None))
    for frame1 in range(len(rgb_sequence)):
        for frame2 in range(frame1 + 1, min(frame1 + 20, len(rgb_sequence))):
            corrs = build_correspondences(corner_storage[frame1], corner_storage[frame2])
            if len(corrs.ids) < 10:
                continue
            E, mask_em = cv2.findEssentialMat(
                corrs.points_1,
                corrs.points_2,
                intrinsic_mat,
                threshold=2.0,
                method=cv2.RANSAC
            )

            if E is None:
                continue

            corrs_mask = mask_em.astype(np.bool).flatten()
            corrs = Correspondences(
                corrs.ids[corrs_mask],
                corrs.points_1[corrs_mask],
                corrs.points_2[corrs_mask]
            )

            _, mask_hom = cv2.findHomography(
                corrs.points_1,
                corrs.points_2,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0
            )

            if np.count_nonzero(mask_hom) / len(corrs.ids) > 0.5:
                continue

            R1, R2, t = cv2.decomposeEssentialMat(E)

            poses = np.array([np.hstack((R, t)) for R, t in [(R1, t), (R1, -t), (R2, t), (R2, -t)]])

            for view_mat_2 in poses:
                view_mat_1 = eye3x4()

                points3d, ids, median_cos = triangulate_correspondences(
                    corrs,
                    view_mat_1,
                    view_mat_2,
                    intrinsic_mat,
                    triang_params
                )
                # min_cos = min(min_cos, median_cos)
                # max_len_ps = max(max_len_ps, len(points3d))
                cur_pair = (
                        (frame1, view_mat3x4_to_pose(view_mat_1)),
                        (frame2, view_mat3x4_to_pose(view_mat_2))
                )
                if len(points3d) > init_pair_threshold:
                    return cur_pair
                if len(points3d) > best_points_cnt:
                    best_points_cnt = len(points3d)
                    best_pair = cur_pair
    return best_pair


def print_proc_info(cur_frame, frame_cnt, inliers_cnt):
    print('frame:', cur_frame, '/', frame_cnt, ', inliers count: ', inliers_cnt)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = init_tracking(intrinsic_mat, corner_storage, rgb_sequence)
        print('Initial frames:', known_view_1[0], known_view_2[0])

    corrs = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])

    view_mat_1 = pose_to_view_mat3x4(known_view_1[1])
    view_mat_2 = pose_to_view_mat3x4(known_view_2[1])

    points3d, ids, _ = triangulate_correspondences(
        corrs,
        view_mat_1,
        view_mat_2,
        intrinsic_mat,
        triang_params
    )

    frame_count = len(rgb_sequence)

    view_mats = [None] * frame_count
    view_mats[known_view_1[0]] = view_mat_1
    view_mats[known_view_2[0]] = view_mat_2

    point_cloud_builder = PointCloudBuilder(ids, points3d)
    got_updated = True
    while got_updated:
        got_updated = False
        for i, (frame, corners) in enumerate(zip(rgb_sequence, corner_storage)):
            if view_mats[i] is not None:
                continue
            _, indices = snp.intersect(
                point_cloud_builder.ids.flatten(),
                corners.ids.flatten(),
                indices=True
            )
            known_3d = point_cloud_builder.points[indices[0]]
            known_2d = corners.points[indices[1]]
            try:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=known_3d,
                    imagePoints=known_2d,
                    cameraMatrix=intrinsic_mat,
                    distCoeffs=None,
                    reprojectionError=proj_err
                )

                inliers_count = len(inliers)
                if inliers_count > 0:
                    view_mats[i] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                    got_updated = True
                print_proc_info(i, frame_count, inliers_count)
            except Exception:
                print_proc_info(i, frame_count, 0)
            if view_mats[i] is None:
                continue

            for j in range(frame_count):
                if view_mats[j] is None:
                    continue
                corrs = build_correspondences(corner_storage[j], corner_storage[i])
                if len(corrs.ids) == 0:
                    continue
                points3d, ids, _ = triangulate_correspondences(
                    corrs,
                    view_mats[j],
                    view_mats[i],
                    intrinsic_mat,
                    triang_params
                )
                point_cloud_builder.add_points(ids, points3d)

    view_mats[0] = next((mat for mat in view_mats if mat is not None), None)
    for i in range(1, len(view_mats)):
        if view_mats[i] is None:
            view_mats[i] = view_mats[i - 1]
    view_mats_np = np.array(view_mats)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats_np,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats_np))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
