import logging
from .common.conversions import str_to_coords
import numpy as np


class CoordsSequenceSimilarity:

    @staticmethod
    def compute_score(pred_keypoints, gt_keypoints, k=10):

        try:
            pred_keypoints = np.array(pred_keypoints)
        except ValueError:

            return 0

        gt_keypoints = np.array(gt_keypoints)


        if pred_keypoints.shape != gt_keypoints.shape:
            return 0


        distances = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)


        max_distance = np.sqrt(2)


        normalized_distances = distances / max_distance


        per_keypoint_scores = np.exp(-k * normalized_distances)


        score = np.mean(per_keypoint_scores)

        return score

    @classmethod
    def match(cls, responses, targets) -> float:
        logging.debug(f"{responses=}, {targets=}")
        if not isinstance(responses, (tuple | list)):
            responses = str_to_coords(responses, dim=2)
        if not isinstance(targets, (tuple | list)):
            targets = str_to_coords(targets, dim=2)

        return cls.compute_score(responses, targets)
