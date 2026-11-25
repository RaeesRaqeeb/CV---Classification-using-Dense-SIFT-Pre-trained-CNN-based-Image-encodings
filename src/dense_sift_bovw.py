from typing import List, Optional
import cv2
import numpy as np


class DenseSIFTBoVW:
    def __init__(
        self,
        vocab_size: int = 100,
        step_size: int = 8,
        scales: Optional[List[int]] = None,
        max_train_descriptors: int = 200_000,
        random_state: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.step_size = step_size
        self.scales = scales if scales is not None else [4, 8, 12]
        self.max_train_descriptors = max_train_descriptors
        self.random_state = random_state
        self.sift = cv2.SIFT_create()
        self.kmeans = None

    def extract_dense_descriptors(self, image_path: str) -> np.ndarray:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to read image: {image_path}")

        h, w = gray.shape
        all_descriptors: List[np.ndarray] = []

        for scale in self.scales:
            keypoints = [
                cv2.KeyPoint(float(x), float(y), float(scale))
                for y in range(self.step_size // 2, h, self.step_size)
                for x in range(self.step_size // 2, w, self.step_size)
            ]
            if not keypoints:
                continue
            _, descriptors = self.sift.compute(gray, keypoints)
            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors.astype(np.float32))

        if not all_descriptors:
            return np.zeros((0, 128), dtype=np.float32)

        return np.vstack(all_descriptors)