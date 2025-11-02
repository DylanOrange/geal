import os
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset

from dataset.data_utils import normalize_point_cloud, CLASSES, AFFORDANCES, VIEWPOINTS, UNSEEN_REMOVE

# ------------------------------
# LASO Dataset Class
# ------------------------------

class LasoDataset(Dataset):
    """
    LASO: Language-driven Affordance Segmentation on Objects.

    Each sample includes:
        - normalized point cloud
        - object class ID
        - binary affordance mask
        - affordance question (prefixed with viewpoint)
        - affordance label ID
        - raw ground truth mask
    """

    def __init__(self, split: str = "train", setting: str = "seen", data_root: str = "LASO_dataset"):
        """
        Args:
            split (str): "train" or "test"
            setting (str): "seen" or "unseen"
            data_root (str): root path to LASO dataset
        """
        self.split = split
        self.data_root = data_root

        # Mapping class/affordance names to IDs
        self.class_to_idx = {cls.lower(): i for i, cls in enumerate(CLASSES)}
        self.aff_to_idx = {aff: i for i, aff in enumerate(AFFORDANCES)}

        # Load annotations and object point clouds
        with open(os.path.join(data_root, f"anno_{split}.pkl"), "rb") as f:
            annotations = pickle.load(f)
        with open(os.path.join(data_root, f"objects_{split}.pkl"), "rb") as f:
            self.objects = pickle.load(f)

        # Filter out unseen affordance-object pairs (for unseen training)
        if split == "train" and setting == "unseen":
            annotations = [
                item for item in annotations
                if not (item["affordance"] in UNSEEN_REMOVE and item["class"] in UNSEEN_REMOVE[item["affordance"]])
            ]

        self.annotations = annotations
        self.questions = pd.read_csv(os.path.join(data_root, "Affordance-Question.csv"))

        print(f"[LASO] Loaded {split} split ({len(self.annotations)} samples, setting={setting})")

    # ------------------------------------------------------------------
    def _sample_question(self, object_name: str, affordance: str) -> str:
        """
        Retrieve one rephrased question for given object-affordance pair.
        Randomly selects from 15 variants during training.
        """
        qid = f"Question{np.random.randint(1, 15)}" if self.split == "train" else "Question0"
        row = self.questions.loc[
            (self.questions["Object"] == object_name) & (self.questions["Affordance"] == affordance),
            [qid]
        ]
        if not row.empty:
            return row.iloc[0][qid]
        raise ValueError(f"No question found for {object_name}-{affordance}")

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """
        Retrieve one dataset sample.
        Returns:
            point_input (np.ndarray): (3, N) normalized point cloud
            class_id (int): object class index
            binary_mask (np.ndarray): binary mask (0/1)
            questions (tuple[str]): 12 rephrased affordance questions
            affordance_id (int): affordance label index
            gt_mask (np.ndarray): original affordance mask
        """
        data = self.annotations[idx]
        shape_id = data["shape_id"]
        obj_class = data["class"]
        affordance = data["affordance"]
        gt_mask = data["mask"]

        # Load and normalize point cloud
        points = self.objects[str(shape_id)]
        points, _, _ = normalize_point_cloud(points)

        # Build binary mask (0 or 1)
        binary_mask = (gt_mask > 0).astype(np.uint8)

        # Retrieve affordance question
        question = self._sample_question(obj_class, affordance)

        # Combine with viewpoint prefix
        questions = tuple(
            f"This is a depth map of a {obj_class} viewed {vp}. {question}"
            for vp in VIEWPOINTS
        )

        # Convert to model-ready format
        point_input = points.T  # shape: (3, N)
        class_id = self.class_to_idx[obj_class.lower()]
        affordance_id = self.aff_to_idx[affordance]

        return point_input, class_id, binary_mask, questions, affordance_id, gt_mask

    # ------------------------------------------------------------------
    def __len__(self):
        """Return dataset length."""
        return len(self.annotations)