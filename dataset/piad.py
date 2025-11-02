import os
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset

from dataset.data_utils import normalize_point_cloud, CLASSES, AFFORDANCES, VIEWPOINTS

# ------------------------------
# PIAD Dataset
# ------------------------------

class PiadDataset(Dataset):
    """
    PIAD: Point-based Interactive Affordance Dataset.

    Each sample contains:
        - normalized point cloud (N, 3)
        - object class ID
        - binary affordance mask
        - 12 viewpoint-conditioned affordance questions
        - affordance label ID
        - ground truth affordance mask
    """

    def __init__(self, split: str = "train", setting: str = "seen", data_root: str = "piad_dataset"):
        """
        Args:
            split (str): "train" or "test"
            setting (str): "seen" or "unseen"
            data_root (str): path to PIAD dataset root
        """
        self.split = split
        self.setting = setting
        self.data_root = data_root

        # Build class and affordance name → index mappings
        self.class_to_idx = {cls.lower(): i for i, cls in enumerate(CLASSES)}
        self.aff_to_idx = {aff: i for i, aff in enumerate(AFFORDANCES)}

        # Load annotations (contains point cloud + labels)
        anno_path = os.path.join(data_root, f"{setting}_{split}.pkl")
        with open(anno_path, "rb") as f:
            self.annotations = pickle.load(f)

        # Load affordance rephrasing table
        self.questions = pd.read_csv(os.path.join(data_root, "Affordance-Question.csv"))

        print(f"[PIAD] Loaded {split} split ({len(self.annotations)} samples, setting={setting})")

    # ------------------------------------------------------------------
    def _sample_question(self, object_name: str, affordance: str) -> str:
        """
        Retrieve one random rephrased question for an object-affordance pair.
        Training randomly samples from 15 variants; test uses 'Question0'.

        Args:
            object_name (str): object category name
            affordance (str): affordance type
        Returns:
            str: question text
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
        Retrieve one PIAD sample.
        Returns:
            point_input (np.ndarray): (3, N) normalized point cloud
            class_id (int): object class index
            binary_mask (np.ndarray): binary affordance mask (0/1)
            questions (tuple[str]): 12 rephrased affordance questions
            affordance_id (int): affordance label index
            gt_mask (np.ndarray): original affordance mask
        """
        data = self.annotations[idx]
        obj_class = data["class"]
        affordance = data["affordance"]
        gt_mask = data["mask"]
        points = data["point"]

        # Normalize point cloud
        points, _, _ = normalize_point_cloud(points)

        # Convert mask to binary
        binary_mask = (gt_mask > 0).astype(np.uint8)

        # Retrieve affordance question
        question = self._sample_question(obj_class, affordance)

        # Construct viewpoint-prefixed questions
        questions = tuple(
            f"This is a depth map of a {obj_class} viewed {vp}. {question}"
            for vp in VIEWPOINTS
        )

        # Convert to model input format
        point_input = points.T  # shape: (3, N)
        class_id = self.class_to_idx[obj_class.lower()]
        affordance_id = self.aff_to_idx[affordance]

        return point_input, class_id, binary_mask, questions, affordance_id, gt_mask

    # ------------------------------------------------------------------
    def __len__(self):
        """Return dataset size."""
        return len(self.annotations)