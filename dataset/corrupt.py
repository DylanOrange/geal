import os
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset
from dataset.data_utils import normalize_point_cloud, CLASSES, AFFORDANCES


class CorruptDataset(Dataset):
    """
    Corrupt Dataset: LASO-C / PIAD-C

    Used for evaluating model robustness under synthetic corruptions
    (e.g., noise, rotation, scaling, occlusion, etc.).

    Each sample contains:
        - corrupted normalized point cloud
        - object class ID
        - ground truth affordance mask
        - corresponding affordance question
        - affordance label ID
    """

    def __init__(self,
                 corrupt_type: str = "scale",
                 level: int = 0,
                 data_root: str = "LASO-C_dataset"):
        """
        Args:
            corrupt_type (str): type of corruption, e.g. 'scale', 'rotate', 'noise'
            level (int): corruption severity level (0–5)
            data_root (str): dataset root path containing 'point/' and 'text/'
        """
        self.corrupt_type = corrupt_type
        self.level = level
        self.data_root = data_root

        # Class and affordance name mappings
        self.class_to_idx = {cls.lower(): i for i, cls in enumerate(CLASSES)}
        self.aff_to_idx = {aff: i for i, aff in enumerate(AFFORDANCES)}

        # Load corrupted point cloud annotations
        file_name = f"{corrupt_type}_{level}.pkl"
        with open(os.path.join(data_root, "point", file_name), "rb") as f:
            self.annotations = pickle.load(f)

        # Load affordance question CSV
        self.questions = pd.read_csv(os.path.join(data_root, "text", "Affordance-Question.csv"))

        print(f"[CorruptDataset] Loaded corruption='{corrupt_type}' level={level} ({len(self.annotations)} samples)")

    # ------------------------------------------------------------------
    def _get_question(self, object_name: str, affordance: str) -> str:
        """
        Retrieve the canonical (non-random) question for a given object-affordance pair.
        Always uses 'Question0' for deterministic evaluation.
        """
        qid = "Question0"
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
        Retrieve one corrupted sample.
        Returns:
            point_input (np.ndarray): (3, N) normalized corrupted point cloud
            class_id (int): object class index
            gt_mask (np.ndarray): original affordance mask
            question (str): affordance question
            affordance_id (int): affordance label index
        """
        data = self.annotations[idx]
        obj_class = data["class"]
        affordance = data["affordance"]
        gt_mask = data["mask"]
        points = data["point"]

        # Normalize corrupted point cloud
        points, _, _ = normalize_point_cloud(points)

        # Retrieve question
        question = self._get_question(obj_class, affordance)

        # Convert to model input format
        point_input = points.T  # (3, N)
        class_id = self.class_to_idx[obj_class.lower()]
        affordance_id = self.aff_to_idx[affordance]

        return point_input, class_id, gt_mask, question, affordance_id

    # ------------------------------------------------------------------
    def __len__(self):
        """Return dataset size."""
        return len(self.annotations)
