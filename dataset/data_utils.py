import numpy as np
# ------------------------------
# Global Constants
# ------------------------------

CLASSES = ["Bag", "Bed", "Bowl","Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
    "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
    "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle"]

AFFORDANCES = ['lay','sit','support','grasp','lift','contain','open','wrap_grasp','pour', 
                'move','display','push','pull','listen','wear','press','cut','stab']

VIEWPOINTS = [
    "from the rear view", "from the left-rear view", "from the left side view", 
    "from the left-front view", "from the front view", "from the right-front view", 
    "from the right side view", "from the right-rear view", "from the bottom view", 
    "from the lower diagonal view", "from the upper diagonal view", "from the top view"
]

# Used to filter out specific affordance-object pairs in unseen split
UNSEEN_REMOVE = {
    'contain': ['microwave', 'vase', 'mug', 'trashcan'],
    'cut': ['scissors'],
    'display': ['display', 'laptop'],
    'grasp': ['mug', 'scissors', 'earphone', 'hat'],
    'move': ['table'],
    'open': ['microwave', 'trashcan', 'door', 'bottle'],
    'pour': ['mug', 'trashcan', 'vase'],
    'press': ['keyboard'],
    'stab': ['scissors'],
    'support': ['chair', 'bed'],
    'wrap_grasp': ['vase', 'mug']
}

# ------------------------------
# Utility: Normalize Point Cloud
# ------------------------------

def normalize_point_cloud(pc: np.ndarray):
    """
    Normalize a 3D point cloud to fit inside a unit sphere.

    Args:
        pc (np.ndarray): (N, 3) array of points

    Returns:
        normalized_pc (np.ndarray): scaled to 0.5 radius
        centroid (np.ndarray): center of point cloud
        scale (float): normalization scale factor
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = (pc / scale) * 0.5
    return pc, centroid, scale