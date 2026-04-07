import numpy as np
import torch
from glob import glob
from sklearn.neighbors import BallTree

def get_gt_labels(database_folder, queries_folder, threshold=25):
   # 1. Gather file paths
    db_paths    = sorted(glob(f"{database_folder}/*.jpg"))
    query_paths = sorted(glob(f"{queries_folder}/*.jpg"))

    # 2. Unrolled loop for Database UTMs
    db_list = []
    for p in db_paths:
        parts = p.split("@")
        # index 1 is Easting, index 2 is Northing
        db_list.append([float(parts[1]), float(parts[2])])
    db_utms = np.array(db_list)

    # 3. Unrolled loop for Query UTMs
    query_list = []
    for p in query_paths:
        parts = p.split("@")
        query_list.append([float(parts[1]), float(parts[2])])
    query_utms = np.array(query_list)

    # 4. Spatial Indexing
    # We use a BallTree to find the nearest neighbor in 2D space (UTM)
    tree = BallTree(db_utms)
    
    # query(..., k=1) returns (distance to nearest, index of nearest)
    distances, gt_labels = tree.query(query_utms, k=1)
    
    # Flatten the array from (N, 1) to (N,) to make it easier to use
    gt_labels = gt_labels.flatten()

    return gt_labels
                       

def evaluate_matches(
    j_match:   np.ndarray,   # (N_q,)
    gt_labels: np.ndarray,   # (N_q,)
    tolerance: int = 0,
) -> dict:
    assert j_match.shape == gt_labels.shape # comparing "j_match" i.e your guesses to answer key i.e gt_labels
                                            # Hence shape of both of these need to be the same

   # Calculates the numerical distance between your guess and the truth. 
   # If GT is 100 and you guessed 105, the error is 5.
    errors  = np.abs(j_match.astype(int) - gt_labels.astype(int))

    # Creates a list of "True" or "False" values. 
    # It's True if the error is within your allowed wiggle room.
    # Tolerance = 0: You must find the exact same frame index. 
    # If the ground truth is index 50 and you pick 51, you are wrong.
    correct = errors <= tolerance
    
    # Since True counts as 1 and False as 0, the average (mean) gives you the accuracy percentage.
    recall_at_1 = correct.mean() * 100.0

    print(f"[Evaluation] N_queries={len(j_match)}  "
          f"tolerance=±{tolerance}  "
          f"Recall@1={recall_at_1:.2f}%")

    return {
        "recall_at_1": recall_at_1,
        "correct":     correct,
        "errors":      errors,
    }