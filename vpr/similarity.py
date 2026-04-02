import numpy as np
import torch

"""
Build the pairwise cosine similarity matrix between all database
and query descriptors.
Args:
    ref_descriptors:   (N_db, D) float32 — L2-normalised DB descriptors
    query_descriptors: (N_q,  D) float32 — L2-normalised query descriptors
Returns:
    S: (N_db, N_q) float32
       S[i, j] = cosine similarity between ref image i and query image j
       Values are in [-1.0, 1.0]:
           +1.0  →  identical descriptors (perfect match)
            0.0  →  orthogonal (unrelated)
           -1.0  →  opposite   (very different)
"""
def compute_similarity_matrix(
    ref_descriptors:   np.ndarray,     # (N_db, 2048)  — from ResNet50
    query_descriptors: np.ndarray,     # (N_q,  2048)
) -> np.ndarray:
    assert ref_descriptors.ndim   == 2, "ref_descriptors must be 2-D (N_db, D)"
    assert query_descriptors.ndim == 2, "query_descriptors must be 2-D (N_q, D)"
    assert ref_descriptors.shape[1] == query_descriptors.shape[1], (
        f"Descriptor dimensions must match: "
        f"ref={ref_descriptors.shape[1]} vs query={query_descriptors.shape[1]}" # This assert statement checks 'D' 
                                                                                # value in both is 2048 before computing dot product below. 
                                                                                # Checks the feature descriptors have 2048 
                                                                                # elements as expected with ResNet
    )

    # ── Re-normalise defensively (no-op if already unit vectors) ─────────────
    # ResNet50 outputs 2048-D pool features; compute_descriptor() already
    # L2-normalises, but we guard here in case raw features are passed in.
    ref_norm   = ref_descriptors   / (np.linalg.norm(ref_descriptors,   axis=1, keepdims=True) + 1e-8)
    query_norm = query_descriptors / (np.linalg.norm(query_descriptors, axis=1, keepdims=True) + 1e-8)

    # ── Core operation: one matrix multiply gives ALL pairwise similarities ───
    #    ref_norm   : (N_db, D)
    #    query_norm : (N_q,  D)  →  .T gives (D, N_q)
    #    result S   : (N_db, N_q)
   # The function computes cosine similarity between every query image and 
   # every reference image in one matrix multiply. The output is a 2D grid of similarity scores.
    S = ref_norm @ query_norm.T                        # (N_db, N_q)

    # ── Clip to [-1, 1] to correct floating-point drift ──────────────────────
    S = np.clip(S, -1.0, 1.0).astype(np.float32)

    # min = means lowest similarity in whole square matrix
    # max = highest similarity in whole square matrix
    # mean = Average across all 1231×1231 pairs
    print(f"[Similarity] S.shape = {S.shape}  "
          f"min={S.min():.3f}  max={S.max():.3f}  mean={S.mean():.3f}")

    return S



"""
Persist the similarity matrix to disk so it can be reused
without recomputing descriptors.

Args:
    S:    (N_db, N_q) similarity matrix
    path: output file path (.npy)
"""
def save_similarity_matrix(S: np.ndarray, path: str = "vpr_output/similarity_matrix.npy") -> None:

    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, S)
    print(f"[Similarity] Saved → {path}  ({S.nbytes / 1e6:.1f} MB)")

   
   
"""
Load a previously saved similarity matrix from disk.
Args:
    path: .npy file written by save_similarity_matrix()
Returns:
    S: (N_db, N_q) float32
"""
def load_similarity_matrix(path: str = "vpr_output/similarity_matrix.npy") -> np.ndarray:

    S = np.load(path).astype(np.float32)
    print(f"[Similarity] Loaded {path}  shape={S.shape}")
    return S