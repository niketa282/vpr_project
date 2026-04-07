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

Cosine similarity has range between -1 and 1.
E.g if cosine similarity is -1 two vectors are very different
if 1 then they are identical.

Similarity matrix 'S' is square matrix that telly you
cosine similarity between every single query image and every reference image 
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

    S = ref_norm @ query_norm.T                        # (N_db, N_q)

    # ── Clip to [-1, 1] to correct floating-point drift ──────────────────────
    S = np.clip(S, -1.0, 1.0).astype(np.float32)

    # min = means lowest similarity in whole square matrix
    # max = highest similarity in whole square matrix
    # mean = Average across all 1231×1231 pairs
    print(f"[Similarity] S.shape = {S.shape}  "
          f"min={S.min():.3f}  max={S.max():.3f}  mean={S.mean():.3f}")
    
    return S  # (N_db, N_q) — just the matrix, nothing else

'''
S[i][j], will tell you the cosine similiarty between query image i
 and reference image j. 

To make a place prediction, for query image i, you want to find the most similiar reference image.
This function is j_match = argmax(S[i][:]) // for a particar query image i, search for all [:] reference images
and return the one with highest similarity and put that in variable j_max

So this find_match function will 
tell you that for query image i, the most similar match according
to model is refernce image j_match.
'''

# J-match effectively answers the following question
# For every query it looks at the similarity scores and picks the single
# image index with highest score
def find_match(S: np.ndarray) -> np.ndarray:
    """
    For each query i, find the reference index with highest similarity.

    Parameters
    ----------
    S : similarity matrix of shape (N_db, N_q)

    Returns
    -------
    j_match : shape (N_q,) — best reference index for each query
    """
    j_match = np.argmax(S, axis=0)   # (N_q,)
    return j_match