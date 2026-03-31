def main():

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Read query and reference images from dataset directory
    #
    # PSEUDOCODE:
    #   db_paths  ← sorted list of image file paths under dataset/database/
    #   q_paths   ← sorted list of image file paths under dataset/queries/
    #   for each path in db_paths + q_paths:
    #       img ← PIL.Image.open(path).convert("RGB")
    #       tensor ← resize → to_tensor → normalise(ImageNet mean/std)
    #   return query_images[ N_q × 3 × H × W ],
    #          ref_images  [ N_db × 3 × H × W ]
    # ─────────────────────────────────────────────────────────────────────
    query_images, ref_images = read_images()


    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Compute a single global descriptor per image
    #         using a pretrained backbone (e.g. DINOv2 ViT-S/14)
    #
    # PSEUDOCODE:
    #   model ← load_pretrained_backbone()           # e.g. DINOv2 / ResNet
    #   for each batch in query_images:
    #       feats ← model.forward(batch)             # (B, D)  CLS token
    #       query_descriptors ← L2_normalise(feats)  # unit vectors for cosine sim
    #   for each batch in ref_images:
    #       feats ← model.forward(batch)
    #       ref_descriptors ← L2_normalise(feats)
    #   return query_descriptors [ N_q  × D ],
    #          ref_descriptors   [ N_db × D ]
    # ─────────────────────────────────────────────────────────────────────
    query_descriptor, ref_descriptors = compute_descriptor(query_images, ref_images)


    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Build the pairwise similarity matrix  S ∈ ℝ^(N_db × N_q)
    #         S[i, j] = cosine_similarity(ref[i], query[j])
    #
    # PSEUDOCODE:
    #   # Because descriptors are L2-normalised, cosine sim == dot product
    #   S ← ref_descriptors @ query_descriptors.T    # (N_db × N_q)
    #   # S[i, j] ∈ [-1, 1]:  +1 = identical,  0 = orthogonal, -1 = opposite
    #   save S to disk as similarity_matrix.npy
    # ─────────────────────────────────────────────────────────────────────
    S = compute_similarity_matrix(ref_descriptors, query_descriptor)


    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Matching — for each query column in S, find the top-K
    #         reference rows with the HIGHEST similarity scores
    #
    # PSEUDOCODE:
    #   for j in range(N_q):
    #       col ← S[:, j]                            # similarity to all DB images
    #       top_k_idx ← argsort(col, descending=True)[:K]
    #       top_k_scores ← col[top_k_idx]
    #       matches[j] ← { query_idx: j,
    #                       db_idx:   top_k_idx,
    #                       scores:   top_k_scores }
    #   # Higher score → more similar → better match candidate
    # ─────────────────────────────────────────────────────────────────────
    matches = match(S, top_k=10)


    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Calculate Precision and Recall at rank K
    #
    # PSEUDOCODE:
    #   for each query j:
    #       retrieved   ← top-K db indices from matches[j]
    #       relevant    ← GT[:, j]                   # ground-truth positives
    #       true_pos    ← count(retrieved ∩ relevant)
    #
    #       Precision@K ← true_pos / K
    #       # "Of the K results I returned, how many were correct?"
    #
    #       Recall@K    ← 1  if true_pos > 0  else  0
    #       # "Did I find at least one correct match in my top-K?"
    #
    #   mean_precision@K ← average Precision@K over all queries
    #   mean_recall@K    ← average Recall@K    over all queries
    #
    #   mAP ← mean over queries of:
    #       AP_j = Σ_i [ P@i × rel(i) ] / n_positives_j
    #       # area under the per-query precision-recall curve
    # ─────────────────────────────────────────────────────────────────────
    precision_at_k, recall_at_k = compute_precision_recall(matches, top_k=10)


    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Evaluation using ground truth
    #
    # PSEUDOCODE:
    #   GT ← load ground_truth.npy                  # bool (N_db × N_q)
    #        OR generate synthetic GT:
    #            for j in range(N_q):
    #                db_equiv ← j * N_db // N_q     # positional proxy
    #                GT[ db_equiv ± radius, j ] ← True
    #
    #   metrics ← evaluate(matches, GT, ks=[1, 5, 10])
    #   print  Recall@1, Recall@5, Recall@10
    #   print  Precision@1, Precision@5, Precision@10
    #   print  mAP
    #   plot   similarity_matrix heatmap  →  similarity_matrix.png
    #   plot   precision-recall curve     →  precision_recall.png
    #   plot   top-K visual grid          →  top_matches.png
    #                                        (green border = correct match)
    #                                        (red   border = wrong  match)
    # ─────────────────────────────────────────────────────────────────────
    metrics = evaluate(matches, ground_truth=None, ks=[1, 5, 10])
    print_metrics(metrics)


if __name__ == "__main__":
    # This condition is called when you execute
    # 'python feature_descriptor_encoding.py' from the Linux terminal.
    # It prevents main() from running if this file is imported as a module
    # elsewhere (e.g. during unit testing or from another script).
    main()
    print('SUCCESS!')