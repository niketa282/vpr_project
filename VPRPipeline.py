
from vpr.dataset     import read_images
from vpr.model       import get_model_resnet50
from vpr.descriptor  import compute_descriptor
from vpr.similarity  import compute_similarity_matrix, save_similarity_matrix
#from vpr.matching    import match
#from vpr.metrics     import compute_precision_recall
#from vpr.evaluation  import evaluate, print_metrics


def main():
    query_images, ref_images = read_images()
    print(f"Number of query images : {len(query_images)}") # Expect 1231
    print(f"Number of ref images   : {len(ref_images)}") # Expect 1231

    # Load the pretrained ResNet50 model and its expected image transform
    model, transform = get_model_resnet50()

    # Pass raw image paths into compute_descriptor, which loads each image,
    # runs it through the model, and returns L2-normalized feature embeddings
    # that numerically represent the content of each image
    query_descriptors, ref_descriptors = compute_descriptor(query_images, ref_images, model, transform)
    print(f'query_descriptors.shape = {query_descriptors.shape}')  # expect (N_q, 2048)
    print(f'ref_descriptors.shape   = {ref_descriptors.shape}')    # expect (N_db, 2048)

# Sanity check: N_q and N_db should match your input counts
    assert query_descriptors.shape[0] == len(query_images), "query count mismatch — unpacking may be swapped"
    assert ref_descriptors.shape[0]   == len(ref_images),   "ref count mismatch — unpacking may be swapped"
    
    S = compute_similarity_matrix(ref_descriptors, query_descriptors)
    save_similarity_matrix(S, path="vpr_output/similarity_matrix.npy")

    # matches = match(S, top_k=10)
    # precision_at_k, recall_at_k = compute_precision_recall(matches, top_k=10)
    # metrics = evaluate(matches, ground_truth=None, ks=[1, 5, 10])
    # print_metrics(metrics)


if __name__ == "__main__":
    main()
    print('SUCCESS!')