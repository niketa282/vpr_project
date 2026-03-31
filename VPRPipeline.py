
from vpr.dataset     import read_images
from vpr.descriptor  import compute_descriptor
#from vpr.similarity  import compute_similarity_matrix
#from vpr.matching    import match
#from vpr.metrics     import compute_precision_recall
#from vpr.evaluation  import evaluate, print_metrics


def main():
    query_images, ref_images = read_images()
    # query_descriptor, ref_descriptors = compute_descriptor(query_images, ref_images)
    
    # S = compute_similarity_matrix(ref_descriptors, query_descriptor)
    # matches = match(S, top_k=10)
    # precision_at_k, recall_at_k = compute_precision_recall(matches, top_k=10)
    # metrics = evaluate(matches, ground_truth=None, ks=[1, 5, 10])
    # print_metrics(metrics)


if __name__ == "__main__":
    main()
    print('SUCCESS!')