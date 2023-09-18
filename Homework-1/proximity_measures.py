import numpy as np


def euclidean_distance(v1, v2):
    """
    Euclidean Distance - L2 Norm

    Args:
        v1: The first vector
        v2: The second vector

    Returns:
        Euclidean distance between v1 and v2

    """
    difference = v1 - v2

    difference_squared = difference * difference

    summation = np.sum(difference_squared)

    euclidean_distance = np.sqrt(summation)

    return euclidean_distance

    # return np.linalg.norm(v1 - v2)


def cosine_similarity(v1, v2):
    """
    Cosine Similarity

    Args:
        v1: The first vector.
        v2: The second vector.

    Returns:
        Cosine similarity of v1 and v2.
    """
    v1_length = np.linalg.norm(v1)
    v2_length = np.linalg.norm(v2)

    v1_dot_v2 = np.dot(v1, v2)

    cosine_similarity = v1_dot_v2 / (v1_length * v2_length)

    return cosine_similarity


def correlation(v1, v2):
    """
    Pearson's correlation

    Args:
        v1: The first vector
        v2: The second vector

    Returns:
        Correlation between v1 and v2
    """
    
    return np.corrcoef(v1, v2)[0, 1]


if __name__ == "__main__":
    """
    Expected Output:
    
    Euclidean Distance between two vectors:  17.05872210923198
    Cosine similarity of two document vectors:  0.314970394174356
    Pearson's correlation between two vectors:  0.4238058708549457
    """
    np.random.seed(1)

    v1 = np.random.randint(0, 20, 10)
    v2 = np.random.randint(0, 20, 10)
    print("Euclidean Distance between two vectors: ", euclidean_distance(v1, v2))

    doc_v1 = np.array([3, 2, 0, 5, 0, 0, 0, 2, 0, 0])
    doc_v2 = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 2])
    print("Cosine similarity of two document vectors: ",
          cosine_similarity(doc_v1, doc_v2))

    print("Pearson's correlation between two vectors: ", correlation(v1, v2))
