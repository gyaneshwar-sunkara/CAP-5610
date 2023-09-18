import numpy as np
from scipy.spatial import distance
from scipy import stats
import pandas as pd


def euclidean_distance_scipy(v1, v2):
    return distance.euclidean(v1, v2)


def cosine_similarity_scipy(v1, v2):
    return 1 - distance.cosine(v1, v2)


def correlation_scipy(v1, v2):
    return stats.pearsonr(v1, v2).statistic


def correlation_pandas(v1, v2):
    df = pd.DataFrame({'v1': v1, 'v2': v2})
    correlation = df.corr(method="pearson")["v1"]["v2"]
    return correlation


if __name__ == "__main__":
    """
    Expected Output:
    
    Eculidean Distance between two vectors using scipy:  17.05872210923198
    Cosine similarity of two document vectors using scipy:  0.314970394174356
    Pearson's correlation between two vectors using scipy:  0.4238058708549456
    Pearson's correlation between two vectors using pandas:  0.4238058708549455
    """
    np.random.seed(1)

    v1 = np.random.randint(0, 20, 10)
    v2 = np.random.randint(0, 20, 10)
    print("Eculidean Distance between two vectors using scipy: ",
          euclidean_distance_scipy(v1, v2))

    doc_v1 = np.array([3, 2, 0, 5, 0, 0, 0, 2, 0, 0])
    doc_v2 = np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 2])
    print("Cosine similarity of two document vectors using scipy: ",
          cosine_similarity_scipy(doc_v1, doc_v2))

    print("Pearson's correlation between two vectors using scipy: ",
          correlation_scipy(v1, v2))
    print("Pearson's correlation between two vectors using pandas: ",
          correlation_pandas(v1, v2))


