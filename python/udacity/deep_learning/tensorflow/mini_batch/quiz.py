__author__ = 'admin'
#import math
import numpy as np

def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    feature_x, feature_y = np.array(features).shape

    all_batches = []

    for i in range(0,len(features),batch_size):
        #all_batches.append(features[i:i+batch_size])
        #all_batches.append(labels[i:i+batch_size])
        batch = [features[i:i+batch_size],
                 labels[i:i+batch_size]]
        all_batches.append(batch)
    return all_batches


