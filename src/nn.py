import numpy as np
import tensorflow as tf



def normalizer_fn(self, x, mean, std):
    """Z-score feature tensor.
    Args:
        x: TF feature tensor.
        mean: Feature mean.
        std: Feature standard deviation.
    Returns:
        x: Normalized TF feature tensor.
    """

    # Z-score
    x = (x - mean) / std

    return x

def build_feature_columns(df, label_col, metadata):
    """Build TF feature columns using metadata.
    Args:
        None.
    Returns:
        None.
    """

    # Initialize feature columns
    feature_columns = []

    # Add associated numeric or categorical feature column
    for feature in df:

        if feature == label_col:
            continue

        elif metadata[feature]['ds_type'] == 'exclude':
            continue

        elif metadata[feature]['ds_type'] == 'float':

            # Retrieve mean and std for feature
            mean = np.array(metadata[feature]['mean'])
            std = np.sqrt(metadata[feature]['variance'])

            # Create numeric feature column
            feature_column = tf.feature_column.numeric_column(
                feature,
                normalizer_fn=lambda x, mean=mean, std=std:
                normalizer_fn(x, mean, std)
            )

        else:  # Categorical

            # Retrieve vocabulary list for feature
            vocabulary_list = metadata[feature]['vocab']

            # Create categorical feature column
            feature_column = \
                tf.feature_column.categorical_column_with_vocabulary_list(
                    feature, vocabulary_list=vocabulary_list)

            if metadata[feature]['ds_type'] == 'cat_onehot':
                feature_column = tf.feature_column.indicator_column(
                    feature_column)

            elif metadata[feature]['ds_type'] == 'cat_embed':
                feature_column = tf.feature_column.embedding_column(
                    feature_column, metadata[feature]['embed_size'])

        feature_columns.append(feature_column)
        return feature_columns

