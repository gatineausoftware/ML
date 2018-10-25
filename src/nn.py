import numpy as np
import tensorflow as tf
from estimator_models import build_autoencoder, build_classifier, build_regressor
from functools import partial
import math

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
    for colpos, feature in enumerate(df.columns):

        if feature == label_col:
            continue

        elif metadata[feature]['ds_type'] == 'exclude':
            continue

        elif metadata[feature]['ds_type'] == 'float':

            # Retrieve mean and std for feature
            mean = np.array(metadata[feature]['mean'])
            std = np.array(metadata[feature]['stddev'])

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


def build_estimator(metadata, batch_size, optimizer, project_dir, model_version, feature_columns, label_col, num_layers, num_nodes, dropout):
    """Build TF Estimator for problem type.
    Args:
        num_layers: Number of hidden layers in DNN.
        num_nodes: Number of nodes per hidden layer.
        dropout: Dropout percentage per hidden layer.
    Returns:
        None.
    """

    # Retrieve from metadata parameters to contruct Estimator
    hidden_units = [num_nodes for i in range(num_layers)]
    activation_name = metadata['_ml']['ffnn']['activation_functions'][0]
    steps_per_epoch = math.ceil(
        metadata['_global']['num_rows'] / batch_size)
    patience = metadata['_ml']['ffnn']['patience'][0]

    # Construct Estimator
    if metadata['_ml']['problem_type'] == 'classification':
        n_classes = metadata[label_col]['num_levels']
        model = build_classifier(activation_name, steps_per_epoch, patience, feature_columns,
                                      hidden_units, n_classes, dropout,
                                      optimizer, project_dir, model_version)

    elif metadata['_ml']['problem_type'] == 'regression':
        model = build_regressor(activation_name, steps_per_epoch, patience, feature_columns,
                                     hidden_units, dropout,
                                     optimizer, project_dir, model_version)
    return model



def train_input_fn(training_df, metadata, label_col, config):
    """Training input function for Estimator.
    Args:
        None.
    Returns:
        Dataset: TF Dataset for training.
    """

    # Retrieve appropriate shuffle buffer size (dataset length)
    buffer_size = metadata['_global']['num_rows']

    # Retrieve feature dictionary
    feature_dict = dict(
        training_df.loc[:, training_df.columns != label_col]
    )

    # Retrieve labels
    if label_col == None:
        tensors = feature_dict
    else:
        labels = training_df.loc[:, training_df.columns == label_col]
        tensors = (feature_dict, labels)

    # Create TF Dataset for training
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.shuffle(buffer_size).repeat().batch(config['batch_size'])

    return dataset

def eval_input_fn(validation_df, metadata, label_col, config, mode):
    """Evaluation input function for Estimator.
    Args:
        mode: Available modes are 'eval' or 'test'.
    Returns:
        Dataset: TF Dataset for evaluation.
    """

    # Handle evaluation or testing input functions
    if mode == 'eval':
        df = validation_df
    else:
        df = validation_df

    # Retrieve feature dictionary
    feature_dict = dict(
        df.loc[:, df.columns != label_col]
    )

    # Retrieve labels
    if label_col == None:
        tensors = feature_dict
    else:
        labels = df.loc[:, df.columns == label_col]
        tensors = (feature_dict, labels)

    # Create TF Dataset for evaluation
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    dataset = dataset.batch(config['batch_size'])

    return dataset

def build_specs(model, df, label_col, metadata, config):
    """Build training and evaluation specs.
    Args:
        None.
    Returns:
        None.
    """

    # Create early stopping hook for problem type
    patience = metadata['_ml']['ffnn']['patience'][0]
    steps_per_epoch = math.ceil(metadata['_global']['num_rows'] / config['batch_size'])
    max_steps_without = steps_per_epoch * patience

    if metadata['_ml']['problem_type'] == 'classification':
        hook = tf.contrib.estimator.stop_if_no_increase_hook(
            model,
            'accuracy',
            max_steps_without,
            run_every_secs=None,
            run_every_steps=steps_per_epoch
        )

    else:  # regression or autoencoder
        hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            model,
            'loss',
            max_steps_without,
            run_every_secs=None,
            run_every_steps=steps_per_epoch
        )

    # Compute number of maximum steps for training
    max_epochs = metadata['_ml']['ffnn']['max_epochs'][0]
    max_steps = max_epochs * steps_per_epoch


    # Create TrainSpec and EvalSpec
    train_spec = tf.estimator.TrainSpec(
        partial(train_input_fn, df, metadata, label_col, config), max_steps=max_steps, hooks=[hook])

    eval_spec = tf.estimator.EvalSpec(
        lambda: partial(eval_input_fn, df, metadata, label_col, config),
        steps=None,
        start_delay_secs=0,
        throttle_secs=0
    )

    return train_spec, eval_spec




def train_and_evaluate(model, df, label_col, metadata, config):
        '''
        model has already been created
        '''
        # FIXME: Should not be needed!
        #os.makedirs(self.model.eval_dir())

        # Train and evaluate Estimator
        train_spec, eval_spec = build_specs(model, df, label_col, metadata, config)
        tf.estimator.train_and_evaluate(
            model, train_spec, eval_spec)



