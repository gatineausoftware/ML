import numpy as np
import tensorflow as tf
from estimator_models import build_autoencoder, build_classifier, build_regressor
from functools import partial
import math
import os


#even if i specify what columns to write out and don't include index, index is still written out
def cache_df(df, table, config):
    os.makedirs(config['data_cache']+'/' + table, exist_ok=False)
    df.to_csv(config['data_cache']+'/' + table + '/cache-*.csv')
    root_dir = config['data_cache']+'/' + table
    f = os.listdir(root_dir)
    g = [root_dir + '/' + s for s in f]
    #return os.listdir(config['data_cache']+'/' + table)
    return g


def normalizer_fn(x, mean, std):
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





def prep(feature_names, label_col, *features):
    f = dict(zip(feature_names, features))
    label = f[label_col]
    f.pop(label_col)
    #so annoying!  dask insists on writing out id....even though it is otherwise inaccessible
    f.pop('id')
    return (f, label)




def csv_train_input_fn(data, feature_names, label_col, defaults):
    dataset = tf.contrib.data.CsvDataset(data, defaults, header=True)

    dataset = dataset.shuffle(32).repeat().batch(32)
    dataset = dataset.map(partial(prep, feature_names, label_col))
    return dataset

def csv_eval_input_fn(data, feature_names, label_col, defaults, mode):
    dataset = tf.contrib.data.CsvDataset(data, defaults, header=True)

    dataset = dataset.batch(32)
    dataset = dataset.map(partial(prep, feature_names, label_col))
    return dataset



def build_specs(model, data, feature_names, label_col, metadata, config, defaults):
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
        partial(csv_train_input_fn, data, feature_names, label_col, defaults), max_steps=max_steps, hooks=[hook])

    #does lambda do ths same thing as partial?
    eval_spec = tf.estimator.EvalSpec(
        lambda: partial(csv_eval_input_fn, data, feature_names, label_col, defaults)('eval'),
        steps=None,
        start_delay_secs=0,
        throttle_secs=0
    )

    return train_spec, eval_spec




def train_and_evaluate(model, data, feature_names,label_col, metadata, config, defaults):
        '''
        model has already been created
        '''
        # FIXME: Should not be needed!
        #os.makedirs(self.model.eval_dir())

        # Train and evaluate Estimator
        # problem with getting feature names from dask df
        feature_names = ['id'] + feature_names
        train_spec, eval_spec = build_specs(model, data, feature_names, label_col, metadata, config, defaults)
        tf.estimator.train_and_evaluate(
            model, train_spec, eval_spec)



def cat_float_to_str(df, metadata):
    """Convert categorical floats to strings.
    Args:
        None.
    Returns:
        None.
    """

    # Convert categorical float features to str
    for _, feature in enumerate(df.columns):
        dtype = df[feature].dtype
        ds_type = metadata[feature]['ds_type']

        if dtype == float and (ds_type == 'cat_onehot' or ds_type == 'cat_embed'):

            # Cast metadata vocab as str
            vocabulary_list = metadata[feature]['vocab']
            metadata[feature]['vocab'] = [str(item)
                for item in vocabulary_list]

            metadata[feature]['mode'] = str(metadata[feature]['mode'])

            # Cast dataframe columns as str
            df[feature] = df[feature].astype(str)


def get_defaults(df, metadata):
    d = [[0]]  #the 0 is for the id!!! painful
    for _, feature in enumerate(df.columns):
       a = []
       a.append(metadata[feature]['mode'])
       d.append(a)

    return d
