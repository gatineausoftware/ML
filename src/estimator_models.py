import tensorflow as tf
import os
from tensorflow.python.feature_column.feature_column import _NumericColumn
from tensorflow.python.feature_column.feature_column import _IndicatorColumn


def modelfnDNNAutoencoder(features, labels, mode, params):
    """An autoencoder for TensorFlow DNN models.
    Args:
        features: This is batch_features from input_fn.
        labels: This is batch_labels from input_fn.
        mode: An instance of tf.estimator.ModeKeys.
        params: Additional configuration.
    Returns:
        EstimatorSpec: Fully defines the model to be run by an Estimator.
    """

    # Feature column shape indices
    KEY, VOCAB_LIST = 0, 1

    # Available optimizers
    OPTIMIZERS = {'Adam': tf.train.AdamOptimizer}

    # Use feature columns to preprocess raw inputs
    temp = []
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    temp.append(net)

    # Construct encoder portion of network
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units, activation=params['activation_fn'])
        temp.append(net)
        net = tf.layers.dropout(
            net,
            rate=params['dropout'],
            training=(mode == tf.estimator.ModeKeys.TRAIN)
        )
        temp.append(net)

    # Construct bottleneck hidden layer
    bottleneck = tf.layers.dense(
        net, params['bottleneck'], activation=params['activation_fn']
    )
    temp.append(bottleneck)
    net = tf.layers.dropout(
        bottleneck,
        rate=params['dropout'],
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )
    temp.append(net)

    # Construct decoder portion of network
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units, activation=params['activation_fn'])
        temp.append(net)
        net = tf.layers.dropout(
            net,
            rate=params['dropout'],
            training=(mode == tf.estimator.ModeKeys.TRAIN)
        )
        temp.append(net)

    # Construct output portion of network with logits and predictions
    logits = {}
    predictions = {}
    for feature_column in params['feature_columns']:

        if type(feature_column) == _NumericColumn:
            feature = feature_column[KEY]
            logits[feature] = tf.layers.dense(net, 1, activation=None)
            predictions[feature] = logits[feature]

        elif type(feature_column) == _IndicatorColumn:
            feature = feature_column[0][KEY]
            logits[feature] = tf.layers.dense(
                net, len(feature_column[0][VOCAB_LIST]), activation=None
            )
            predictions[feature] = tf.argmax(logits[feature], 1)
            predictions[feature] = predictions[feature][:, tf.newaxis]

    # Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions['bottleneck'] = bottleneck
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute total_loss by summing the loss for every feature reconstruction
    total_loss = 0
    for feature_column in params['feature_columns']:

        if type(feature_column) == _NumericColumn:
            feature = feature_column[KEY]
            labels = tf.feature_column.input_layer(features, feature_column)
            current_loss = tf.losses.mean_squared_error(
                labels, logits[feature]
            )
            total_loss += current_loss

        elif type(feature_column) == _IndicatorColumn:
            feature = feature_column[0][KEY]
            labels = tf.feature_column.input_layer(features, feature_column)
            labels = tf.argmax(labels, 1)[:, tf.newaxis]
            current_loss = tf.losses.sparse_softmax_cross_entropy(
                labels, logits[feature]
            )
            total_loss += current_loss



    # Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss)

    # Training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = OPTIMIZERS[params['optimizer']]()
        train_op = optimizer.minimize(
            total_loss, global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss, train_op=train_op
        )



EXTRA_CHECKPOINTS = 2
ACTIVATION_FN = {'relu': tf.nn.relu, 'selu': tf.nn.selu}

def build_autoencoder(activation_name, steps_per_epoch, patience, feature_columns, hidden_units, bottleneck, dropout,
                      optimizer, model_dir, model_version):


    config = tf.estimator.RunConfig(
        save_checkpoints_steps=steps_per_epoch,
        keep_checkpoint_max=patience + EXTRA_CHECKPOINTS)
    model = tf.estimator.Estimator(
        model_fn=modelfnDNNAutoencoder,
        model_dir=os.path.join(model_dir, str(model_version)),
        config=config,
        params={
            'hidden_units': hidden_units,
            'feature_columns': feature_columns,
            'bottleneck': bottleneck,
            'optimizer': optimizer,
            'activation_fn': ACTIVATION_FN[activation_name],
            'dropout': dropout
        })
    return model


def build_classifier(activation_name, steps_per_epoch, patience, feature_columns, hidden_units, n_classes, dropout,
                      optimizer, model_dir, model_version):
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=steps_per_epoch,
        keep_checkpoint_max=patience + EXTRA_CHECKPOINTS)

    model = tf.estimator.DNNClassifier(
        hidden_units,
        feature_columns,
        model_dir=os.path.join(model_dir, str(model_version)),
        n_classes=n_classes,
        optimizer=optimizer,
        activation_fn=ACTIVATION_FN[activation_name],
        dropout=dropout,
        config=config
    )

    return model

def build_regressor(activation_name, steps_per_epoch, patience, feature_columns, hidden_units, dropout,
                      optimizer, model_dir, model_version):

    config = tf.estimator.RunConfig(
        save_checkpoints_steps=steps_per_epoch,
        keep_checkpoint_max=patience + EXTRA_CHECKPOINTS)

    model = tf.estimator.DNNRegressor(
        hidden_units,
        feature_columns,
        model_dir=os.path.join(model_dir, str(model_version)),
        optimizer=optimizer,
        activation_fn=ACTIVATION_FN[activation_name],
        dropout=dropout,
        config=config)

    return model