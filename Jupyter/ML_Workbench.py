"""NeuralNet Class used by ML_Workbench_demo.ipynb.
"""

# Import statements
import os
import math
import json
import threading
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import metadata_extract
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set TF verbosity level
tf.logging.set_verbosity(tf.logging.ERROR)


class NeuralNet:

    def __init__(self, delimiter=',', na_values='\\N',
                 postalcode_info='postalcode_info.json', test_size=0.2,
                 optimizer='Adam', batch_size=32, patience=3, max_steps=None):
        """Class instance initialization.
        Args:
            delimiter: Delimiter used by dataset.
            na_values: Additional strings to recognize as NaN.
            postalcode_info: Path to postalcode_info json file.
            test_size: Test split between 0.0 and 1.0.
            optimizer: Training optimizer.
            batch_size: Training batch size.
            patience: Early stopping number of epochs patience.
            max_steps: Maximum number of training steps.
        Returns:
            None.
        """

        # Initialize instance variables
        self.delimiter = delimiter
        self.na_values = na_values
        self.postalcode_info = json.load(open(postalcode_info))
        self.test_size = test_size
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.patience = patience
        self.max_steps = max_steps
        self.feature_engineering = {}


    def _split_dataset(self):
        """Split dataset into train/eval/test sets.
        Args:
            None.
        Returns:
            None.
        """

        # Split dataframe into train/eval/test and delete full df
        self._train_df, self._test_df = train_test_split(
            self._df, test_size=self.test_size)
        self._train_df, self._eval_df = train_test_split(
            self._train_df, test_size=self.test_size)
        self._df = None


    def _load_dataset(self):
        """Load dataset as a dataframe.
        Args:
            None.
        Returns:
            None.
        """

        # Initialize dataset, label, and model directory
        self._dataset = self._dataset_widget.value
        self._label_col = self._label_widget.value
        self._model_dir = self._model_dir_widget.value

        # Use pandas to read in dataset
        self._df = pd.read_csv(
            self._dataset,
            delimiter=self.delimiter,
            skipinitialspace=True,
            na_values=self.na_values
        )
        self._original_columns = list(self._df)

        # Convert bool to int
        dtype_conversion = {feature: int for feature in self._df
            if self._df[feature].dtype == 'bool'}
        self._df = self._df.astype(dtype_conversion)

        # Display dataset head (features)
        print('_______________________________________________________________')
        print('SAMPLE OF DATASET FEATURES:')
        display(self._df.loc[:, self._df.columns != self._label_col].head())

        # Display dataset head (labels)
        print('_______________________________________________________________')
        print('SAMPLE OF DATASET LABELS:')
        display(self._df.loc[:, self._df.columns == self._label_col].head())


    def _add_new_feature(self):
        """Add new feature information to feature engineering dictionary.
        Args:
            None.
        Returns:
            None.
        """

        # Initialize name, fn_type, and source
        name = self._name_widget.value
        fn_type = self._fn_type_widget.value
        source = self._source_widget.value

        # Store new feature information
        self.feature_engineering[name] = {'fn_type': fn_type, 'source': source}


    def _input_new_feature_handle_submit(self, sender):
        """Handle input new feature UI button on click.
        Args:
            sender: Widget that sent event.
        Returns:
            None.
        """

        # Add feature engineering request and restart process
        self._add_new_feature()
        self._feature_engineering()


    def _input_new_feature(self):
        """Display widgets to input a new feature.
        Args:
            None.
        Returns:
            None.
        """

        # Constant values
        STYLE = {'description_width': '150px'}
        FN_TYPES = ['datetime', 'postalcode']

        # Initialize input feature widgets
        self._name_widget = widgets.Text(
            placeholder='feature_name', description='Name:', style=STYLE
        )
        self._fn_type_widget = widgets.Dropdown(
            options=FN_TYPES, description='Type:', style=STYLE
        )
        self._source_widget = widgets.Dropdown(
            options=list(self._df.loc[:, self._df.columns != self._label_col]),
            description='Source:', style=STYLE
        )

        # Create submit button
        self._submit_widget = widgets.Button(description = 'Submit')
        self._submit_widget.on_click(self._input_new_feature_handle_submit)

        # Display widgets
        print('_______________________________________________________________')
        print('NEW FEATURE:')
        display(
            self._name_widget,
            self._fn_type_widget,
            self._source_widget,
            self._submit_widget
        )


    def _add_handle_submit(self, sender):
        """Handle add UI button on click.
        Args:
            sender: Widget that sent event.
        Returns:
            None.
        """

        # Let user input new feature information
        self._input_new_feature()


    def _datetime(self, df, name, source):
        """Create various datetime period features from a source datetime.
        Args:
            df: Dataframe being changed by transformation.
            name: Base name of new features.
            source: Source feature being transformed in dataframe.
        Returns:
            None.
        """

        # Create various datetime period features
        datetime = pd.to_datetime(df[source])
        df[name + '_quarter'] = datetime.dt.quarter
        df[name + '_month'] = datetime.dt.month
        #df[name + '_week'] = datetime.dt.week
        #df[name + '_dayofyear'] = datetime.dt.dayofyear
        #df[name + '_weekday'] = datetime.dt.weekday
        #df[name + '_day'] = datetime.dt.day


    def _postalcode(self, df, name, source):
        """Create lat/lon postalcode features from a source postalcode.
        Args:
            df: Dataframe being changed by transformation.
            name: Base name of new features.
            source: Source feature being transformed in dataframe.
        Returns:
            None.
        """

        # Create lat/lon postalcode features
        df[name + '_lat'] = df[source].apply(
            lambda x: self.postalcode_info[str(x)]['lat'])
        df[name + '_lon'] = df[source].apply(
            lambda x: self.postalcode_info[str(x)]['lon'])


    def _build_new_features(self, df):
        """Build requested engineered features.
        Args:
            None.
        Returns:
            None.
        """

        # Constant value
        TYPE_TO_FN = {
            'datetime': self._datetime, 'postalcode': self._postalcode
        }

        # Build every request engineered feature
        for feature in self.feature_engineering:
            fn_type = self.feature_engineering[feature]['fn_type']
            source = self.feature_engineering[feature]['source']
            TYPE_TO_FN[fn_type](df, feature, source)


    def _done_handle_submit(self, sender):
        """Handle done UI button on click.
        Args:
            sender: Widget that sent event.
        Returns:
            None.
        """

        # Perform feature engineering and split dataset
        self._build_new_features(self._df)
        self._split_dataset()

        # Extract metadata and confirm recommendations
        self._metadata = metadata_extract.df_metadata(
            self._train_df, label_col=self._label_col)
        self._confirm_recommendations()


    def _feature_engineering(self):
        """Displays the feature engineering widgets.
        Args:
            None.
        Returns:
            None.
        """

        # Initialize feature engineering widgets
        self._add_widget = widgets.Button(description='Add')
        self._done_widget = widgets.Button(description='Done')

        # Define event handlers
        self._add_widget.on_click(self._add_handle_submit)
        self._done_widget.on_click(self._done_handle_submit)

        # Display widgets
        print('_______________________________________________________________')
        print('FEATURE ENGINEERING:')
        display(self._add_widget, self._done_widget)


    def _update_metadata(self):
        """Update and display metadata with data types.
        Args:
            None.
        Returns:
            None.
        """

        # Constant values
        FLOAT_METADATA = [
            'pd_dtype', 'nan_percentage', 'min', 'max', 'mode', 'load_factor',
            'mean', 'variance', 'median', 'iqr', 'skew', 'kurtosis', 'range',
            'ds_type'
        ]
        CAT_METADATA = [
            'pd_dtype', 'nan_percentage', 'min', 'max', 'mode', 'vocab',
            'num_levels', 'embed_size', 'load_factor', 'ds_type'
        ]

        # Initialize type dictionaries for output
        float_dict = {}
        onehot_dict = {}
        embed_dict = {}

        # Update metadata with problem type
        self._metadata['_ml']['problem_type'] = self._problem_type_widget.value

        # Update metadata with type selections and separate types
        for feature in self._train_df:

            if feature == self._label_col:
                continue

            self._metadata[feature]['ds_type'] = \
                self._recommendation_widgets[feature].value

            if self._metadata[feature]['ds_type'] == 'float':
                float_dict[feature] = self._metadata[feature]

            elif self._metadata[feature]['ds_type'] == 'cat_onehot':
                onehot_dict[feature] = self._metadata[feature]

            elif self._metadata[feature]['ds_type'] == 'cat_embed':
                embed_dict[feature] = self._metadata[feature]

        # Display float features and metadata
        if len(float_dict) > 0:
            print('_______________________________________________________________')
            print('PREPROCESSING METADATA - FLOAT FEATURES:')
            display(pd.DataFrame.from_dict(float_dict, orient='index')[FLOAT_METADATA])

        # Display one-hot features and metadata
        if len(onehot_dict) > 0:
            print('_______________________________________________________________')
            print('PREPROCESSING METADATA - ONE-HOT FEATURES:')
            display(pd.DataFrame.from_dict(onehot_dict, orient='index')[CAT_METADATA])

        # Display embedding features and metadata
        if len(embed_dict) > 0:
            print('_______________________________________________________________')
            print('PREPROCESSING METADATA - EMBEDDING FEATURES:')
            display(pd.DataFrame.from_dict(embed_dict, orient='index')[CAT_METADATA])

        # Display suggested neural network architecture
        print('_______________________________________________________________')
        print('PREPROCESSING METADATA - SUGGESTED NEURAL NETWORK:')
        display(pd.DataFrame.from_dict({'ffnn': self._metadata['_ml']['ffnn']}))


    def _cat_float_to_str(self):
        """Convert categorical floats to strings.
        Args:
            None.
        Returns:
            None.
        """

        # Convert categorical float features to str
        for feature in self._train_df:
            dtype = self._train_df[feature].dtype
            ds_type = self._metadata[feature]['ds_type']

            if dtype == float and (ds_type == 'cat_onehot' or ds_type == 'cat_embed'):

                # Cast metadata vocab as str
                vocabulary_list = self._metadata[feature]['vocab']
                self._metadata[feature]['vocab'] = [str(item)
                    for item in vocabulary_list]

                # Cast dataframe columns as str
                self._train_df[feature] = self._train_df[feature].astype(str)
                self._eval_df[feature] = self._eval_df[feature].astype(str)
                self._test_df[feature] = self._test_df[feature].astype(str)


    def _normalizer_fn(self, x, mean, std):
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


    def _build_feature_columns(self):
        """Build TF feature columns using metadata.
        Args:
            None.
        Returns:
            None.
        """

        # Initialize feature columns
        self._feature_columns = []

        # Add associated numeric or categorical feature column
        for feature in self._train_df:

            if feature == self._label_col:
                continue

            elif self._metadata[feature]['ds_type'] == 'exclude':
                continue

            elif self._metadata[feature]['ds_type'] == 'float':

                # Retrieve mean and std for feature
                mean = np.array(self._metadata[feature]['mean'])
                std = np.sqrt(self._metadata[feature]['variance'])

                # Create numeric feature column
                feature_column = tf.feature_column.numeric_column(
                    feature,
                    normalizer_fn=lambda x, mean=mean, std=std:
                        self._normalizer_fn(x, mean, std)
                )

            else:  # Categorical

                # Retrieve vocabulary list for feature
                vocabulary_list = self._metadata[feature]['vocab']

                # Create categorical feature column
                feature_column = \
                    tf.feature_column.categorical_column_with_vocabulary_list(
                        feature, vocabulary_list=vocabulary_list)

                if self._metadata[feature]['ds_type'] == 'cat_onehot':
                    feature_column = tf.feature_column.indicator_column(
                        feature_column)

                elif self._metadata[feature]['ds_type'] == 'cat_embed':
                    feature_column = tf.feature_column.embedding_column(
                        feature_column, self._metadata[feature]['embed_size'])

            self._feature_columns.append(feature_column)


    def _build_estimator(self):
        """Build TF Estimator for problem type.
        Args:
            None.
        Returns:
            None.
        """

        # Constant values
        EXTRA_CHECKPOINTS = 2
        ACTIVATION_FN = {'relu': tf.nn.relu, 'selu': tf.nn.selu}

        # Retrieve from metadata parameters to contruct Estimator
        hidden_units = [self._metadata['_ml']['ffnn']['num_nodes'][0]
            for i in range(self._metadata['_ml']['ffnn']['num_layers'][0])]
        n_classes = self._metadata[self._label_col]['num_levels']
        activation_name = self._metadata['_ml']['ffnn']['activation_functions'][0]
        dropout = self._metadata['_ml']['ffnn']['dropouts'][0]
        self._steps_per_epoch = math.ceil(
            self._metadata['_global']['num_rows'] / self.batch_size)
        config = tf.estimator.RunConfig(
            save_checkpoints_steps=self._steps_per_epoch,
            keep_checkpoint_max=self.patience + EXTRA_CHECKPOINTS)

        # Construct Estimator
        if self._metadata['_ml']['problem_type'] == 'classification':
            self._model = tf.estimator.DNNClassifier(
                hidden_units,
                self._feature_columns,
                model_dir=self._model_dir,
                n_classes=n_classes,
                optimizer=self.optimizer,
                activation_fn=ACTIVATION_FN[activation_name],
                dropout=dropout,
                config=config)

        elif self._metadata['_ml']['problem_type'] == 'regression':
            self._model = tf.estimator.DNNRegressor(
                hidden_units,
                self._feature_columns,
                model_dir=self._model_dir,
                optimizer=self.optimizer,
                activation_fn=ACTIVATION_FN[activation_name],
                dropout=dropout,
                config=config)


    def _build_specs(self):
        """Build training and evaluation specs.
        Args:
            None.
        Returns:
            None.
        """

        # Create early stopping hook for problem type
        max_steps_without = self._steps_per_epoch * self.patience

        if self._metadata['_ml']['problem_type'] == 'classification':
            hook = tf.contrib.estimator.stop_if_no_increase_hook(
                self._model,
                'accuracy',
                max_steps_without,
                run_every_secs=None,
                run_every_steps=self._steps_per_epoch
            )

        elif self._metadata['_ml']['problem_type'] == 'regression':
            hook = tf.contrib.estimator.stop_if_no_decrease_hook(
                self._model,
                'loss',
                max_steps_without,
                run_every_secs=None,
                run_every_steps=self._steps_per_epoch
            )

        # Create TrainSpec and EvalSpec
        self._train_spec = tf.estimator.TrainSpec(
            self._train_input_fn, max_steps=self.max_steps, hooks=[hook])
        self._eval_spec = tf.estimator.EvalSpec(
            lambda:self._eval_input_fn('eval'),
            steps=None,
            start_delay_secs=0,
            throttle_secs=0
        )


    def _tensorboard(self):
        """Start Tensorboard for monitoring.
        Args:
            None.
        Returns:
            None.
        """

        def tensorboard_subprocess():

            # Constant value
            BASH_BASE = 'tensorboard --logdir=%s'

            # Construct tensorboard bash command and process
            bash_command = BASH_BASE % self._model_dir
            process = subprocess.Popen(
                bash_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Run tensorboard
            stdout, stderr = process.communicate()

        # Run tensorboard subprocess on another thread
        t = threading.Thread(target=tensorboard_subprocess)
        t.start()


    def _train_input_fn(self):
        """Training input function for Estimator.
        Args:
            None.
        Returns:
            Dataset: TF Dataset for training.
        """

        # Retrieve appropriate shuffle buffer size (dataset length)
        buffer_size = self._metadata['_global']['num_rows']

        # Retrieve feature dictionary and labels
        feature_dict = dict(
            self._train_df.loc[:, self._train_df.columns != self._label_col])
        labels = self._train_df.loc[:, self._train_df.columns == self._label_col]

        # Create TF Dataset for training
        dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
        dataset = dataset.shuffle(buffer_size).repeat().batch(self.batch_size)

        return dataset


    def _eval_input_fn(self, mode):
        """Evaluation input function for Estimator.
        Args:
            mode: Available modes are 'eval' or 'test'.
        Returns:
            Dataset: TF Dataset for evaluation.
        """

        # Retrieve feature dictionary and labels
        if mode == 'eval':
            feature_dict = dict(
                self._eval_df.loc[:, self._eval_df.columns != self._label_col])
            labels = self._eval_df.loc[:, self._eval_df.columns == self._label_col]
        elif mode == 'test':
            feature_dict = dict(
                self._test_df.loc[:, self._test_df.columns != self._label_col])
            labels = self._test_df.loc[:, self._test_df.columns == self._label_col]

        # Create TF Dataset for evaluation
        dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
        dataset = dataset.batch(self.batch_size)

        return dataset


    def _get_best_checkpoint(self):
        """Retrieve path to best checkpoint.
        Args:
            None.
        Returns:
            None.
        """

        # Tuple indices
        (TIME, STEP, VALUE) = (0, 1, 2)

        # Load evaluation event
        event = EventAccumulator(self._model.eval_dir()).Reload()

        # Find best metric step
        if self._metadata['_ml']['problem_type'] == 'classification':
            best_step = max(event.Scalars('accuracy'),
                key=lambda x: x[VALUE])[STEP]

        elif self._metadata['_ml']['problem_type'] == 'regression':
            best_step = min(event.Scalars('loss'),
                key=lambda x: x[VALUE])[STEP]

        # Retrieve best model checkpoint
        self._best_checkpoint = os.path.join(
            self._model_dir, 'model.ckpt-%d' % best_step)


    def _export_savedmodel(self):
        """Export the best model checkpoint and necessary metadata.
        Args:
            None.
        Returns:
            None.
        """

        # Constant value
        INDENT = 4
        FEATURE_SPEC = 'feature_spec.json'
        FEATURE_ENGINEERING = 'feature_engineering.json'

        # Initialize feature dictionary and specification
        feature_dict = {}
        feature_spec = []

        # Create mapping from feature name to tensor placeholder
        for feature in self._train_df:

            if feature == self._label_col:
                continue

            elif self._train_df[feature].dtype == int:
                feature_dict[feature] = tf.placeholder(tf.int32, shape=[None, 1])
                feature_spec.append(
                    {'name': feature, 'dtype': 'tf.int32', 'shape': '(None, 1)'})

            elif self._train_df[feature].dtype == float:
                feature_dict[feature] = tf.placeholder(tf.float32, shape=[None, 1])
                feature_spec.append(
                    {'name': feature, 'dtype': 'tf.float32', 'shape': '(None, 1)'})

            elif self._train_df[feature].dtype == object:
                feature_dict[feature] = tf.placeholder(tf.string, shape=[None, 1])
                feature_spec.append(
                    {'name': feature, 'dtype': 'tf.string', 'shape': '(None, 1)'})

        # Export best model checkpoint
        serving_input_receiver_fn = \
            tf.estimator.export.build_raw_serving_input_receiver_fn(feature_dict)
        self._model.export_savedmodel(
            self._model_dir,
            serving_input_receiver_fn,
            checkpoint_path=self._best_checkpoint
        )

        # Export necessary metadata
        with open(os.path.join(self._model_dir, FEATURE_SPEC), 'w') as f:
            json.dump(feature_spec, f, indent=INDENT, sort_keys=True)
        with open(os.path.join(self._model_dir, FEATURE_ENGINEERING), 'w') as f:
            json.dump(self.feature_engineering, f, indent=INDENT, sort_keys=True)


    def _train_and_evaluate(self):
        """Train and evaluate (with early stopping) an Estimator.
        Args:
            None.
        Returns:
            None.
        """

        # Update metadata and dataframes after recommendation confirmation
        self._update_metadata()
        self._cat_float_to_str()

        # Build TF feature columns using metadata
        print('_______________________________________________________________')
        print('NEURAL NETWORK TRAINING:\n')
        print('\tBUILD TENSORFLOW FEATURE COLUMNS')
        self._build_feature_columns()

        # Build TF Estimator for problem type
        print('\tBUILD TENSORFLOW ESTIMATOR\n')
        self._build_estimator()

        # Start tensorboard
        self._tensorboard()

        # FIXME: Should not be needed!
        os.makedirs(self._model.eval_dir())

        # Train and evaluate Estimator
        print('\tBEGIN TRAINING: http://localhost:6006')
        self._build_specs()
        tf.estimator.train_and_evaluate(
            self._model, self._train_spec, self._eval_spec)
        print('\tTRAINING COMPLETED')

        # Export best model checkpoint
        self._get_best_checkpoint()
        self._export_savedmodel()
        print('\n\tMODEL EXPORTED')

        # Evaluate Estimator on test set
        print('_______________________________________________________________')
        print('NEURAL NETWORK TESTING:')
        result = self._model.evaluate(
            lambda:self._eval_input_fn('test'),
            checkpoint_path=self._best_checkpoint)
        display(pd.DataFrame.from_dict({'result': result}))


    def _confirm_recommendations_handle_submit(self, sender):
        """Handle confirm recommendations UI button on click.
        Args:
            sender: Widget that sent event.
        Returns:
            None.
        """

        # Train and evaluate Estimator
        self._train_and_evaluate()


    def _confirm_recommendations(self):
        """Confirm metadata recommendations (allowing for changes).
        Args:
            None.
        Returns:
            None.
        """

        # Constant values
        PROBLEM_TYPE = {
            'categorical_crossentropy': 'classification',
            'binary_crossentropy': 'classification',
            'mse': 'regression'
        }
        FEATURE_TYPE = {
            'float': 'float',
            'cat_onehot': 'cat_onehot',
            'cat_embed': 'cat_embed',
            'cat_longtail': 'cat_embed',
            'datetime': 'exclude'
        }
        STYLE = {'description_width': '175px'}

        # Initialize recommendation widgets
        self._recommendation_widgets = {}

        # Create a widget for every feature
        print('_______________________________________________________________')
        print('PREPROCESSING METADATA:')
        print('\n\tSUGGESTED DATA TYPES:')
        for feature in self._train_df:

            if feature == self._label_col:
                continue

            else:
                self._recommendation_widgets[feature] = widgets.Dropdown(
                    options=set(FEATURE_TYPE.values()),
                    value=FEATURE_TYPE[self._metadata[feature]['ds_type']],
                    description='%s:' % feature,
                    style=STYLE
                )
                display(self._recommendation_widgets[feature])

                # if 'histogram' in self._metadata[feature]:
                #     hist_freqs, hist_bins = self._metadata[feature]['histogram']
                #     hist_freqs, hist_bins = np.array(hist_freqs), np.array(hist_bins)
                #     width = 0.7 * (hist_bins[1] - hist_bins[0])
                #     center = (hist_bins[:-1] + hist_bins[1:]) / 2
                #     plt.figure(figsize=(4,1.5))
                #     plt.bar(center, hist_freqs, align='center', width=width)
                #     plt.title(feature)
                #     plt.show()

        # Create a widget for the problem type
        print('\n\tSUGGESTED PROBLEM TYPE:')
        self._problem_type_widget = widgets.Dropdown(
            options=set(PROBLEM_TYPE.values()),
            value=PROBLEM_TYPE[self._metadata['_ml']['losses'][0]],
            description='Problem Type:',
            style=STYLE
        )

        # Create submit button widget
        self._submit_widget = widgets.Button(description = 'Submit')
        self._submit_widget.on_click(self._confirm_recommendations_handle_submit)

        # Display widgets
        display(self._problem_type_widget, self._submit_widget)


    def _train_handle_submit(self, sender):
        """Handle train UI button on click.
        Args:
            sender: Widget that sent event.
        Returns:
            None.
        """

        # Load dataset and begin feature engineering process
        self._load_dataset()
        self._feature_engineering()


    def train(self):
        """Displays the training input widgets.
        Args:
            None.
        Returns:
            None.
        """

        # Constant value
        STYLE = {'description_width': '150px'}

        # Initialize training input widgets
        self._dataset_widget = widgets.Text(
            description='Dataset:',
            placeholder='dataset.csv',
            style=STYLE
        )
        self._label_widget = widgets.Text(
            description='Label Name:',
            placeholder='target_feature',
            style=STYLE
        )
        self._model_dir_widget = widgets.Text(
            description='Model Directory:',
            placeholder='./model',
            style=STYLE
        )

        # Create submit button
        self._submit_widget = widgets.Button(description = 'Submit')
        self._submit_widget.on_click(self._train_handle_submit)

        # Display widgets
        display(
            self._dataset_widget,
            self._label_widget,
            self._model_dir_widget,
            self._submit_widget
        )


    def _predict_input_fn(self, sample_df):
        """Prediction input function for Estimator.
        Args:
            sample: Dataframe containing single sample.
        Returns:
            Dataset: TF Dataset for prediction.
        """

        # Construct feature dictionary
        feature_dict = dict(sample_df)

        # Construct single sample Dataset
        dataset = tf.data.Dataset.from_tensor_slices(feature_dict)
        dataset = dataset.batch(1)

        return dataset


    def _predict(self):
        """Prediction with an Estimator.
        Args:
            None.
        Returns:
            None.
        """

        # Initialize sample df
        sample_df = pd.DataFrame(columns=list(self._feature_widgets))

        # Retrieve sample values from widgets
        for feature in self._feature_widgets:

            if self._train_df[feature].dtype == int:
                sample_df[feature] = [int(self._feature_widgets[feature].value)]

            elif self._train_df[feature].dtype == float:
                sample_df[feature] = [float(self._feature_widgets[feature].value)]

            elif self._train_df[feature].dtype == object:
                sample_df[feature] = [self._feature_widgets[feature].value]

        # Perform feature engineering
        self._build_new_features(sample_df)

        # Model prediction
        prediction = self._model.predict(
            lambda: self._predict_input_fn(sample_df),
            checkpoint_path=self._best_checkpoint)
        prediction_dict = next(prediction)

        # Prepare result for output
        if self._metadata['_ml']['problem_type'] == 'classification':
            prediction_dict = {
                'result': {
                    'class': prediction_dict['classes'][0].decode('ascii'),
                    'probability': max(prediction_dict['probabilities'])
                }
            }

        elif self._metadata['_ml']['problem_type'] == 'regression':
            prediction_dict = {
                'result': {
                    'prediction': prediction_dict['predictions'][0]
                }
            }

        # Display model prediction
        print('_______________________________________________________________')
        print('PREDICTION:')
        display(pd.DataFrame.from_dict(prediction_dict))


    def _predict_handle_submit(self, sender):
        """Handle predict UI button on click.
        Args:
            sender: Widget that sent event.
        Returns:
            None.
        """

        # Use model to predict on a single sample
        self._predict()


    def predict(self):
        """Displays the prediction input widgets.
        Args:
            None.
        Returns:
            None.
        """

        # Constant value
        STYLE = {'description_width': '150px'}

        # Initialize prediction input widgets
        self._feature_widgets = {}

        # Create a widget for every feature
        for feature in self._original_columns:

            if feature == self._label_col:
                continue

            else:
                self._feature_widgets[feature] = widgets.Text(
                    description='%s:' % feature,
                    value=str(self._metadata[feature]['mode']),
                    style=STYLE
                )
                display(self._feature_widgets[feature])

        # Create submit button widget
        self._submit_widget = widgets.Button(description = 'Submit')
        self._submit_widget.on_click(self._predict_handle_submit)

        # Display widgets
        display(self._submit_widget)
