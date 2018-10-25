''' Given a Pandas dataframe, infer as much about the datatype of each column as possible.
'''

from __future__ import print_function
import sys
import argparse
import warnings
import json
import re
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pdb import set_trace as dbg

def df_metadata(df, label_col=None, datetime_regex="^\d{4}-\d{2}-\d{2}", time_series_len=None, existing_md=None) -> dict:
    ''' Given a Pandas dataframe, return a dictionary of lots of metadata about each column. '''

    # Metadata.  If there's an existing metadata dict, use it as a starting point.
    if type(existing_md) == dict:
        md = existing_md
    else:
        md = {}

    import dask.array as da


    # TODO: chi-squared for categ.
    corr = df.corr() # Easiest to run correlation among all features at once.

    # Determine which column (if any) is the label.  If it's not provided, use last column
    if label_col is None:
        label_col = df.columns[-1]
    elif type(label_col) == int: # Is column number (int), rather than name
        label_col = df.columns[label_col]
    elif label_col.isdigit(): # Is column number (string), rather than name
        label_col = df.columns[int(label_col)]

    # Global info
    md['_global'] = {}
    md['_global']['num_features'] = len(df.columns)
    if label_col in df.columns: # we know which column is the ML label
        md['_global']['num_features'] -= 1 # Don't count label column
    md['_global']['pd_dtype_counts'] = dict([(str(dtype), count) for dtype, count in df.dtypes.value_counts().items()])
    if time_series_len:
        md['_global']['is_time_series'] = True
        md['_global']['num_rows'] = len(df) / time_series_len
    else:
        md['_global']['num_rows'] = len(df)
        md['_global']['is_time_series'] = False
    md['_global']['label_col'] = label_col


    for colpos, colname in enumerate(df.columns):
        print(colname)
        df_col = df[colname]

        # Check if it's a pathological column, containing all NaNs.  If so, skip.
        if df_col.isnull().all().compute():
            # Throw error if it's the label column
            if colname == label_col:
                ValueError("Error: label column \"%s\" has all NaNs!  There's probably an error in the column formatting of the input.\n\n" % colname)
            else:
                warnings.warn("\n\nWarning: column \"%s\" has all NaNs!  There's probably an error in the column formatting of the input.\n\n" % colname)
                continue

        if colname not in md:
            md[colname] = {}
        md[colname]['pd_dtype'] = str(df_col.dtype) # Pandas-inferred type

        # Deal with NaN's
        md[colname]['nan_percentage'] = df_col.isnull().values.sum().compute() / md['_global']['num_rows']
        df_col = df_col.dropna()


        value_counts = df_col.value_counts().compute() # cache this for vocab, mode, skew & kurtosis
        md[colname]['mode'] = value_counts.keys()[0]
        md[colname]['vocab'] = sorted(value_counts.keys()) # We'll need this for both numerical and categorical data
        md[colname]['num_levels'] = len(md[colname]['vocab'])
        md[colname]['embed_size'] = max(4, int(round(np.log(md[colname]['num_levels'])))) # Not always applicable, but whatevs
        md[colname]['load_factor'] = md[colname]['num_levels'] / md['_global']['num_rows'] # ratio of number of levels to number of rows
        if md[colname]['load_factor'] > 0.9: # Don't need to store vocab for floats and other dense data like datetimes
            md[colname].pop('vocab', None)


        # Numerical data descriptive stats
        if pd.api.types.is_numeric_dtype(df_col):
            md[colname]['corr'] = dict(corr[colname].compute()) # Pearson correlation coefficient among all features
            md[colname]['corr'].pop(colname) # we don't need self-correlation
            df_col_describe = dict(df_col.describe().compute()) # currently Dask's .describe() only works on numeric data

            md[colname]['min'] = df_col_describe['min']
            md[colname]['max'] = df_col_describe['max']
            md[colname]['mean'] = df_col_describe['mean']
            md[colname]['stddev'] = df_col_describe['std']
            md[colname]['median'] = df_col_describe['50%']
            md[colname]['iqr'] = df_col_describe['75%'] - df_col_describe['25%'] # inter-quantile range. More robust than variance to non-normal data

            md[colname]['skew'] = value_counts.skew()
            md[colname]['kurtosis'] = value_counts.kurtosis()

            # Dask's da.histogram has a different interface than numpy's: the defaults for bins and range are None, so
            # you have to manually supply those.  You also have to compute the first element of the output tuple.
            hist_freqs, hist_ranges = da.histogram(df_col, bins=12, range=[md[colname]['min'], md[colname]['max']])
            md[colname]['histogram'] = (list(hist_freqs.compute()), list(hist_ranges)) # Lists are prolly easier to deal with than np.arrays

            # WTF Pandas??!!  Boolean is not a numeric dtype!
            if df_col.dtype != bool:
                md[colname]['range'] = md[colname]['max'] - md[colname]['min']
                # If all floats end with .0 then it's an int.  Another WTF: True % 1.0 == 0.0
                if da.all(df_col % 1.0 == 0.0).compute():
                    df_col = df_col.astype('int64')
        del value_counts


        # Infer data science datatype
        if 'is_inferred' not in md[colname]  or  md[colname]['is_inferred'] == True: # Don't try to infer if we already told it not to.
            if pd.api.types.is_float_dtype(df_col):
                if md[colname]['num_levels'] <= 10  and  md[colname]['load_factor'] < 0.02:
                    # Not really a float. Treat as onehot
                    md[colname]['ds_type'] = 'cat_onehot'
                    md[colname]['is_inferred'] = True
                else:
                    md[colname]['ds_type'] = 'float'
                    md[colname]['is_inferred'] = True
                    if 'vocab' in md[colname]: # May have been already deleted
                        md[colname].pop('vocab', None)
            # Datetimes.  Should be a string of the pattern "XXXX-XX-XX"
#            elif pd.api.types.is_object_dtype(df_col) and re.match(datetime_regex, df_col.loc[0].compute()[0]):
#                md[colname]['ds_type'] = 'datetime'
#                md[colname]['is_inferred'] = True
            # Maybe some kind of categorical data
            else:
                # Categorical one-hot type
                if md[colname]['num_levels'] < 25:
                    md[colname]['ds_type'] = 'cat_onehot'
                    md[colname]['is_inferred'] = True
                # Represented as ints, but should be treated as a float
                elif pd.api.types.is_integer_dtype(df_col) and is_ordinal(np.array(md[colname]['histogram'][0])):
                    md[colname]['ds_type'] = 'float'
                    md[colname]['is_inferred'] = True
                    md[colname].pop('vocab', None) # None allows the key to not exist, nbd.
                # Long-tail cat-embed type
                elif 'kurtosis' in md[colname] and md[colname]['kurtosis'] > 5:
                    md[colname]['ds_type'] = 'cat_longtail'
                    md[colname]['is_inferred'] = True
                # Nominal, cat-embed type
                else:
                    md[colname]['ds_type'] = 'cat_embed'
                    md[colname]['is_inferred'] = True

    # List suspicous columns -- those that are highly correlated with the label column
    # Correlations only given if label column is numeric.
    if 'corr' in md[label_col]:
        md['_global']['suspicious_cols'] = [(col, md[label_col]['corr'][col]) for col in md[label_col]['corr'] if abs(md[label_col]['corr'][col]) > .9]

    # ML recommendations
    if '_ml' not in md:
        md['_ml'] = {}
    # Is regression task.  Categorical labels could mistakenly be inferred by Pandas as floats
    if md[label_col]['ds_type'] == 'float' and md[label_col]['load_factor'] > 0.2:
        md['_ml']['task'] = 'regression'
        md['_ml']['losses'] = ['mse', 'mae', 'mape', 'cosine', 'logcosh']
        md['_ml']['metrics'] = ['mse', 'mae', 'mape', 'cosine', 'logcosh']
    # Binary classification
    elif md[label_col]['num_levels'] == 2:
        md['_ml']['task'] = 'classification'
        md['_ml']['losses'] = ['binary_crossentropy']
        md['_ml']['metrics'] = ['acc']
    # Multiclass classification
    else:
        md['_ml']['task'] = 'classification'
        md['_ml']['losses'] = ['categorical_crossentropy']
        md['_ml']['metrics'] = ['acc']

    # FFNN
    md['_ml']['ffnn'] = calc_ffnn_arch(num_features=md['_global']['num_features'], num_rows=md['_global']['num_rows'])

    # RNN
    if time_series_len:
        md['_ml']['rnn'] = calc_rnn_arch(num_features=md['_global']['num_features'], num_rows=md['_global']['num_rows'])

    # Boosted Trees
    md['_ml']['lightgbm'] = calc_lgbm_arch(num_features=md['_global']['num_features'], num_rows=md['_global']['num_rows'])

    return md

def is_ordinal(hist_freqs):
    ''' Given histogram frequencies, derived from an array of ints, infers if
        the numeric data is ordinal, rather than nominal.  Using 12 bins in
        the histogram seems to give good results.
        The input for this function is obtained by something like:
        >>> data = np.array([1,2,3,2,8,1,2])
        >>> hist_freqs, _ = np.histogram(data, bins=12)
        >>> is_ordinal(hist_freqs)
        True
    '''

    # Is the sign change high between adjacent values in a sorted array?  If so, could be a candidate as nominal.
    # I haven't seen any research on this area, so this is a new idea AFAIK -- JMD
    signs_of_diffs = np.sign(np.diff(hist_freqs))
    changes_of_signs = np.sum(np.abs(np.diff(signs_of_diffs)))

    if changes_of_signs < len(hist_freqs):
        # Not many changes in sign, so maybe exhibits some statistical pattern
        return True
    else:
        # Many changes in sign, so appears to jump around between sorted values
        return False


def calc_ffnn_arch(num_features=None, num_rows=None, num_nodes_min=20, node_multiplier=1.8):
    ''' Calculate size of feed-forward neural network, given the number of input features and
        the number of rows.  This is done using basic heuristics. YMMV!
        node_multiplier is how much to increase for grid search. For
        decreased values, the inverse is derived from this.
    '''
    md_nn = {}
    md_nn['activation_functions'] = ['relu', 'selu']
    md_nn['max_epochs'] = [15]
    md_nn['patience'] = [6]

    # Mostly based on number of features, lesser extent on number of rows. Use at least 40 nodes, as Ben requested.
    num_nodes = max(num_nodes_min, int(round(np.sqrt((num_features / 2) * np.log(num_rows)))))
    # Generate higher and possibly lower values to look at as well.
    if num_nodes < num_nodes_min * 1.25: # too few to go lower: When they go low, you go high! --M.O.
        md_nn['num_nodes'] = list(np.round([num_nodes, num_nodes * .8*node_multiplier, num_nodes * node_multiplier]).astype(int))
    else:
        md_nn['num_nodes'] = list(np.round([num_nodes, num_nodes / node_multiplier, num_nodes * node_multiplier]).astype(int))

    # Number of hidden layers.
    # Mostly based on number of rows, lesser extent to number of features.  Then use log base 1000 for
    #  very slow growth of number of layers.  Then correct a little by subtracting by 0.25 .
    num_layers = max(0, int(np.round((np.log2(num_rows * np.sqrt(num_features)) / np.log2(1000)) - 0.25)))
    if num_layers < 1:
        md_nn['num_layers'] = [num_layers, num_layers+1]
    else:
        md_nn['num_layers'] = [num_layers, num_layers-1, num_layers+1]

    # Basic heuristic, based on number of layers (which is based on datasize). Change to something more
    #  sensible if you can think of one.
    if md_nn['num_layers'][0] < 2:
        md_nn['dropouts'] = [0.3, 0.15, 0.45]
    elif md_nn['num_layers'][0] == 2:
        md_nn['dropouts'] = [0.1, 0.25, 0.0]
    else:
        md_nn['dropouts'] = [0.0, 0.1, 0.2]

    return md_nn

def calc_rnn_arch(num_features=None, num_rows=None, num_nodes_min=20, node_multiplier=1.8):
    ''' Calculate size of recurrent neural network, given the number of input features and
        the number of rows.  This is done using basic heuristics. YMMV!
        node_multiplier is how much to increase for grid search. For
        decreased values, the inverse is derived from this.
    '''
    md_nn = {}
    md_nn['rnn_types'] = ['lstm', 'gru']
    md_nn['activation_functions'] = ['relu', 'selu']
    md_nn['max_epochs'] = [15]
    md_nn['patience'] = [6]


    # Mostly based on number of features, lesser extent on number of rows.
    num_nodes = max(num_nodes_min, int(round(np.sqrt((num_features / 2) * np.log(num_rows)))))
    # Generate higher and possibly lower values to look at as well.
    if num_nodes < num_nodes_min * 1.25: # too few to go lower: When they go low, you go high! --M.O.
        md_nn['num_nodes'] = list(np.round([num_nodes, num_nodes * .8*node_multiplier, num_nodes * node_multiplier]).astype(int))
    else:
        md_nn['num_nodes'] = list(np.round([num_nodes, num_nodes / node_multiplier, num_nodes * node_multiplier]).astype(int))

    # Mostly based on number of rows, lesser extent to number of features.  Then use log base 1000 for
    #  very slow growth of number of layers.  Then correct a little by subtracting by 0.25 .
    num_layers = max(0, int(np.round((np.log2(num_rows * np.sqrt(num_features)) / np.log2(1000)) - 0.25)))
    if num_layers < 1:
        md_nn['num_layers'] = [num_layers, num_layers+1]
    else:
        md_nn['num_layers'] = [num_layers, num_layers-1, num_layers+1]

    # Basic heuristic, based on number of layers (which is based on datasize). Change to something more
    #  sensible if you can think of one.
    if md_nn['num_layers'][0] == 1:
        md_nn['dropouts'] = [0.3, 0.15, 0.45]
    elif md_nn['num_layers'][0] == 2:
        md_nn['dropouts'] = [0.1, 0.25, 0.0]
    else:
        md_nn['dropouts'] = [0.0, 0.1, 0.2]

    return md_nn

def calc_lgbm_arch(num_features=None, num_rows=None, depth_min=5, num_leaves_min=10, multiplier=1.8):
    ''' Calculate hyperparameters for a LightGBM model, given the number of input features and
        the number of rows.  This is done using basic heuristics. YMMV!
        multiplier is how much to increase for grid search. For
        decreased values, the inverse is derived from this.
    '''
    md = {}
    md['learning_rates'] = [.1, .032, .32, .01]
    md['max_epochs'] = [75, 50, 100]
    md['patience'] = [50]

    # Mostly based on number of features, lesser extent on number of rows.
    num_leaves = max(num_leaves_min, int(round(np.sqrt(num_features * np.log2(num_rows)) * 1.5 + 20 )))
    # Generate higher and possibly lower values to look at as well.
    if num_leaves < num_leaves_min * 1.25: # too few to go lower: When they go low, you go high! --M.O.
        md['num_leaves'] = list(np.round([num_leaves, num_leaves * .8*multiplier, num_leaves * multiplier]).astype(int))
    else:
        md['num_leaves'] = list(np.round([num_leaves, num_leaves / multiplier, num_leaves * multiplier]).astype(int))

    # Maximum depth of trees.
    # Mostly based on number of rows, lesser extent to number of features.  Then use log for
    #  slow growth of number of depth.
    depth = max(depth_min, int(np.round(np.log2(num_rows * np.sqrt(num_features)))))
    if depth < depth_min * multiplier:
        md['depths'] = list(np.round([depth, depth * multiplier]).astype(int))
    else:
        md['depths'] = list(np.round([depth, depth / multiplier, depth * multiplier]).astype(int))

    return md

# Fix json.JSONEncoder's shortcoming that it can't handle numpy datatypes.
# From https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
class NPEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def main():
    parser = argparse.ArgumentParser(description='Infer information from a dataframe')

    #parser.add_argument('-', '--', help='', type=, default=)
    parser.add_argument('--csv', help='Specify data CSV file.', type=str, required=True)
    parser.add_argument('--no-header', help='Input CSV does not have a header in the first line.', action='store_true')
    parser.add_argument('--out-format', help='Output format. Default: "%(default)s"', type=str, default='pprint', choices=['dict', 'json', 'pprint'])
    parser.add_argument('--sep', help='Specify column separator for data CSV files. Default: "%(default)s"', type=str, default=',')
    parser.add_argument('--label-col', help='Specify which column name is the ML label. Default: last column', type=str)
    parser.add_argument('--datetime-format', help='Specify regex format for matching datetimes. Default: "%(default)s"', type=str, default="^\d{4}-\d{2}-\d{2}")
    parser.add_argument('--time-series-len', help='Data is a time series, of a fixed number of specified timesteps.', type=int)
    #parser.add_argument('--time-series-col', help='Data is a time series, of a variable number of timesteps. Specify which column (number or name) is the key')
    # TODO: --time-series-col <INT>|<STR>
    args = parser.parse_args()

    if args.no_header:
        header_handling = None
    else:
        header_handling = 'infer'

    df = dd.read_csv(args.csv, sep=args.sep, na_values='\\N', skipinitialspace=True, header=header_handling)
    df.info(verbose=False, buf=sys.stderr)
    print(df.head(), '\n\n', file=sys.stderr)

    # Metadata related to the dataframe are represented as a dictionary.  Keys are the column names
    md = df_metadata(df, label_col=args.label_col, datetime_regex=args.datetime_format, time_series_len=args.time_series_len)

    # Format if necessary
    if args.out_format == 'json':
        print(json.dumps(md, cls=NPEncoder))
    elif args.out_format == 'dict':
        print(md)
    elif args.out_format == 'pprint':
        import pprint
        pp = pprint.PrettyPrinter(indent=4, width=100)
        pp.pprint(md)

if __name__ == '__main__':
    main()
