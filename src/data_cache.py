
import dask.dataframe as dd
import metadata_extract
import nn
import tensorflow as tf
import features


tf.logging.set_verbosity(tf.logging.INFO)

config = {}
config['batch_size'] = 16
config['data_cache'] = '/home/benmackenzie/Projects/ML/cache/'
config['model_dir'] = '/home/benmackenzie/Projects/ML/models/'
config['optimizer'] = 'Adam'

#this info would be supplied through UI
table_name = 'heart2'
index_col = 'id'
label_col = 'narrowing_diagnosis'

uri = "mysql+pymysql://root:teradata@192.168.99.100:31280/health"


df = dd.read_sql_table(table_name, uri, index_col, divisions=None, npartitions=None, limits=None, columns=None,
                            bytes_per_chunk=268435456, head_rows=5, schema=None, meta=None)



p = features.get_postalcode()
p(df, 'location', 'postalcode')


#get metadata
metadata = metadata_extract.df_metadata(df, label_col=label_col, datetime_regex="^\d{4}-\d{2}-\d{2}", time_series_len=None, existing_md=None)


#from UI
metadata['_ml']['problem_type'] = 'classification'

#convert categorical floats to string
nn.cat_float_to_str(df, metadata)

#get defaults for missing features
d = nn.get_defaults(df, metadata)


fc = nn.build_feature_columns(df, label_col, metadata)


#cache data on file system for training
cache = nn.cache_df(df, table_name, config)

#cache = ["/home/benmackenzie/Projects/ML/cache/heart2/cache-0.csv"]


model = nn.build_estimator(metadata, 64, config['optimizer'], config['model_dir'], 1, fc, label_col, 2, 10, 0.1)

feature_names = [colname for colname in df.columns]

nn.train_and_evaluate(model, cache, feature_names, label_col, metadata, config, d)



