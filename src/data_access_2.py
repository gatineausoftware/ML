
import dask.dataframe as dd
import metadata_extract
import nn
import tensorflow as tf



tf.logging.set_verbosity(tf.logging.INFO)



config = {}
config['batch_size'] = 16
config['data_cache'] = 'home/benmackenzie/Projects/ML/export/'
config['model_dir'] = '/home/benmackenzie/Projects/ML/models/'
config['optimizer'] = 'Adam'

#this info would be supplied through UI
table_name = 'heart'
index_col = 'max_heart_rate'  #there isn't an obvious index column
label_col = 'narrowing_diagnosis'

uri = "mysql+pymysql://root:teradata@192.168.99.100:31280/health"

#read data into dask df
df = dd.read_sql_table(table_name, uri, index_col, divisions=None, npartitions=None, limits=None, columns=None,
                            bytes_per_chunk=268435456, head_rows=5, schema=None, meta=None)


#do feature engineering on dask


#get metadata
metadata = metadata_extract.df_metadata(df, label_col=None, datetime_regex="^\d{4}-\d{2}-\d{2}", time_series_len=None, existing_md=None)

metadata['_ml']['problem_type'] = 'classification'


fc = nn.build_feature_columns(df, label_col, metadata)

# cache  = df.to_csv('/home/benmackenzie/Projects/ML/export/heart-*.csv')
cache = "/home/benmackenzie/Projects/ML/export/heart-0.csv"

model = nn.build_estimator(metadata, 64, config['optimizer'], config['model_dir'], 1, fc, label_col, 2, 10, 0.1)


nn.train_and_evaluate(model, cache, 'narrowing_diagnosis', metadata, config)



