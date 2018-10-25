
import dask.dataframe as dd
import metadata_extract
import nn


uri = "mysql+pymysql://root:teradata@192.168.99.100:31280/health"

df = dd.read_sql_table('heart', uri, 'max_heart_rate', divisions=None, npartitions=None, limits=None, columns=None,
                            bytes_per_chunk=268435456, head_rows=5, schema=None, meta=None)

metadata = metadata_extract.df_metadata(df, label_col=None, datetime_regex="^\d{4}-\d{2}-\d{2}", time_series_len=None, existing_md=None)

metadata['_ml']['problem_type'] = 'classification'


fc = nn.build_feature_columns(df, 'narrowing_diagnosis', metadata)

model = nn.build_estimator(metadata, 64, 'Adam', '/home/benmackenzie/Projects/ML/models/', 1, fc, 'narrowing_diagnosis',2, 10, 0.1)

config = {}
config['batch_size'] = 16

nn.train_and_evaluate(model, df, 'narrowing_diagnosis',metadata, config)

#print(metadata)

#df.to_csv('/home/benmackenzie/Projects/ML/export/heart-*.csv')