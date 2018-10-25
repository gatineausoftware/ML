
import dask.dataframe as dd
import metadata_extract
import nn


uri = "mysql+pymysql://root:teradata@192.168.99.100:31280/health"

df = dd.read_sql_table('heart', uri, 'max_heart_rate', divisions=None, npartitions=None, limits=None, columns=None,
                            bytes_per_chunk=268435456, head_rows=5, schema=None, meta=None)

metadata = metadata_extract.df_metadata(df, label_col=None, datetime_regex="^\d{4}-\d{2}-\d{2}", time_series_len=None, existing_md=None)



#print(metadata)

#df.to_csv('/home/benmackenzie/Projects/ML/export/heart-*.csv')