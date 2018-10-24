
import dask.dataframe as dd

uri = "mysql+pymysql://root:teradata@192.168.99.100:31280/health"

df = dd.read_sql_table('heart', uri, 'max_heart_rate', divisions=None, npartitions=None, limits=None, columns=None,
                            bytes_per_chunk=268435456, head_rows=5, schema=None, meta=None)




df.to_csv('/home/benmackenzie/Projects/ML/export/heart-*.csv')