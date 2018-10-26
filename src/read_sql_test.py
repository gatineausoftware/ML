import pymysql
import pandas as pd



#

conn = pymysql.connect(
    host='192.168.99.100',
    port=int('31280'),
    user='root',
    passwd='teradata',
    db='health',
    charset='utf8mb4')


df = pd.read_sql_query("select * from heart", conn)

print(df.head())

