import pymysql






conn = pymysql.connect(
    host='192.168.99.100',
    port=int('31280'),
    user='root',
    passwd='teradata',
    db='health',
    charset='utf8mb4')


cursor = conn.cursor()
cursor.execute("select * from heart")
rows = cursor.fetchmany(8)
print(rows)



