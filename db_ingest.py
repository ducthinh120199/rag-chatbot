import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="chinook",
    user="postgres",
    password="pass"
)

cursor = conn.cursor()
cursor.execute('SELECT * FROM "customer" LIMIT 10')
rows = cursor.fetchall()

for row in rows:
    print(row)

cursor.close()
conn.close()