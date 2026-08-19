import sqlite3
import json

conn = sqlite3.connect('reports.db')
cur = conn.cursor()
cur.execute("SELECT id, summary FROM reports WHERE department = 'General'")
rows = cur.fetchall()
print(f"Found {len(rows)} General reports:")
for r in rows:
    print(r)
conn.close()
