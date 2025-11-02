import sqlite3
from hashing_module.config.config import DB_PATH

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# How many frames were hashed for your test video
cur.execute("SELECT COUNT(*) FROM frame_hashes WHERE video_id='VID001'")
print("Total frames hashed:", cur.fetchone()[0])

# Show the first 5 frame hashes
cur.execute("""
    SELECT frame_number, timestamp, hash
    FROM frame_hashes
    WHERE video_id='VID001'
    ORDER BY frame_number
    LIMIT 5
""")
for row in cur.fetchall():
    print(row)

conn.close()
