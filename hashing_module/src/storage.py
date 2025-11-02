import sqlite3
from hashing_module.config.config import DB_PATH

def save_hashes_to_db(records):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO frame_hashes
        (camera_id, video_id, frame_number, timestamp, hash, hash_algo,
         gray, width, height, serialize_mode, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()
    print(f"[INFO] Inserted {len(records)} frame hashes into {DB_PATH}")
