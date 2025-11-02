import sqlite3
from hashing_module.config.config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS frame_hashes (
            camera_id TEXT,
            video_id TEXT,
            frame_number INTEGER,
            timestamp REAL,
            hash TEXT,
            hash_algo TEXT,
            gray INTEGER,
            width INTEGER,
            height INTEGER,
            serialize_mode TEXT,
            status TEXT DEFAULT 'saved'
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_video_frame ON frame_hashes(video_id, frame_number)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hash ON frame_hashes(hash)")
    conn.commit()
    conn.close()
    print(f"[INFO] Initialized DB at {DB_PATH}")

if __name__ == "__main__":
    init_db()
