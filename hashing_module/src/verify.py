import sqlite3
from config.config import DB_PATH
from src.hasher import hash_video_frames

def verify_video(video_path, camera_id="CAM01", min_contiguous=30):
    new_records = hash_video_frames(video_path, camera_id, video_id="verify_run")
    new_hashes = [r[4] for r in new_records]  # extract hash column

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    matches = []
    for idx, h in enumerate(new_hashes):
        cur.execute("SELECT video_id, frame_number FROM frame_hashes WHERE hash = ?", (h,))
        rows = cur.fetchall()
        if rows:
            matches.append((idx, h, rows))

    conn.close()
    return matches
