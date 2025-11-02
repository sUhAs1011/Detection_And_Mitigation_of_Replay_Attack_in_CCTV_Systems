import sqlite3
from hashing_module.config.config import DB_PATH

def verify_video(video_path, camera_id="CAM01", min_contiguous=30):
    from hashing_module.src.hasher import hash_video_frames
    new_records = hash_video_frames(video_path, camera_id, video_id="VERIFY_RUN")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    matches = []
    for rec in new_records:
        frame_no, h = rec[2], rec[4]
        cur.execute("SELECT video_id, frame_number FROM frame_hashes WHERE hash=?", (h,))
        rows = cur.fetchall()
        if rows:
            matches.extend([(frame_no, h, r[0], r[1]) for r in rows])

    conn.close()
    return matches
