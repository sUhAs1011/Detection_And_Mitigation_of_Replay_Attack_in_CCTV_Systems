import cv2
import hashlib
import os
from datetime import datetime
from hashing_module.config.config import FRAME_SIZE, TO_GRAY, EVERY_NTH_FRAME, HASH_ALGO

def normalize_frame(frame):
    if TO_GRAY:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return frame

def hash_frame(frame):
    frame_bytes = frame.tobytes()
    if HASH_ALGO == "sha256":
        return hashlib.sha256(frame_bytes).hexdigest()
    elif HASH_ALGO == "blake2b":
        return hashlib.blake2b(frame_bytes, digest_size=32).hexdigest()
    else:
        raise ValueError("Unsupported hash algorithm")

def hash_video_frames(video_path, camera_id="CAM01", video_id=None):
    if video_id is None:
        video_id = os.path.basename(video_path) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_idx = -1
    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % EVERY_NTH_FRAME != 0:
            continue

        ts_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if fps > 0 else None
        norm = normalize_frame(frame)
        h = hash_frame(norm)

        records.append((camera_id, video_id, frame_idx, ts_sec, h,
                        HASH_ALGO, int(TO_GRAY), FRAME_SIZE[0], FRAME_SIZE[1], "raw", "saved"))

    cap.release()
    return records
