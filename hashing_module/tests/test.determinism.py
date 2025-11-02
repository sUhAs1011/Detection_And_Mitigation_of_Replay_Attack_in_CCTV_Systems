from src.hasher import hash_video_frames

def test_determinism():
    video = "data/videos/sample.mp4"
    run1 = hash_video_frames(video, camera_id="TEST", video_id="run1")
    run2 = hash_video_frames(video, camera_id="TEST", video_id="run2")

    hashes1 = [r[4] for r in run1]
    hashes2 = [r[4] for r in run2]

    assert hashes1 == hashes2, "Hashes differ between runs!"
    print("[PASS] Determinism confirmed")

if __name__ == "__main__":
    test_determinism()
