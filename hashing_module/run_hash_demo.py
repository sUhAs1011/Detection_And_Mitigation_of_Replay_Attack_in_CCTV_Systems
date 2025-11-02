from hashing_module.src.hasher import hash_video_frames
from hashing_module.src.storage import save_hashes_to_db
from hashing_module.db.schema import init_db

def main():
    # 1. Ensure DB schema exists
    init_db()

    # 2. Hash a sample MP4
    records = hash_video_frames(
        r"C:\Users\NITRO\OneDrive\Desktop\Capstone\Capstone_Code\Crime Footages\78.mp4",
        camera_id="CAM03",
        video_id="VID003"
    )

    # 3. Save hashes into DB
    save_hashes_to_db(records)

    print(f"Hashed {len(records)} frames from 25.mp4")

if __name__ == "__main__":
    main()
