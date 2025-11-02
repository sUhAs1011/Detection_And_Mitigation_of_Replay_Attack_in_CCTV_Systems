import os

# Base directory = hashing_module/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Database path inside hashing_module/db/
DB_PATH = os.path.join(BASE_DIR, "db", "hash_store.db")

# Canonicalization parameters
FRAME_SIZE = (320, 240)   # width, height
TO_GRAY = True
EVERY_NTH_FRAME = 1

# Hashing parameters
HASH_ALGO = "sha256"      # or "blake2b"
