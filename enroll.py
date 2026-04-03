import argparse
import json
import os

import numpy as np
import psycopg2
from deepface import DeepFace

import config


def get_connection():
    try:
        return psycopg2.connect(
            host=config.DB_HOST,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASS,
            port=config.DB_PORT,
            connect_timeout=5,
        )
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None


def extract_embedding(image_path):
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=config.MODEL_NAME,
            detector_backend=config.DETECTOR_BACKEND,
            align=config.ALIGN,
            enforce_detection=True,
        )
        if not results:
            return None
        return results[0]["embedding"]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def save_to_cache(name, embedding):
    cache_file = "enrollment_cache.json"
    cache_data = []
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

    cache_data.append({"name": name, "embedding": embedding})
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)
    print(f"Enrollment for {name} saved to local cache ({cache_file}).")


def enroll_member(name, folder_path):
    conn = get_connection()
    db_available = conn is not None
    if db_available:
        conn.close()
    else:
        print("Warning: Database not available. Progress will be cached locally.")

    embeddings = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            print(f"Extracting embedding from {filename}...")
            embedding = extract_embedding(image_path)
            if embedding:
                embeddings.append(embedding)

    if not embeddings:
        print("Error: No faces could be extracted from the provided images.")
        return

    avg_embedding = np.mean(embeddings, axis=0).tolist()

    if db_available:
        conn = None
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO members (name, face_embedding) VALUES (%s, %s)",
                (name, avg_embedding),
            )
            conn.commit()
            print(f"Successfully enrolled member: {name}")
            return
        except Exception as e:
            print(f"Error saving to database: {e}")
        finally:
            if conn:
                conn.close()

    save_to_cache(name, avg_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a new member into the Face Attendance system.")
    parser.add_argument("--name", required=True, help="Full name of the member")
    parser.add_argument("--folder", required=False, default=None, help="Path to a folder containing photos of the member")

    args = parser.parse_args()

    if args.folder:
        # Explicit folder provided — must exist
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}. Create it and add photos first.")
            exit(1)
        enroll_member(args.name, args.folder)
    else:
        # Auto-generate folder from name
        parts = args.name.strip().split()
        first_name = parts[0].lower()
        last_name = parts[-1].lower() if len(parts) > 1 else first_name

        folder_name = f"{first_name[0]}{last_name}"
        folder_path = os.path.join("data", "faces", folder_name)

        if os.path.exists(folder_path):
            folder_name = f"{first_name[:2]}{last_name}"
            folder_path = os.path.join("data", "faces", folder_name)

        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}/ - Add photos and re-run, or press Enter to continue if photos are already there.")
        input()
        enroll_member(args.name, folder_path)
