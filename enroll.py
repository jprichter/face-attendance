import argparse
import os
import psycopg2
import json
import numpy as np
from deepface import DeepFace
import config

def get_connection():
    try:
        conn = psycopg2.connect(
            host=config.DB_HOST,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASS,
            port=config.DB_PORT,
            connect_timeout=5
        )
        return conn
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
            enforce_detection=True
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
    # Enhancement: Verify DB connection BEFORE work
    conn = get_connection()
    db_available = conn is not None
    if not db_available:
        print("Warning: Database not available. Progress will be cached locally.")
    else:
        conn.close()

    embeddings = []
    # Process all images in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Extracting embedding from {filename}...")
            embedding = extract_embedding(image_path)
            if embedding:
                embeddings.append(embedding)

    if not embeddings:
        print("Error: No faces could be extracted from the provided images.")
        return

    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0).tolist()

    if db_available:
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO members (name, face_embedding) VALUES (%s, %s)",
                (name, avg_embedding)
            )
            conn.commit()
            print(f"Successfully enrolled member: {name}")
            conn.close()
            return
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    # Enhancement: Cache if DB fails or is unavailable
    save_to_cache(name, avg_embedding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a new member into the Face Attendance system.")
    parser.add_argument("--name", required=True, help="Full name of the member")
    parser.add_argument("--folder", required=True, help="Path to a folder containing photos of the member")
    
    args = parser.parse_args()
    enroll_member(args.name, args.folder)
