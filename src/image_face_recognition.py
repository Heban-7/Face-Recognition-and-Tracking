import os
import cv2
import numpy as np
import psycopg2
from deepface import DeepFace
from mtcnn import MTCNN
from dotenv import load_dotenv
import uuid
import logging
import argparse

# Ensure logs folder exist
os.makedirs('../logs', exist_ok=True)

# Cinfigure logging
logging.basicConfig(
    level=logging.INFO,
    format= "%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('../logs/image_based_face_recognition.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv("../.env")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Global face detector instance
detector = MTCNN()

def get_db_connection():
    """Establish a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logging.error("Error connecting to database: %s", e)
        return None

def find_nearest_face(embedding):
    """
    Find the nearest face in the database using pgvector similarity search.
    Returns the matched name and the cosine distance.
    """
    conn = get_db_connection()
    if conn is None:
        return None, None
    try:
        cursor = conn.cursor()
        # Convert embedding to the pgvector literal format.
        embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
        query = """
        SELECT id, name, embedding <=> %s::vector AS cosine_distance
        FROM face_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
        """
        cursor.execute(query, (embedding_str, embedding_str))
        result = cursor.fetchone()
        cursor.close()
        if result is None:
            return None, None
        return result[1], result[2]  # name and cosine_distance
    except Exception as e:
        logging.error("Error in find_nearest_face: %s", e)
        return None, None
    finally:
        conn.close()

def store_new_face(name, embedding):
    """Store a new face embedding in the database."""
    conn = get_db_connection()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        user_id = uuid.uuid4()
        embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
        query = "INSERT INTO face_embeddings (id, name, embedding) VALUES (%s, %s, %s::vector)"
        cursor.execute(query, (str(user_id), name, embedding_str))
        conn.commit()
        logging.info("New user '%s' stored with ID %s", name, user_id)
        cursor.close()
    except Exception as e:
        logging.error("Error storing new face: %s", e)
    finally:
        conn.close()

def extract_face(frame):
    """Detect and extract the face from the given image frame."""
    faces = detector.detect_faces(frame)
    if not faces:
        logging.warning("No face detected in the frame.")
        return None
    x, y, width, height = faces[0]['box']
    x, y = abs(x), abs(y)
    face = frame[y:y+height, x:x+width]
    try:
        face = cv2.resize(face, (160, 160))
    except Exception as e:
        logging.error("Error resizing face image: %s", e)
        return None
    return face

def get_face_embedding(face_img):
    """Extract and normalize the face embedding using DeepFace."""
    try:
        result = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)
        embedding = np.array(result[0]['embedding'])
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        logging.error("Error during face embedding extraction: %s", e)
        return None

def recognize_face(frame, threshold=0.5):
    """
    Recognize the face in a given frame.
    Returns the recognized name if found (within threshold); otherwise returns the new embedding.
    """
    face = extract_face(frame)
    if face is None:
        return None, None
    embedding = get_face_embedding(face)
    if embedding is None:
        return None, None
    recognized_name, cosine_distance = find_nearest_face(embedding)
    if recognized_name and cosine_distance is not None and cosine_distance < threshold:
        return recognized_name, None
    return None, embedding

def process_image_recognition(image_path):
    """Process an image file for face recognition and, if needed, register a new user."""
    frame = cv2.imread(image_path)
    if frame is None:
        logging.error("Error loading image from path: %s", image_path)
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    name, new_embedding = recognize_face(frame_rgb)
    
    if name:
        logging.info("Hello, %s! Welcome back!", name)
    elif new_embedding is not None:
        new_name = input("New user detected. Please enter your name: ")
        store_new_face(new_name, new_embedding)
        logging.info("Welcome, %s! You have been registered.", new_name)
    else:
        logging.info("No face recognized or detected.")

    cv2.imshow("Face Recognition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image-based Face Recognition Pipeline")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()
    process_image_recognition(args.image_path)
