import os
import cv2
import numpy as np
import logging
import psycopg2
from deepface import DeepFace
from mtcnn import MTCNN
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Ensure logs folder exist
os.makedirs('../logs', exist_ok=True)

# Cinfigure logging
logging.basicConfig(
    level=logging.INFO,
    format= "%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('../logs/realtime_face_recognition.log'),
        logging.StreamHandler()
    ]
)

# Global detector 
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
        logging.info("Database connection established.")
        return conn
    except Exception as e:
        logging.error("Error connecting to database: %s", e)
        return None

def find_nearest_face(conn, embedding):
    """
    Find the nearest face from the database using pgvector similarity search.
    Returns the name and cosine distance.
    """
    try:
        cursor = conn.cursor()
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
        return result[1], result[2]
    except Exception as e:
        logging.error("Error in find_nearest_face: %s", e)
        return None, None

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
        logging.warning("Error extracting face embedding: %s", e)
        return None

def detect_and_recognize_faces(frame, conn, threshold=0.3):
    """
    Detect and recognize faces in a video frame.
    Returns a list of tuples: (x, y, width, height, recognized_name).
    """
    recognized_faces = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)
    
    for face in faces:
        try:
            x, y, width, height = face['box']
            x, y = abs(x), abs(y)
            face_img = frame_rgb[y:y+height, x:x+width]
            face_img = cv2.resize(face_img, (160, 160))
            embedding = get_face_embedding(face_img)
            if embedding is None:
                continue
        except Exception as e:
            logging.warning("Error processing face: %s", e)
            continue

        recognized_name = "Unknown"
        name, cosine_distance = find_nearest_face(conn, embedding)
        if cosine_distance is not None and cosine_distance < threshold:
            recognized_name = name

        recognized_faces.append((x, y, width, height, recognized_name))
    return recognized_faces

def run_video_recognition():
    """Run real-time face recognition using the webcam."""
    conn = get_db_connection()
    if conn is None:
        logging.error("Database connection failed. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open video stream.")
        return

    logging.info("Starting real-time face recognition. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame. Exiting...")
            break

        recognized_faces = detect_and_recognize_faces(frame, conn)
        for (x, y, width, height, recognized_name) in recognized_faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, recognized_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()
    logging.info("Resources released. Exiting.")

if __name__ == "__main__":
    run_video_recognition()
