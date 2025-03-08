import os
import numpy as np
import cv2
import psycopg2
from deepface import DeepFace
import uuid
from mtcnn import MTCNN
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Path to the folder containing face images
FACE_IMAGE_FOLDER = "../data/faces/"

# Create a global MTCNN detector
detector = MTCNN()

# Connect to PostgreSQL database
def connect_db():
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
        print("Error connecting to database:", e)
        return None

# Create table
def create_table():
    conn = connect_db()
    if conn is None:
        return
    cursor = conn.cursor()
    # Create table with vector column 
    query_table = '''
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id UUID PRIMARY KEY,
        name TEXT NOT NULL,
        embedding vector(128) NOT NULL
    );
    '''
    cursor.execute(query_table)
    # Create HNSW index on the embedding column for fast similarity search
    query_index = '''
    CREATE INDEX IF NOT EXISTS face_embeddings_hnsw_idx ON face_embeddings USING hnsw (embedding vector_l2_ops);
    '''
    cursor.execute(query_index)
    conn.commit()
    cursor.close()
    conn.close()
    print("Database table and HNSW index created successfully!")

# Preprocess image: detect and crop the face using MTCNN
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None
    x, y, width, height = faces[0]['box']
    x, y = abs(x), abs(y)
    face = image_rgb[y:y+height, x:x+width]
    face = cv2.resize(face, (160, 160))
    return face

# Extract face embedding using DeepFace
def get_face_embedding(image_path):
    face = preprocess_image(image_path)
    if face is None:
        return None
    try:
        result = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
        embedding = result[0]['embedding']
        # Normalize the embedding
        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
# Store embedding in PostgreSQL
def store_face_embedding(name, embedding):
    conn = connect_db()
    if conn is None:
        return
    cursor = conn.cursor()
    user_id = uuid.uuid4()  # Unique ID
    embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'    # Convert embedding to a vector literal string (e.g., "[0.1,0.2,...]")
    query = "INSERT INTO face_embeddings (id, name, embedding) VALUES (%s, %s, %s::vector)"
    cursor.execute(query, (str(user_id), name, embedding_str))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Stored embedding for {name} (ID: {user_id})")

# Process all images in the folder
def process_all_faces():
    create_table()
    for filename in os.listdir(FACE_IMAGE_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(FACE_IMAGE_FOLDER, filename)
            name = os.path.splitext(filename)[0]
            print(f"Processing {name} from {filename}...")
            embedding = get_face_embedding(image_path)
            if embedding is not None:
                store_face_embedding(name, embedding)
            else:
                print(f"Skipping {name} due to processing error.")

if __name__ == "__main__":
    process_all_faces()
