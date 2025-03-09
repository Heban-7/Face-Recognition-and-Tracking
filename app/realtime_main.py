import os
import cv2
import numpy as np
import psycopg2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
from mtcnn import MTCNN
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv("../.env")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Create FastAPI app instance
app = FastAPI(title="Real-Time Face Recognition API")

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

def find_nearest_face(conn, embedding):
    """
    Perform a vector similarity search (using pgvector) to find the closest face embedding.
    Returns the matched name and the cosine distance.
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
    """
    Extract and normalize the face embedding using DeepFace.
    Uses the Facenet model and returns a unit vector.
    """
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
    Detects faces in a given frame and recognizes each one.
    Returns a list of tuples with bounding box coordinates and the recognized name.
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

@app.post("/recognize_faces")
async def recognize_faces_api(file: UploadFile = File(...)):
    """
    API endpoint for real-time face recognition.
    Accepts an image file (frame), detects faces, and returns bounding box coordinates and recognized names.
    """
    try:
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file."})
    except Exception as e:
        logging.error("Error processing uploaded file: %s", e)
        return JSONResponse(status_code=500, content={"error": "Error processing image."})
    
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "Database connection failed."})
    
    recognized_faces = detect_and_recognize_faces(frame, conn)
    conn.close()
    
    # Format the results as a list of dictionaries
    response_faces = []
    for (x, y, width, height, recognized_name) in recognized_faces:
        response_faces.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "name": recognized_name
        })
    
    return {"recognized_faces": response_faces}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
