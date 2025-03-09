import os
import uuid
import cv2
import numpy as np
import psycopg2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
from mtcnn import MTCNN
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env
load_dotenv("../.env")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Create a FastAPI app instance
app = FastAPI(title="Image-based Face Recognition API")

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
    Returns the matched name and cosine distance.
    """
    conn = get_db_connection()
    if conn is None:
        return None, None
    try:
        cursor = conn.cursor()
        # Convert embedding to a pgvector literal string
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
        return result[1], result[2]  # result[1] is the name, result[2] is the distance
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
    Returns (recognized_name, None) if a match is found within threshold;
    otherwise, returns (None, new_embedding) for registration.
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

@app.post("/recognize_face")
async def recognize_face_api(
    file: UploadFile = File(...),
    name: str = Form(None)
):
    """
    API endpoint for image-based face recognition.
    - If the face is recognized (embedding similarity within threshold), returns the recognized name.
    - If not recognized and a 'name' is provided in the form, registers the new face and returns a welcome message.
    - Otherwise, returns a message prompting for a name.
    """
    try:
        # Read uploaded file and convert to image frame
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file."})
        # Convert image from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error("Error processing uploaded file: %s", e)
        return JSONResponse(status_code=500, content={"error": "Error processing image."})
    
    # Recognize the face in the image
    recognized_name, new_embedding = recognize_face(frame_rgb)
    
    # Return recognized name if found
    if recognized_name:
        return {"recognized": True, "name": recognized_name}
    
    # If face not recognized but embedding was obtained, check if 'name' is provided for registration
    if new_embedding is not None:
        if name:
            store_new_face(name, new_embedding)
            return {"recognized": False, "message": f"New face registered as {name}."}
        else:
            return {
                "recognized": False,
                "message": "Face not recognized. Provide a 'name' field to register this new face."
            }
    
    # In case no face was detected or embedding could not be extracted
    return {"recognized": False, "message": "No face detected or error during processing."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

