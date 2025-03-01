import os, sys
import numpy as np
import cv2
import psycopg2
from deepface import DeepFace
from mtcnn import MTCNN
from dotenv import load_dotenv
import uuid
import json

# Load environment variables
load_dotenv("../.env")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

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

# Fetch stored face embeddings from the database
def get_stored_faces():
    conn = connect_db()
    if conn is None:
        return []
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding FROM face_embeddings")
    faces = cursor.fetchall()
    cursor.close()
    conn.close()
    return [(str(face[0]), face[1], np.array(face[2])) for face in faces]

# Store a new face embedding in the database
def store_new_face(name, embedding):
    conn = connect_db()
    if conn is None:
        return
    cursor = conn.cursor()
    user_id = uuid.uuid4()
    query = "INSERT INTO face_embeddings (id, name, embedding) VALUES (%s, %s, %s)"
    cursor.execute(query, (str(user_id), name, embedding.tolist()))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"New user {name} stored with ID {user_id}")

# Extract face from frame using MTCNN
def extract_face(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    if len(faces) == 0:
        return None
    x, y, width, height = faces[0]['box']
    x, y = abs(x), abs(y)
    face = frame[y:y+height, x:x+width]
    face = cv2.resize(face, (160, 160))
    return face

# Recognize face in the given frame
def recognize_face(frame):
    face = extract_face(frame)
    if face is None:
        print("No face detected!")
        return None, None
    try:
        # Use the image array with the "img" parameter
        result = DeepFace.represent(img_path=face, model_name="Facenet", enforce_detection=False)
        embedding = result[0]['embedding']
    except Exception as e:
        print("Error during face embedding:", e)
        return None, None

    stored_faces = get_stored_faces()
    recognized_name = None
    threshold = 0.6  # Matching threshold
    for user_id, name, stored_embedding in stored_faces:
        distance = np.linalg.norm(np.array(embedding) - stored_embedding)
        if distance < threshold:
            recognized_name = name
            break

    # If a match is found, no need to register a new user; otherwise, return the embedding for registration
    if recognized_name:
        return recognized_name, None
    else:
        return None, np.array(embedding)

# Process an image for recognition and optionally register a new user
def image_recognition(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error loading image!")
        return
    # Convert the frame to RGB for proper face detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    name, new_embedding = recognize_face(frame_rgb)
    
    if name:
        print(f"Hello, {name}! Welcome back!")
    elif new_embedding is not None:
        new_name = input("New user detected. Please enter your name: ")
        store_new_face(new_name, new_embedding)
        print(f"Welcome, {new_name}! You have been registered.")
    else:
        print("No face recognized or detected.")

    cv2.imshow("Face Recognition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("Enter the image path: ")
    image_recognition(image_path)
