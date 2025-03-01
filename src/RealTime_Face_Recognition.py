import os
import cv2
import numpy as np
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

# Connect to PostgreSQL
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

# Load stored face embeddings from the database
def get_stored_faces():
    conn = connect_db()
    if conn is None:
        return []
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, embedding FROM face_embeddings")
    faces = cursor.fetchall()
    cursor.close()
    conn.close()
    # Convert embedding from list to numpy array for comparison
    return [(str(face[0]), face[1], np.array(face[2])) for face in faces]

# Process each frame to detect and recognize faces
def recognize_faces_in_frame(frame, stored_faces, detector, threshold=0.6):
    recognized_faces = []
    # Convert frame to RGB (MTCNN requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)
    for face in faces:
        x, y, width, height = face['box']
        # Ensure coordinates are valid
        x, y = abs(x), abs(y)
        face_img = frame_rgb[y:y+height, x:x+width]
        face_img = cv2.resize(face_img, (160, 160))
        
        # Extract embedding using DeepFace (pass the image array)
        try:
            result = DeepFace.represent(img_path=face_img, model_name="Facenet", enforce_detection=False)
            embedding = result[0]['embedding']
        except Exception as e:
            print("Error extracting face embedding:", e)
            continue

        recognized_name = "Unknown"
        for user_id, name, stored_embedding in stored_faces:
            distance = np.linalg.norm(np.array(embedding) - stored_embedding)
            if distance < threshold:
                recognized_name = name
                break

        recognized_faces.append((x, y, width, height, recognized_name))
    return recognized_faces

def main():
    # Load stored embeddings once at the start
    stored_faces = get_stored_faces()
    if len(stored_faces) == 0:
        print("No stored face embeddings found in the database.")
        return

    detector = MTCNN()

    # Open the default video capture device (usually the webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting ...")
            break

        # Recognize faces in the current frame
        recognized_faces = recognize_faces_in_frame(frame, stored_faces, detector, threshold=0.6)

        # Draw bounding boxes and names for each detected face
        for (x, y, width, height, name) in recognized_faces:
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Real-Time Face Recognition", frame)
        # Press 'q' to exit the video loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
