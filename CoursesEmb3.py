import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from db_operations12 import create_connection
import mysql.connector

def get_face_analysis_model():
    app = FaceAnalysis()
    app.prepare(ctx_id=-1)  # Use CPU; set to a GPU if available
    return app

def preprocess_image(image_path, app):
    img = cv2.imread(image_path)
    faces = app.get(img)
    
    # Get the first face's embedding (assuming one face per image)
    if faces:
        return faces[0].normed_embedding
    return None


def update_course_embeddings(course_id, student_name, student_images_folder):
    embeddings = {}

    # Load existing embeddings for the course using course_id
    embeddings_file = f'embeddings/{course_id}.pkl'  # Use course_id for the filename
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)

    # Initialize the face analysis model inside this function
    app = get_face_analysis_model()

    # Generate embeddings for the student's images
    student_embeddings = []
    for image_name in os.listdir(student_images_folder):
        image_path = os.path.join(student_images_folder, image_name)
        embedding = preprocess_image(image_path, app)
        if embedding is not None:
            student_embeddings.append(embedding)

    # Average the embeddings and store them in the course's embedding file
    if student_embeddings:
        embeddings[student_name] = np.mean(student_embeddings, axis=0)

    # Save the updated embeddings to the course's `.pkl` file using the course ID
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    # Now, save the embeddings to the database
    save_embeddings_to_db(course_id, embeddings)

    print(f"Updated embeddings for course ID '{course_id}' with student '{student_name}'")


# Function to save embeddings to the database
def save_embeddings_to_db(course_id, embeddings):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Serialize the embeddings to store in the database
        embeddings_binary = pickle.dumps(embeddings)
        print(f"Serialized embeddings for course ID '{course_id}': {embeddings_binary[:100]}...")  # Print part of the binary data for debugging

        # Insert the embeddings into the database using the course_id
        cursor.execute(
            "INSERT INTO embeddings (course_id, embedding) VALUES (%s, %s)",
            (course_id, embeddings_binary)  # Use course_id directly
        )

        connection.commit()
        print(f"Embeddings for course ID '{course_id}' saved to the database.")
    except mysql.connector.Error as err:
        print(f"Error saving embeddings to database: {err}")
    finally:
        cursor.close()
        connection.close()


# Ensure there's no code outside functions that runs automatically
if __name__ == '__main__':
    # Only for testing or direct execution, not when imported
    update_course_embeddings('course_id', 'student_name', 'student_images_folder')

