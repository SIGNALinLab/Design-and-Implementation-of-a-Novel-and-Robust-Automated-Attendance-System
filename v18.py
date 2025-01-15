import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle
import os
import math
from flask import session
from db_operations12 import process_and_store_attendance

face_database_path = '/home/engine8/Desktop/SDP/database'

def get_all_faces():
    face_ids = set(os.listdir(face_database_path))
    return face_ids


def load_course_embeddings(embeddings_directory):
    # Get the selected course ID from the session
    course_id = session.get('selected_course_id')  # Use the course ID stored in the session

    if course_id:
        # Construct the path to the corresponding .pkl file based on the course ID
        embeddings_file = os.path.join(embeddings_directory, f'{course_id}.pkl')
        print(f"Looking for embeddings file: {embeddings_file}")  # Debugging statement

        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"Embeddings loaded successfully for course ID {course_id}")  # Debugging statement
            return embeddings
        else:
            print(f"Embeddings file {embeddings_file} not found")
            return None
    else:
        print("No course selected")
        return None



def find_closest_embedding(embedding, embeddings):
    min_distance = float('inf')
    recognized_name = None
    for name, known_embedding in embeddings.items():
        distance = np.linalg.norm(embedding - known_embedding)
        if distance < min_distance:
            min_distance = distance
            recognized_name = name
    recognition_rate = distance_to_recognition_rate(min_distance)
    return recognized_name, min_distance, recognition_rate



def distance_to_recognition_rate(distance, offset=1.1, scale=0.05):
    recognition_rate = 1 / (1 + math.exp((distance - offset) / scale))
    return recognition_rate * 100  # Convert to percentage



def preprocess_image_with_clahe(img, convert_to_gray=False):
    if convert_to_gray:
        # Convert to grayscale if the option is set
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(18, 18))
        clahe_img = clahe.apply(img)
    else:
        # Otherwise, apply CLAHE to each channel separately
        channels = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(18, 18))
        clahe_channels = [clahe.apply(channel) for channel in channels]
        clahe_img = cv2.merge(clahe_channels)
    
    return clahe_img 


def recognize_and_draw(input_image_path, embeddings, threshold):
    app = FaceAnalysis(name='buffalo_l')
    
    app.prepare(ctx_id=0)  # Use GPU if available

    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Could not load image at {input_image_path}. Please check the file path.")
    
    #img = preprocess_image_with_clahe(img, convert_to_gray=True)  # Preprocess image

    save_faces_folder = "detected_faces"
    if not os.path.exists(save_faces_folder):
        os.makedirs(save_faces_folder)

    if embeddings is None:
        print("No embeddings available for the selected course")
        return None  # Or handle this more gracefully as needed

    recognition_results = {name: 0 for name in embeddings.keys()}

    # Process the image (this could be the original or a cropped image)
    print("Processing the image for face detection and recognition...")
    faces = app.get(img)
    recognition_results = process_faces(faces, img, embeddings, threshold, recognition_results, save_faces_folder, [])

    # Save the final image with recognized faces
    cv2.imwrite("recognized_faces.jpg", img)
    print(f"Recognition results: {recognition_results}")

    return recognition_results


def process_faces(faces, img_color, embeddings, threshold, recognition_results, save_faces_folder, detected_faces):
    # detected_faces is a list of already processed faces (bounding boxes or embeddings)
    
    for i, face in enumerate(faces):
        if face.normed_embedding is not None:
            recognized_name, distance, recognition_rate = find_closest_embedding(face.normed_embedding, embeddings)
            
            # Check if this face is already detected (using bounding box or embedding similarity)
            if is_duplicate(face, detected_faces):
                continue  # Skip if duplicate
            
            if recognition_rate >= threshold:
                recognition_results[recognized_name] = 1  # Mark as Present (1)
                bbox = face.bbox.astype(int)

                # Validate the bounding box coordinates
                x1, y1, x2, y2 = bbox
                if x1 < 0 or y1 < 0 or x2 > img_color.shape[1] or y2 > img_color.shape[0]:
                    print(f"Invalid bounding box for {recognized_name}: {bbox}")
                    continue  # Skip this face if the bounding box is invalid

                cropped_face = img_color[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    print(f"Skipping empty cropped face for {recognized_name}")
                    continue

                # Save the cropped face
                face_filename = os.path.join(save_faces_folder, f"{recognized_name}_{i}.jpg")
                cv2.imwrite(face_filename, cropped_face)

                # Draw bounding box and label on the image
                cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{recognized_name} ({recognition_rate:.2f}%)"
                cv2.putText(img_color, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                
                # Add this face to the list of detected faces (track either bounding box or embedding)
                detected_faces.append(face.normed_embedding)  # OR append bbox if comparing bounding boxes

    return recognition_results



def is_duplicate(new_face, detected_faces, method="embedding", threshold=0.6):
    """
    Check if a new face is a duplicate based on existing detected faces.
    method: "embedding" or "bbox" to determine comparison method.
    threshold: similarity threshold for embeddings (0.6 is a common threshold for cosine similarity).
    """
    if method == "embedding":
        for existing_embedding in detected_faces:
            similarity = np.dot(new_face.normed_embedding, existing_embedding)
            if similarity >= threshold:  # If embeddings are too similar, consider it a duplicate
                return True
    elif method == "bbox":
        new_bbox = new_face.bbox.astype(int)
        for existing_bbox in detected_faces:
            if overlap(new_bbox, existing_bbox):  # Check if bounding boxes overlap significantly
                return True

    return False



def overlap(bbox1, bbox2, iou_threshold=0.5):
    """
    Check if two bounding boxes overlap significantly (IoU threshold).
    """
    x1_max = max(bbox1[0], bbox2[0])
    y1_max = max(bbox1[1], bbox2[1])
    x2_min = min(bbox1[2], bbox2[2])
    y2_min = min(bbox1[3], bbox2[3])

    # Compute the area of intersection
    intersection_area = max(0, x2_min - x1_max + 1) * max(0, y2_min - y1_max + 1)

    # Compute the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    # Compute the Intersection over Union (IoU)
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou > iou_threshold  # Return True if IoU exceeds the threshold (considered duplicate)



def crop_and_zoom(image, crop_coords, zoom_factor=3, index=0):
    # Crop the region of interest (ROI) from the image
    x, y, w, h = crop_coords
    cropped_image = image[y:y+h, x:x+w]

    # Print the original size of the cropped image
    print(f"Original cropped image {index} size: {cropped_image.shape}")

    # Resize the cropped image to zoom in
    zoomed_image = cv2.resize(cropped_image, None, fx=3, fy=zoom_factor)

    # Print the zoomed size
    print(f"Zoomed image {index} size: {zoomed_image.shape}")

    return zoomed_image


def resize_image(image_path, output_size=(1280, 720)):
    """
    Resizes the image to the specified output size.
    Args:
        image_path (str): Path to the original image.
        output_size (tuple): Desired output size (width, height).
    Returns:
        str: Path to the resized image.
    """
    # Read the image from the given path
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = cv2.resize(image, output_size)

    # Save the resized image (overwrite or save with a new name)
    resized_image_path = f"resized_{image_path}"
    cv2.imwrite(resized_image_path, resized_image)

    return resized_image_path


def classify_students(recognition_results_T1, recognition_results_T2, recognition_results_T3, course_id, session_id):
    classified_results = {
        "Attended": [],
        "Absent": [],
        "Late": [],
        "Out-of-class": []
    }

    # Sets to track students
    attended_set = set()
    late_set = set()
    out_of_class_set = set()

    # Students recognized in the first, second, and third images
    students_T1 = set(os.path.basename(student_path) for student_path, status in recognition_results_T1.items() if status == 1)
    students_T2 = set(os.path.basename(student_path) for student_path, status in recognition_results_T2.items() if status == 1)
    students_T3 = set(os.path.basename(student_path) for student_path, status in recognition_results_T3.items() if status == 1)

    # Out-of-class Students:
    # 1. Recognized in the first image but not in the second (even if recognized in the third)
    out_of_class_set.update(students_T1 - students_T2)
    
    # 2. Recognized in the second image but not in the third (even if recognized in the first)
    out_of_class_set.update(students_T2 - students_T3)

    # Attended Students: Recognized in the first image and not marked as out-of-class
    attended_set = students_T1 - out_of_class_set

    # Late Students: Recognized in both the second and third images but not in the first image, and not out-of-class
    late_set = (students_T2 & students_T3) - students_T1 - out_of_class_set

    # Absent Students: Students who weren't recognized in the first or second images
    all_students = set(os.path.basename(student_path) for student_path in recognition_results_T1.keys())
    absent_set = all_students - attended_set - late_set - out_of_class_set

    # Add sets to classified results
    classified_results["Attended"] = list(attended_set)
    classified_results["Late"] = list(late_set)
    classified_results["Out-of-class"] = list(out_of_class_set)
    classified_results["Absent"] = list(absent_set)

    print(f"Classified results: {classified_results}")
    return classified_results



def main(image_path):
    embeddings_file = 'embeddings'
    embeddings = load_course_embeddings(embeddings_file)

    threshold = 35  # Recognition rate threshold (%)

    try:
        recognition_results = recognize_and_draw(image_path, embeddings, threshold)
        return recognition_results
    except ValueError as e:
        print(e)
        return {}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        main(image_path)
    else:
        print("Please provide an image path")
