from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash, make_response, send_file
import os
from db_operations12 import (
    add_instructor, get_instructor, get_courses, get_student,
    add_course, add_student, get_all_courses, enroll_student_in_course,
    get_students_in_course, remove_course_from_db, remove_student_from_course,
    get_instructor_for_course, get_enrolled_courses, get_course_name_by_id, get_student_name_by_id,
    save_student_embeddings_to_db, check_if_already_enrolled, update_attendance_status, get_student_id_by_name,
    create_session, process_and_store_attendance, create_connection, save_image_to_db, get_most_recent_session
)
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from CoursesEmb3 import update_course_embeddings, preprocess_image, get_face_analysis_model
import time
from aasfinal7 import capture_image
from v18 import main as recognize_faces, classify_students
import random
from v18 import resize_image, load_course_embeddings
import cv2
import pandas as pd
from io import BytesIO
from mysql.connector import Error
from flask_socketio import SocketIO, emit
from yolo_detect5 import run_yolo, detect_faces_retinaface, apply_esrgan, load_esrgan_model
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np


app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = '123'

# Load student names from the database folder
def load_student_names():
    database_path = 'database'
    return [folder for folder in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, folder))]



# Define a folder to store the uploaded images
UPLOAD_FOLDER = 'student_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


    
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        if role == 'instructor':
            add_instructor(name, email, password)
        else:
            result = add_student(name, email, password)
            if result == "Email already exists":
                return "Email already exists. Please use a different email."
        return redirect(url_for('login'))
    return render_template('register5.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        if role == 'instructor':
            user = get_instructor(email)
        else:
            user = get_student(email)
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['role'] = role
            if role == 'instructor':
                session['instructor_id'] = user[0]  # Store instructor_id for instructors
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials"
    return render_template('login5.html')



@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        if session['role'] == 'instructor':
            instructor_id = session['user_id']
            courses = get_courses(instructor_id)

            # Fetch the most recent session ID (assuming you have a way to track it)
            recent_session_id = get_most_recent_session(instructor_id)

            # Fetch images for the most recent session
            connection = create_connection()
            cursor = connection.cursor()
            image_query = "SELECT id FROM attendance_images WHERE session_id = %s"
            cursor.execute(image_query, (recent_session_id,))
            captured_images = [image[0] for image in cursor.fetchall()]
            print(captured_images)  # Debugging: Check the image IDs in the backend console

            return render_template('dashboard_instructor27.html', courses=courses, captured_images=captured_images, show_images_button=True, show_images=False)
        else:
            student_id = session['user_id']
            courses = get_all_courses()
            enrolled_courses = get_enrolled_courses(student_id)

            # Get instructor name for each course
            courses_with_instructors = []
            for course in enrolled_courses:
                instructor_name = get_instructor_for_course(course[0])
                courses_with_instructors.append({
                    'course_id': course[0],
                    'course_name': course[1],
                    'instructor_name': instructor_name
                })

            return render_template('dashboard_student6.html', courses=courses, enrolled_courses=courses_with_instructors)
    else:
        return redirect(url_for('login'))



@app.route('/students_in_course/<int:course_id>')
def students_in_course(course_id):
    students = get_students_in_course(course_id)
    return jsonify(students)



@app.route('/add_course', methods=['POST'])
def add_course_route():
    if 'user_id' in session and session['role'] == 'instructor':
        instructor_id = session['user_id']
        data = request.get_json()
        course_name = data['name']
        course_id = add_course(instructor_id, course_name)
        if course_id:
            return jsonify({'status': 'success', 'course_id': course_id})
        else:
            return jsonify({'status': 'error', 'message': 'Course already exists'})
    else:
        return jsonify({'status': 'error', 'message': 'Not authenticated'})



@app.route('/remove_course/<int:course_id>', methods=['DELETE'])
def remove_course(course_id):
    if 'user_id' in session and session['role'] == 'instructor':
        instructor_id = session['user_id']
        remove_course_from_db(course_id, instructor_id)
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Not authenticated'})



@app.route('/enroll_in_course/<int:course_id>', methods=['POST'])
def enroll_in_course_route(course_id):
    if 'user_id' in session and session['role'] == 'student':
        student_id = session['user_id']

        try:
            # Check if the student is already enrolled
            if check_if_already_enrolled(student_id, course_id):
                return jsonify({'status': 'error', 'message': 'You are already enrolled in this course'})

            # Try enrolling the student
            result = enroll_student_in_course(student_id, course_id)
            if result == "Enrolled successfully":
                # Fetch student name (you no longer need the course name here)
                student_name = get_student_name_by_id(student_id)

                # Get the path where the student's images are stored
                student_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(student_id))

                # Update the course's .pkl file with the student's embeddings using course_id
                update_course_embeddings(str(course_id), student_folder, student_folder)  # Save or update using course ID

                return jsonify({'status': 'success'})
            else:
                return jsonify({'status': 'error', 'message': result})

        except Exception as e:
            # Log the error
            print(f"An exception occurred during enrollment: {str(e)}")
            return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'})
    else:
        return jsonify({'status': 'error', 'message': 'Not authenticated'})



@app.route('/get_instructor/<int:course_id>', methods=['GET'])
def get_instructor_route(course_id):
    instructor_name = get_instructor_for_course(course_id)
    if instructor_name:
        return jsonify({'status': 'success', 'instructor': instructor_name})
    else:
        return jsonify({'status': 'error', 'message': 'Instructor not found'})



@app.route('/remove_student/<int:student_id>/<int:course_id>', methods=['DELETE'])
def remove_student(student_id, course_id):
    if 'user_id' in session and session['role'] == 'instructor':
        instructor_id = session['user_id']
        remove_student_from_course(student_id, course_id, instructor_id)
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Not authenticated'})



@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    return redirect(url_for('login'))  # Redirect to the login page



@app.route('/upload_images', methods=['POST'])
def upload_images():
    if 'user_id' in session and session['role'] == 'student':
        student_id = session['user_id']
        uploaded_files = request.files.getlist('images')
        student_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(student_id))
        all_embeddings = []

        if not os.path.exists(student_folder):
            os.makedirs(student_folder)

        # Initialize the face analysis model once for this request
        face_model = get_face_analysis_model()

        for file in uploaded_files:
            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(student_folder, filename)
                file.save(file_path)

                # Generate embeddings using preprocess_image and save them
                embedding = preprocess_image(file_path, face_model)
                if embedding is not None:
                    all_embeddings.append(embedding)

        # Save student's embeddings
        save_student_embeddings_to_db(student_id, all_embeddings)

        flash('Images uploaded and embeddings generated successfully!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('You must be logged in as a student to upload images.', 'danger')
        return redirect(url_for('login'))
   


@app.route('/select_course', methods=['POST'])
def select_course():
    data = request.get_json()
    course_id = data.get('courseId')  # Expecting course ID directly from the frontend

    if course_id:
        # Store the selected course ID in the session
        session['selected_course_id'] = course_id
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Course ID not provided'})


@app.route('/confirm_attendance', methods=['POST'])
def confirm_attendance():
    updates = request.json.get('updates', [])
    course_id = request.json.get('course_id')
    session_id = request.json.get('session_id')  # Ensure session_id is passed
    
    # Prepare the results dictionary based on user confirmation
    results = {
        "Attended": [],
        "Late": [],
        "Absent": []
    }

    # Collect the confirmed statuses from the user
    for update in updates:
        student_id = update['studentId']
        status = update['status'].lower()  # Convert status to lowercase to match your DB fields

        # Check if the status is out-of-class and mark as absent if not changed
        if status == 'out_of_class':
            results['Absent'].append(student_id)  # Treat 'out_of_class' as absent
        else:
            results[status.capitalize()].append(student_id)  # Append to the corresponding category

    # Now process and store the confirmed attendance in the database
    process_and_store_attendance(course_id, session_id, results)
    
    return jsonify({'success': True})


@app.route('/attendance_logs', methods=['GET'])
def attendance_logs():
    # Check if the instructor is logged in
    instructor_id = session.get('instructor_id')
    course_id = session.get('selected_course_id')  # Retrieve selected course_id from session
    
    if not instructor_id:
        return redirect('/login')
    
    if not course_id:
        flash("Please select a course.")
        return redirect('/dashboard')  # Redirect back to the dashboard if no course is selected

    # Connect to the database
    connection = create_connection()
    cursor = connection.cursor()

    # Fetch course name for the selected course
    course_name_query = "SELECT name FROM courses WHERE id = %s"
    cursor.execute(course_name_query, (course_id,))
    course_name = cursor.fetchone()[0]  # Fetch the course name

    # Fetch attendance data for the selected course
    query = """
    SELECT a.session_id, a.session_date, s.name AS student_name, 
           a.status, a.timestamp
    FROM attendance a
    JOIN students s ON a.student_id = s.id
    JOIN courses c ON a.course_id = c.id
    WHERE c.instructor_id = %s AND a.course_id = %s
    ORDER BY a.session_id, a.timestamp;
    """
    
    cursor.execute(query, (instructor_id, course_id))
    logs = cursor.fetchall()

    # Group logs by session_id
    grouped_logs = {}
    for log in logs:
        session_id = log[0]
        session_date = log[1]
        student_name = log[2]
        status = log[3]
        
        if session_id not in grouped_logs:
            grouped_logs[session_id] = {
                'session_date': session_date,
                'logs': [],
                'images': []
            }
        
        # Append each student's log to the corresponding session
        grouped_logs[session_id]['logs'].append({
            'student_name': student_name,
            'status': status,
        })

    # Fetch images for the selected course and sessions
    session_ids = tuple(grouped_logs.keys())

    # Ensure session_ids is not empty
    if not session_ids:
        session_ids = (-1,)  # Add a dummy value to avoid errors in empty cases

    # Create the correct number of placeholders for the IN clause
    placeholders = ','.join(['%s'] * len(session_ids))

    # Modify the query to use the dynamically generated placeholders
    image_query = f"SELECT id, session_id FROM attendance_images WHERE session_id IN ({placeholders})"
    cursor.execute(image_query, session_ids)
    images = cursor.fetchall()

    # Add the images to the corresponding session
    for image in images:
        image_id, session_id = image
        grouped_logs[session_id]['images'].append(image_id)

    # Ensure logs is not empty
    if not grouped_logs:
        flash("No attendance records found for the selected course.")
    
    # Pass the course_name and grouped logs to the template
    return render_template('attendance_logs6.html', course_name=course_name, grouped_logs=grouped_logs)



@app.route('/export_attendance_logs', methods=['GET'])
def export_attendance_logs():
    instructor_id = session.get('instructor_id')
    course_id = session.get('selected_course_id')  # Assuming course_id is stored in session after course selection
    if not instructor_id:
        return redirect('/login')

    if not course_id:
        return "Please select a course before exporting logs."

    connection = create_connection()
    cursor = connection.cursor()

    query = """
    SELECT 
        a.session_id, s.name AS student_name, c.name AS course_name, 
        a.status, a.session_date
    FROM 
        attendance a
    JOIN 
        students s ON a.student_id = s.id
    JOIN 
        courses c ON a.course_id = c.id
    WHERE 
        c.instructor_id = %s AND a.course_id = %s;
    """
    
    cursor.execute(query, (instructor_id, course_id))
    logs = cursor.fetchall()

    # Convert logs to a pandas DataFrame
    df = pd.DataFrame(logs, columns=['Session ID', 'Student Name', 'Course Name', 'Status', 'Session Date'])

    # Calculate the total absences and lates for each student
    summary = df.groupby('Student Name')['Status'].value_counts().unstack(fill_value=0)
    
    # Ensure 'absent' and 'late' columns exist even if there are no absences or lates
    if 'absent' not in summary.columns:
        summary['absent'] = 0
    if 'late' not in summary.columns:
        summary['late'] = 0

    # Add a summary row for each student
    summary['Summary'] = summary.apply(lambda row: f"Absent: {row['absent']}, Late: {row['late']}", axis=1)
    summary = summary[['Summary']].reset_index()

    # Create an in-memory BytesIO buffer to save the Excel file
    output = BytesIO()
    
    # Export logs and summary to Excel
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write logs
        df.to_excel(writer, index=False, sheet_name='Attendance Logs')
        # Write summary below logs (on the same sheet)
        summary_start_row = len(df) + 3  # Leave a few rows after logs
        summary.to_excel(writer, index=False, sheet_name='Attendance Logs', startrow=summary_start_row)

    output.seek(0)

    # Create a Flask response to download the file
    response = make_response(output.getvalue())
    #response.headers['Content-Disposition'] = 'attachment; filename=attendance_logs.xlsx'                   To save the excel file locally.
    response.headers['Content-type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    
    return response


@app.route('/student_logs/<course_id>', methods=['GET'])
def get_student_logs(course_id):
    student_id = session.get('user_id')  # Fetching the student ID from the session
    if student_id:
        connection = create_connection()
        cursor = connection.cursor()
        # Query to fetch attendance logs for the logged-in student for a specific course
        query = """
            SELECT session_id, session_date, status
            FROM attendance
            WHERE student_id = %s AND course_id = %s
        """
        cursor.execute(query, (student_id, course_id))
        logs = cursor.fetchall()
        cursor.close()
        
        # Return the fetched logs as a JSON response
        return jsonify({'status': 'success', 'logs': logs})
    else:
        return jsonify({'status': 'error', 'message': 'User not logged in'})


@app.route('/view_image/<int:image_id>')
def view_image(image_id):
    connection = create_connection()
    cursor = None
    try:
        cursor = connection.cursor()

        # Fetch the image binary data from the database
        sql_query = "SELECT image_data, image_name FROM attendance_images WHERE id = %s"
        cursor.execute(sql_query, (image_id,))
        record = cursor.fetchone()

        if record:
            image_data = record[0]  # Image binary data
            image_name = record[1]  # Image name

            # Send the image to the browser
            response = make_response(image_data)
            response.headers.set('Content-Type', 'image/jpg')  # Assuming the images are JPEG
            response.headers.set('Content-Disposition', 'inline', filename=image_name)
            return response
        else:
            return "Image not found", 404
    except Error as e:
        print(f"Error: {e}")
        return "Error retrieving image", 500
    finally:
        if cursor:
            cursor.close()
        if connection.is_connected():
            connection.close()


@app.route('/get_captured_images/<int:session_id>')
def get_captured_images(session_id):
    connection = create_connection()
    cursor = connection.cursor()

    # Fetch the image IDs for the given session
    query = "SELECT id FROM attendance_images WHERE session_id = %s"
    cursor.execute(query, (session_id,))
    captured_images = [image[0] for image in cursor.fetchall()]

    cursor.close()
    connection.close()

    # Return the list of image IDs as a JSON response
    return jsonify({'captured_images': captured_images})


    ##############################################################################################################################
    #                                                     Capture and Analyze Section                                            #
    ##############################################################################################################################

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def capture_and_recognize(course_id, session_id, session_duration):

    # Notify the user that the process has started
    socketio.emit('feedback', {'message': 'Starting the attendance process'})

    # Capture the first image (for attended students)
    #first_image_path = capture_image("captured_image_T1")
    socketio.emit('feedback', {'message': 'Capturing the first image...'})
    first_image_path = "1.png"

    if not first_image_path:
        socketio.emit('feedback', {'message': 'Failed to capture the first image.'})
        return {"error": "Failed to capture the first image."}

    # Resize the image to Full HD before saving
    first_image_path_resized = resize_image(first_image_path)

    # Save the first image to the database
    save_image_to_db(first_image_path_resized, session_id)

    socketio.emit('feedback', {'message': 'First image captured successfully.'})

    # Load ESRGAN model
    esrgan_model_path = 'ESRGAN/models/RRDB_ESRGAN_x4.pth'  # Path to the ESRGAN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    esrgan_model = load_esrgan_model(esrgan_model_path, device)

    # Create necessary directories
    output_dir = "processed_faces"
    ensure_directory(output_dir)
    
    # Use the original high-resolution image for YOLO detection
    image = cv2.imread(first_image_path)

    # Step 1: Run YOLO to detect regions of interest
    yolo_boxes = run_yolo(first_image_path)  # Call the function from the separate script

    # Step 2: Initialize recognition results dynamically from embeddings
    embeddings = load_course_embeddings('embeddings')  # Load embeddings to get student IDs
    recognition_results_T1 = {student_id: 0 for student_id in embeddings.keys()}  # Set all students as absent initially

    # Step 2: Process YOLO-detected regions to extract and recognize individual faces
    zoom_factor = 2  # Adjust the zoom factor as needed

    for i, box in enumerate(yolo_boxes):
        x, y, w, h = box

        # Ensure bounding box is within the image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        # Crop the detected region from the original high-resolution image
        cropped_person = image[y:y+h, x:x+w]

        # Create person-specific directory
        person_dir = os.path.join(output_dir, f"person_{i}")
        ensure_directory(person_dir)

        # Step 2.1: Save the cropped person image
        cropped_person_image_path = os.path.join(person_dir, "full_person.png")
        cv2.imwrite(cropped_person_image_path, cropped_person)

        # Step 3: Run RetinaFace to detect faces within the cropped person region
        face_boxes = detect_faces_retinaface(cropped_person)

        for j, face_box in enumerate(face_boxes):
            fx, fy, fw, fh = face_box

            # Ensure face bounding box is within the person image boundaries
            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(fw, cropped_person.shape[1] - fx)
            fh = min(fh, cropped_person.shape[0] - fy)

            # Create face-specific directory
            face_dir = os.path.join(person_dir, f"face_{j}")
            ensure_directory(face_dir)

            # Crop the face region from the person image
            cropped_face = cropped_person[fy:fy+fh, fx:fx+fw]

            # Step 3.1: Save the exact cropped face image as detected by RetinaFace
            cropped_face_path = os.path.join(face_dir, "1_cropped_face.png")
            cv2.imwrite(cropped_face_path, cropped_face)

            # Step 4: Apply scaling (zoom) to the cropped face region
            new_width = int(cropped_face.shape[1] * zoom_factor)
            new_height = int(cropped_face.shape[0] * zoom_factor)
            zoomed_cropped_face = cv2.resize(cropped_face, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
            # Save zoomed face
            zoomed_face_path = os.path.join(face_dir, "2_zoomed_face.png")
            cv2.imwrite(zoomed_face_path, zoomed_cropped_face)

            # Step 5: Enhance the cropped face using ESRGAN
            enhanced_face = apply_esrgan(zoomed_cropped_face, esrgan_model, device)
            torch.cuda.empty_cache()
        
            # Save enhanced face
            enhanced_face_path = os.path.join(face_dir, "3_enhanced_face.png")
            cv2.imwrite(enhanced_face_path, enhanced_face)

            # Step 6: Resize the enhanced face to a common size
            fixed_width = 512  # Adjust width as needed
            fixed_height = 512  # Adjust height as needed
            resized_face = cv2.resize(enhanced_face, (fixed_width, fixed_height), interpolation=cv2.INTER_LANCZOS4)

            # Step 7: Save the resized face
            final_face_path = os.path.join(face_dir, "4_final_face.png")
            cv2.imwrite(final_face_path, resized_face)

            # Recognize the individual face
            face_recognition_result = recognize_faces(final_face_path)

            # Update recognition results for each recognized student
            for student_id, status in face_recognition_result.items():
                if status == 1:
                    recognition_results_T1[student_id] = 1  # Mark as present if recognized

            # Log and save the recognition results
            results_path = os.path.join(face_dir, "recognition_results.txt")
            with open(results_path, 'w') as f:
                f.write(f"Recognition results for person_{i}_face_{j}:\n")
                f.write(str(face_recognition_result))

            # Log the results
            print(f"Recognition results for person_{i}_face_{j}: {face_recognition_result}")

    # Save all recognition results in a summary file
    summary_path = os.path.join(output_dir, "recognition_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Recognition Results Summary:\n")
        for student_id, status in recognition_results_T1.items():
            f.write(f"{student_id}: {status}\n")

    # Notify the user that the second image will be captured after a short delay
    socketio.emit('feedback', {'message': 'Waiting before capturing the second image...'})

    # Wait before capturing the second image (for late students)
    time.sleep(1) 

    #second_image_path = capture_image("captured_image_T2")
    socketio.emit('feedback', {'message': 'Capturing the second image...'})
    second_image_path = "2.png"
    if not second_image_path:
        socketio.emit('feedback', {'message': 'Failed to capture the second image.'})
        return {"error": "Failed to capture the second image."}

    # Resize the image to Full HD before saving
    second_image_path_resized = resize_image(second_image_path)

    # Save the second image to the database
    save_image_to_db(second_image_path_resized, session_id)

    socketio.emit('feedback', {'message': 'Second image captured successfully.'})

    # Use the original high-resolution image for YOLO detection
    image = cv2.imread(second_image_path)
    
    # Step 1: Run YOLO to detect regions of interest
    yolo_boxes = run_yolo(second_image_path)  # Call the function from the separate script

    recognition_results_T2 = {student_id: 0 for student_id in embeddings.keys()}  # Set all students as absent initially

    # Step 2: Process YOLO-detected regions to extract and recognize individual faces
    zoom_factor = 2  # Adjust the zoom factor as needed

    for i, box in enumerate(yolo_boxes):
        x, y, w, h = box

        # Ensure bounding box is within the image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        # Crop the detected region from the original high-resolution image
        cropped_person = image[y:y+h, x:x+w]

        # Create person-specific directory
        person_dir = os.path.join(output_dir, f"person_{i}")
        ensure_directory(person_dir)

        # Step 2.1: Save the cropped person image
        cropped_person_image_path = os.path.join(person_dir, "full_person.png")
        cv2.imwrite(cropped_person_image_path, cropped_person)

        # Step 3: Run RetinaFace to detect faces within the cropped person region
        face_boxes = detect_faces_retinaface(cropped_person)

        for j, face_box in enumerate(face_boxes):
            fx, fy, fw, fh = face_box

            # Ensure face bounding box is within the person image boundaries
            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(fw, cropped_person.shape[1] - fx)
            fh = min(fh, cropped_person.shape[0] - fy)

            # Create face-specific directory
            face_dir = os.path.join(person_dir, f"face_{j}")
            ensure_directory(face_dir)

            # Crop the face region from the person image
            cropped_face = cropped_person[fy:fy+fh, fx:fx+fw]

            # Step 3.1: Save the exact cropped face image as detected by RetinaFace
            cropped_face_path = os.path.join(face_dir, "1_cropped_face.png")
            cv2.imwrite(cropped_face_path, cropped_face)

            # Step 4: Apply scaling (zoom) to the cropped face region
            new_width = int(cropped_face.shape[1] * zoom_factor)
            new_height = int(cropped_face.shape[0] * zoom_factor)
            zoomed_cropped_face = cv2.resize(cropped_face, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
            # Save zoomed face
            zoomed_face_path = os.path.join(face_dir, "2_zoomed_face.png")
            cv2.imwrite(zoomed_face_path, zoomed_cropped_face)

            # Step 5: Enhance the cropped face using ESRGAN
            enhanced_face = apply_esrgan(zoomed_cropped_face, esrgan_model, device)
            torch.cuda.empty_cache()

            # Save enhanced face
            enhanced_face_path = os.path.join(face_dir, "3_enhanced_face.png")
            cv2.imwrite(enhanced_face_path, enhanced_face)

            # Step 6: Resize the enhanced face to a common size
            fixed_width = 512  # Adjust width as needed
            fixed_height = 512  # Adjust height as needed
            resized_face = cv2.resize(enhanced_face, (fixed_width, fixed_height), interpolation=cv2.INTER_LANCZOS4)

            # Step 7: Save the resized face
            final_face_path = os.path.join(face_dir, "4_final_face.png")
            cv2.imwrite(final_face_path, resized_face)

            # Recognize the individual face
            face_recognition_result = recognize_faces(final_face_path)

            # Update recognition results for each recognized student
            for student_id, status in face_recognition_result.items():
                if status == 1:
                    recognition_results_T2[student_id] = 1  # Mark as present if recognized

            # Log and save the recognition results
            results_path = os.path.join(face_dir, "recognition_results.txt")
            with open(results_path, 'w') as f:
                f.write(f"Recognition results for person_{i}_face_{j}:\n")
                f.write(str(face_recognition_result))

            # Log the results
            print(f"Recognition results for person_{i}_face_{j}: {face_recognition_result}")

    # Save all recognition results in a summary file
    summary_path = os.path.join(output_dir, "recognition_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Recognition Results Summary:\n")
        for student_id, status in recognition_results_T2.items():
            f.write(f"{student_id}: {status}\n")

    # Calculate remaining time before the session ends
    remaining_time_after_T2 = session_duration - 5  # session_duration is the total session time in seconds
    if remaining_time_after_T2 <= 0:
        return {"error": "Session duration too short to capture the third image."}

    # Random time for the third image between T2 and the end of the session
    random_wait_time = random.randint(0, remaining_time_after_T2)

    socketio.emit('feedback', {'message': f'Waiting {random_wait_time} seconds before capturing the third image...'})

    # Wait for a random amount of time before taking the third image
    time.sleep(random_wait_time)

    #third_image_path = capture_image("captured_image_T3")
    socketio.emit('feedback', {'message': 'Capturing the third image...'})
    third_image_path = "3.png"
    if not third_image_path:
        socketio.emit('feedback', {'message': 'Failed to capture the third image.'})
        return {"error": "Failed to capture the third image."}

    # Resize the image to Full HD before saving
    third_image_path_resized = resize_image(third_image_path)

    # Save the second image to the database
    save_image_to_db(third_image_path_resized, session_id)

    socketio.emit('feedback', {'message': 'Third image captured successfully.'})
    
    # Use the original high-resolution image for YOLO detection
    image = cv2.imread(third_image_path)

    # Step 1: Run YOLO to detect regions of interest
    yolo_boxes = run_yolo(third_image_path)  # Call the function from the separate script

    recognition_results_T3 = {student_id: 0 for student_id in embeddings.keys()}  # Set all students as absent initially

    # Step 2: Process YOLO-detected regions to extract and recognize individual faces
    zoom_factor = 2  # Adjust the zoom factor as needed

    for i, box in enumerate(yolo_boxes):
        x, y, w, h = box

        # Ensure bounding box is within the image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        # Crop the detected region from the original high-resolution image
        cropped_person = image[y:y+h, x:x+w]

        # Create person-specific directory
        person_dir = os.path.join(output_dir, f"person_{i}")
        ensure_directory(person_dir)

        # Step 2.1: Save the cropped person image
        cropped_person_image_path = os.path.join(person_dir, "full_person.png")
        cv2.imwrite(cropped_person_image_path, cropped_person)

        # Step 3: Run RetinaFace to detect faces within the cropped person region
        face_boxes = detect_faces_retinaface(cropped_person)

        for j, face_box in enumerate(face_boxes):
            fx, fy, fw, fh = face_box

            # Ensure face bounding box is within the person image boundaries
            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(fw, cropped_person.shape[1] - fx)
            fh = min(fh, cropped_person.shape[0] - fy)

            # Create face-specific directory
            face_dir = os.path.join(person_dir, f"face_{j}")
            ensure_directory(face_dir)

            # Crop the face region from the person image
            cropped_face = cropped_person[fy:fy+fh, fx:fx+fw]

            # Step 3.1: Save the exact cropped face image as detected by RetinaFace
            cropped_face_path = os.path.join(face_dir, "1_cropped_face.png")
            cv2.imwrite(cropped_face_path, cropped_face)

            # Step 4: Apply scaling (zoom) to the cropped face region
            new_width = int(cropped_face.shape[1] * zoom_factor)
            new_height = int(cropped_face.shape[0] * zoom_factor)
            zoomed_cropped_face = cv2.resize(cropped_face, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
            # Save zoomed face
            zoomed_face_path = os.path.join(face_dir, "2_zoomed_face.png")
            cv2.imwrite(zoomed_face_path, zoomed_cropped_face)

            # Step 5: Enhance the cropped face using ESRGAN
            enhanced_face = apply_esrgan(zoomed_cropped_face, esrgan_model, device)
            torch.cuda.empty_cache()

            # Save enhanced face
            enhanced_face_path = os.path.join(face_dir, "3_enhanced_face.png")
            cv2.imwrite(enhanced_face_path, enhanced_face)

            # Step 6: Resize the enhanced face to a common size
            fixed_width = 512  # Adjust width as needed
            fixed_height = 512  # Adjust height as needed
            resized_face = cv2.resize(enhanced_face, (fixed_width, fixed_height), interpolation=cv2.INTER_LANCZOS4)

            # Step 7: Save the resized face
            final_face_path = os.path.join(face_dir, "4_final_face.png")
            cv2.imwrite(final_face_path, resized_face)

            # Recognize the individual face
            face_recognition_result = recognize_faces(final_face_path)

            # Update recognition results for each recognized student
            for student_id, status in face_recognition_result.items():
                if status == 1:
                    recognition_results_T3[student_id] = 1  # Mark as present if recognized

            # Log and save the recognition results
            results_path = os.path.join(face_dir, "recognition_results.txt")
            with open(results_path, 'w') as f:
                f.write(f"Recognition results for person_{i}_face_{j}:\n")
                f.write(str(face_recognition_result))

            # Log the results
            print(f"Recognition results for person_{i}_face_{j}: {face_recognition_result}")

    # Save all recognition results in a summary file
    summary_path = os.path.join(output_dir, "recognition_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Recognition Results Summary:\n")
        for student_id, status in recognition_results_T3.items():
            f.write(f"{student_id}: {status}\n")

    # After the attendance process is completed
    socketio.emit('feedback', {'message': 'Attendance process completed successfully.'})

    # Classify students based on the first, second, and third images
    classified_results = classify_students(recognition_results_T1, recognition_results_T2, recognition_results_T3, course_id, session_id)

    if not classified_results:  # Check if classify_students returned None or empty dictionary
        return {"error": "Failed to classify students."}

    return classified_results



@app.route('/start_attendance', methods=['POST'])
def start_attendance():

    # Define session duration in seconds
    session_duration = 10

    # Get the course_id from the session (or POST data)
    course_id = session.get('selected_course_id') 
    
    # Create a new session and get the session_id
    session_id, session_date = create_session(course_id)  # Create a new session
    
    # Capture and recognize faces, classify students, but do NOT save to the database yet
    classified_results = capture_and_recognize(course_id, session_id, session_duration)

    if "error" in classified_results:  # Check for errors in the classification process
        return jsonify({"success": False, "message": classified_results["error"]})
    
    # Return the classified results to the frontend, but do NOT call process_and_store_attendance yet
    attended_count = len(classified_results["Attended"])
    late_count = len(classified_results["Late"])
    absent_count = len(classified_results["Absent"])
    out_of_class_count = len(classified_results["Out-of-class"])

    return jsonify({
        "success": True,
        "message": "Attendance process completed.",
        "attended": classified_results["Attended"],
        "late": classified_results["Late"],
        "absent": classified_results["Absent"],
        "out_of_class": classified_results["Out-of-class"],
        "attended_count": attended_count,
        "late_count": late_count,
        "absent_count": absent_count,
        "out_of_class_count": out_of_class_count,
        "session_id": session_id  # Send session_id back to the frontend for future use
    })




if __name__ == '__main__':
    #app.run(host = '0.0.0.0', port = 5000)
    socketio.run(app, debug=True)
