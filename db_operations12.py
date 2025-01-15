import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash
import numpy as np
import pickle
from datetime import datetime

def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host="192.168.1.109",
            user="remote_user",
            password="12345-Qwert",
            database="attendance_system",
        )
        if connection.is_connected():
            print("Connection to MySQL DB successful")
        else:
            print("Failed to connect to the MySQL DB")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection


def get_instructor(email):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT id, name, email, password FROM instructors WHERE email = %s", (email,))
        instructor = cursor.fetchone()
        return instructor
    except Error as e:
        print(f"The error '{e}' occurred")
        return None
    finally:
        cursor.close()
        connection.close()


def get_courses(instructor_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT id, name FROM courses WHERE instructor_id = %s", (instructor_id,))
        courses = cursor.fetchall()
        return courses
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        cursor.close()
        db.close()


def get_student(email):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT id, name, email, password FROM students WHERE email = %s", (email,))
        student = cursor.fetchone()
        return student
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        cursor.close()
        db.close()


def get_enrolled_courses(student_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT c.id, c.name 
            FROM courses c 
            JOIN course_students cs ON c.id = cs.course_id 
            WHERE cs.student_id = %s
        """, (student_id,))
        courses = cursor.fetchall()
        return courses
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        cursor.close()
        db.close()


def get_all_courses():
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT id, name FROM courses")
        courses = cursor.fetchall()
        return courses
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        cursor.close()
        db.close()


def get_students_in_course(course_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT s.id, s.name 
            FROM students s 
            JOIN course_students cs ON s.id = cs.student_id 
            WHERE cs.course_id = %s
        """, (course_id,))
        students = cursor.fetchall()
        return [{'id': student[0], 'name': student[1]} for student in students]
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        cursor.close()
        db.close()


def get_instructor_for_course(course_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("""
            SELECT i.name 
            FROM instructors i
            JOIN courses c ON i.id = c.instructor_id
            WHERE c.id = %s
        """, (course_id,))
        instructor = cursor.fetchone()
        return instructor[0] if instructor else None
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        cursor.close()
        db.close()


def add_instructor(name, email, password):
    db = create_connection()
    cursor = db.cursor()
    hashed_password = generate_password_hash(password)
    try:
        cursor.execute(
            "INSERT INTO instructors (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        db.commit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        db.close()


def add_course(instructor_id, course_name):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "SELECT id FROM courses WHERE name = %s AND instructor_id = %s",
            (course_name, instructor_id)
        )
        existing_course = cursor.fetchone()
        if existing_course:
            return None

        cursor.execute(
            "INSERT INTO courses (name, instructor_id) VALUES (%s, %s)",
            (course_name, instructor_id)
        )
        db.commit()
        course_id = cursor.lastrowid
        return course_id
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        cursor.close()
        db.close()


def add_student(name, email, password):
    db = create_connection()
    cursor = db.cursor()
    hashed_password = generate_password_hash(password)
    try:
        cursor.execute(
            "INSERT INTO students (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        db.commit()
    except mysql.connector.IntegrityError as e:
        if e.errno == 1062:
            print("Error: Email already exists")
            return "Email already exists"
        else:
            print(f"Error: {e}")
            return "An error occurred"
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred"
    finally:
        cursor.close()
        db.close()


def enroll_student_in_course(student_id, course_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute(
            "SELECT 1 FROM course_students WHERE student_id = %s AND course_id = %s",
            (student_id, course_id)
        )
        if cursor.fetchone():
            return "Already enrolled"

        cursor.execute(
            "INSERT INTO course_students (student_id, course_id) VALUES (%s, %s)",
            (student_id, course_id)
        )
        db.commit()
        return "Enrolled successfully"
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred"
    finally:
        cursor.close()
        db.close()


def remove_course_from_db(course_id, instructor_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT 1 FROM courses WHERE id = %s AND instructor_id = %s", (course_id, instructor_id))
        if cursor.fetchone():
            cursor.execute("DELETE FROM course_students WHERE course_id = %s", (course_id,))
            cursor.execute("DELETE FROM courses WHERE id = %s AND instructor_id = %s", (course_id, instructor_id))
            db.commit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        db.close()


def remove_student_from_course(student_id, course_id, instructor_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("""
            DELETE cs FROM course_students cs
            JOIN courses c ON cs.course_id = c.id
            WHERE cs.student_id = %s AND cs.course_id = %s AND c.instructor_id = %s
        """, (student_id, course_id, instructor_id))
        db.commit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        db.close()


def save_embeddings_to_db(student_id, embeddings):
    db = create_connection()
    cursor = db.cursor()
    for embedding in embeddings:
        # Convert the numpy array to binary
        embedding_binary = np.array(embedding).tobytes()
        cursor.execute("INSERT INTO embeddings (student_id, embedding) VALUES (%s, %s)", (student_id, embedding_binary))
    db.commit()
    cursor.close()
    db.close()


def get_course_name_by_id(course_id):
    connection = create_connection()  # Ensure this is creating a MySQL connection
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM courses WHERE id = %s", (course_id,))
    course_name = cursor.fetchone()[0]
    cursor.close()
    connection.close()
    return course_name


def get_student_name_by_id(student_id):
    connection = create_connection()  # Ensure this is creating a MySQL connection
    cursor = connection.cursor()
    cursor.execute("SELECT name FROM students WHERE id = %s", (student_id,))
    student_name = cursor.fetchone()[0]
    return student_name


def get_student_id_by_name(student_name):
    connection = create_connection()
    cursor = connection.cursor()
    student_id = None
    try:
        query = "SELECT id FROM students WHERE name = %s"
        cursor.execute(query, (student_name,))
        result = cursor.fetchone()
        if result:
            student_id = result[0]
    except Exception as e:
        print(f"Error retrieving student ID: {e}")
    finally:
        cursor.close()
        connection.close()
    
    return student_id


def enroll_student_in_course(student_id, course_id):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Check if the student is already enrolled
        cursor.execute("SELECT * FROM course_students WHERE student_id = %s AND course_id = %s", (student_id, course_id))
        result = cursor.fetchone()

        if result:
            return "Already enrolled"

        # If not enrolled, insert the student into course_students table
        cursor.execute("INSERT INTO course_students (student_id, course_id) VALUES (%s, %s)", (student_id, course_id))
        connection.commit()

        # After successfully enrolling, return a success message
        return "Enrolled successfully"

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return f"Error during enrollment: {err}"

    finally:
        cursor.close()
        connection.close()


def check_if_already_enrolled(student_id, course_id):
    db = create_connection()
    cursor = db.cursor()
    try:
        cursor.execute("SELECT * FROM course_students WHERE student_id = %s AND course_id = %s", (student_id, course_id))
        result = cursor.fetchone()
        return result is not None
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False
    finally:
        cursor.close()
        db.close()


def update_attendance_status(student_id, status):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        query = "INSERT INTO attendance (student_id, status, date) VALUES (%s, %s, CURDATE()) ON DUPLICATE KEY UPDATE status = %s"
        cursor.execute(query, (student_id, status, status))
        connection.commit()
    except Exception as e:
        print(f"Error updating attendance status: {e}")
    finally:
        cursor.close()
        connection.close()


# Store embeddings for the student (just once)
def save_student_embeddings_to_db(student_id, all_embeddings):
    connection = create_connection()
    cursor = connection.cursor()

    # Assuming all_embeddings is a list of embeddings
    for embedding in all_embeddings:
        embedding_binary = pickle.dumps(embedding)
        cursor.execute(
            "INSERT INTO student_embeddings (student_id, embedding) VALUES (%s, %s)",
            (student_id, embedding_binary)
        )

    connection.commit()
    cursor.close()
    connection.close()


def insert_attendance(student_id, course_id, session_id, status):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        query = """
        INSERT INTO attendance (student_id, course_id, session_id, status, session_date)
        VALUES (%s, %s, %s, %s, NOW())
        """
        cursor.execute(query, (student_id, course_id, session_id, status))
        connection.commit()
        print(f"Attendance record for student {student_id} inserted successfully.")
        print(f"Session ID being passed: {session_id}")
    except Error as e:
        print(f"Error: '{e}'")
    finally:
        cursor.close()
        connection.close()


def process_and_store_attendance(course_id, session_id, results):
    print(f"Processing session_id: {session_id}")  # Debug print to verify session_id
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Process each category of attendance
        for status, students in results.items():
            for student_path in students:
                student_id = student_path.split("\\")[-1]  # Extract student ID from the path
                print(f"Inserting attendance for student {student_id} with session_id {session_id}")  # Debug print
                insert_attendance(student_id, course_id, session_id, status.lower())  # Pass session_id

    except Error as e:
        print(f"Error storing attendance: {e}")
    finally:
        cursor.close()
        connection.close()


def create_session(course_id):
    connection = create_connection()
    cursor = connection.cursor()

    try:
        # Get the current date and time
        session_date = datetime.now()

        # Insert a new session and get the session_id
        query = """
        INSERT INTO sessions (course_id, session_date)
        VALUES (%s, %s)
        """
        cursor.execute(query, (course_id, session_date))
        connection.commit()

        session_id = cursor.lastrowid  # Get the generated session_id
        print(f"Generated session_id: {session_id}")  # Print session_id for debugging

        return session_id, session_date

    except Error as e:
        print(f"Error creating session: {e}")
        return None, None

    finally:
        cursor.close()
        connection.close()


def update_attendance_in_db(connection, student_id, status, course_id, session_id):
    try:
        cursor = connection.cursor()
        # Update the status of the student for the given course and session
        cursor.execute("""
            UPDATE attendance
            SET status = %s
            WHERE student_id = %s AND course_id = %s AND session_id = %s
        """, (status, student_id, course_id, session_id))
        
        connection.commit()
        cursor.close()
    except Exception as e:
        print(f"Error updating attendance: {e}")


def save_image_to_db(image_path, session_id):
    connection = create_connection()
    cursor = None
    try:
        # Read the image file as binary
        with open(image_path, 'rb') as file:
            binary_data = file.read()

        # Insert image data into the database
        sql_query = """
        INSERT INTO attendance_images (session_id, image_data, image_name)
        VALUES (%s, %s, %s)
        """
        image_name = image_path.split('/')[-1]  # Extracts the image name from the path
        cursor = connection.cursor()
        cursor.execute(sql_query, (session_id, binary_data, image_name))
        connection.commit()

        print(f"Image '{image_name}' has been successfully saved to the database.")
    except Error as e:
        print(f"Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection.is_connected():
            connection.close()


def get_image_from_db(image_id):
    connection = create_connection()
    cursor = None
    try:
        cursor = connection.cursor()

        # Select the image data from the database
        sql_query = "SELECT image_data, image_name FROM attendance_images WHERE id = %s"
        cursor.execute(sql_query, (image_id,))
        record = cursor.fetchone()

        if record:
            image_data = record[0]
            image_name = record[1]

            # Save the image to a file (optional)
            with open(f'retrieved_{image_name}', 'wb') as file:
                file.write(image_data)

            print(f"Image '{image_name}' has been successfully retrieved from the database.")
        else:
            print(f"No image found with ID: {image_id}")
    except Error as e:
        print(f"Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection.is_connected():
            connection.close()


def get_most_recent_session(course_id):
    connection = create_connection()
    cursor = connection.cursor()
    try:
        query = """
        SELECT id 
        FROM sessions
        WHERE course_id = %s
        ORDER BY session_date DESC
        LIMIT 1
        """
        cursor.execute(query, (course_id,))
        result = cursor.fetchone()
        if result:
            return result[0]  # Return the session ID
        else:
            return None  # No sessions found for the course
    except mysql.connector.Error as e:
        print(f"Error retrieving the most recent session: {e}")
        return None
    finally:
        cursor.close()
        connection.close()
