import os
import re
import json
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import joblib
from train_model import load_dataset, train_text_classifier, predict_emotion

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.secret_key = 'your-secret-key-here'

# Global variables
model = None
vectorizer = None
training_progress = {"status": "idle", "message": ""}
STUDENTS_FILE = "students.json"
TEACHERS_FILE = "teachers.json"

# Load existing model if available
try:
    if os.path.exists("text_classifier.joblib") and os.path.exists("tfidf_vectorizer.joblib"):
        model = joblib.load("text_classifier.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
except:
    pass


def load_students():
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_students(students):
    with open(STUDENTS_FILE, 'w') as f:
        json.dump(students, f, indent=2)


def load_teachers():
    if os.path.exists(TEACHERS_FILE):
        with open(TEACHERS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_teachers(teachers):
    with open(TEACHERS_FILE, 'w') as f:
        json.dump(teachers, f, indent=2)


def validate_name(name):
    return len(name.strip()) >= 2 and len(name) <= 50


def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone):
    pattern = r'^[0-9\-\+\(\)\s]{10,}$'
    return re.match(pattern, phone) is not None


def validate_class(class_name):
    valid_classes = ['9', '10', '11', '12', 'BCA', 'BTech', 'MBA', 'B.Sc', 'B.Com']
    return class_name in valid_classes


def validate_subject(subject):
    valid_subjects = ['Mathematics', 'English', 'Science', 'History', 'Geography', 'Computer Science', 'Physics', 'Chemistry', 'Biology', 'Economics']
    return subject in valid_subjects


@app.route('/')
def index():
    global model, vectorizer
    model_loaded = model is not None and vectorizer is not None
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/students')
def students_list():
    students = load_students()
    return render_template('students.html', students=students)


@app.route('/teacher-register')
def teacher_register_page():
    return render_template('teacher_register.html')


@app.route('/teachers')
def teachers_list():
    teachers = load_teachers()
    return render_template('teachers.html', teachers=teachers)


@app.route('/api/register', methods=['POST'])
def register_student():
    data = request.json
    
    # Validation
    errors = {}
    
    name = data.get('name', '').strip()
    if not name:
        errors['name'] = 'Name is required'
    elif not validate_name(name):
        errors['name'] = 'Name must be 2-50 characters'
    
    email = data.get('email', '').strip().lower()
    if not email:
        errors['email'] = 'Email is required'
    elif not validate_email(email):
        errors['email'] = 'Invalid email format'
    
    phone = data.get('phone', '').strip()
    if not phone:
        errors['phone'] = 'Phone number is required'
    elif not validate_phone(phone):
        errors['phone'] = 'Invalid phone number (10+ digits)'
    
    class_name = data.get('class', '').strip()
    if not class_name:
        errors['class'] = 'Class is required'
    elif not validate_class(class_name):
        errors['class'] = 'Invalid class'
    
    if errors:
        return jsonify({'status': 'error', 'errors': errors}), 400
    
    # Check for duplicate email
    students = load_students()
    if any(s['email'] == email for s in students):
        return jsonify({'status': 'error', 'message': 'Email already registered'}), 400
    
    # Add new student
    student = {
        'id': len(students) + 1,
        'name': name,
        'email': email,
        'phone': phone,
        'class': class_name,
        'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    students.append(student)
    save_students(students)
    
    return jsonify({'status': 'success', 'message': 'Student registered successfully!', 'student': student})


@app.route('/api/register-teacher', methods=['POST'])
def register_teacher():
    data = request.json
    
    errors = {}
    
    name = data.get('name', '').strip()
    if not name:
        errors['name'] = 'Name is required'
    elif not validate_name(name):
        errors['name'] = 'Name must be 2-50 characters'
    
    email = data.get('email', '').strip().lower()
    if not email:
        errors['email'] = 'Email is required'
    elif not validate_email(email):
        errors['email'] = 'Invalid email format'
    
    phone = data.get('phone', '').strip()
    if not phone:
        errors['phone'] = 'Phone is required'
    elif not validate_phone(phone):
        errors['phone'] = 'Invalid phone number (10+ digits)'
    
    subject = data.get('subject', '').strip()
    if not subject:
        errors['subject'] = 'Subject is required'
    elif not validate_subject(subject):
        errors['subject'] = 'Invalid subject'
    
    if errors:
        return jsonify({'status': 'error', 'errors': errors}), 400
    
    # Check for duplicate email
    teachers = load_teachers()
    if any(t['email'] == email for t in teachers):
        return jsonify({'status': 'error', 'message': 'Email already registered'}), 400
    
    # Add new teacher
    teacher = {
        'id': len(teachers) + 1,
        'name': name,
        'email': email,
        'phone': phone,
        'subject': subject,
        'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    teachers.append(teacher)
    save_teachers(teachers)
    
    return jsonify({'status': 'success', 'message': 'Teacher registered successfully!', 'teacher': teacher})


@app.route('/api/student/<int:student_id>', methods=['GET'])
def get_student(student_id):
    students = load_students()
    student = next((s for s in students if s['id'] == student_id), None)
    
    if not student:
        return jsonify({'status': 'error', 'message': 'Student not found'}), 404
    
    return jsonify({'status': 'success', 'student': student})


@app.route('/api/student/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    students = load_students()
    students = [s for s in students if s['id'] != student_id]
    save_students(students)
    
    return jsonify({'status': 'success', 'message': 'Student deleted'})


@app.route('/api/students', methods=['GET'])
def get_all_students():
    students = load_students()
    return jsonify({'status': 'success', 'students': students, 'total': len(students)})


@app.route('/api/teachers', methods=['GET'])
def get_all_teachers():
    teachers = load_teachers()
    return jsonify({'status': 'success', 'teachers': teachers, 'total': len(teachers)})


@app.route('/api/teacher/<int:teacher_id>', methods=['DELETE'])
def delete_teacher(teacher_id):
    teachers = load_teachers()
    teachers = [t for t in teachers if t['id'] != teacher_id]
    save_teachers(teachers)
    
    return jsonify({'status': 'success', 'message': 'Teacher deleted'})


# Emotion Classifier Routes
def train_api():
    global model, vectorizer, training_progress
    
    data = request.json
    data_path = data.get('data_file', 'train.txt')
    model_path = data.get('model_file', 'text_classifier.joblib')
    vectorizer_path = data.get('vectorizer_file', 'tfidf_vectorizer.joblib')
    
    if not os.path.exists(data_path):
        return jsonify({'status': 'error', 'message': f'File not found: {data_path}'}), 400
    
    def train_background():
        global model, vectorizer, training_progress
        try:
            training_progress['status'] = 'training'
            training_progress['message'] = 'Loading data...'
            
            df = load_dataset(data_path)
            training_progress['message'] = f'Loaded {len(df)} records. Training...'
            
            model, vectorizer = train_text_classifier(data_path, model_path, vectorizer_path)
            
            training_progress['status'] = 'success'
            training_progress['message'] = 'Training completed successfully!'
        except Exception as e:
            training_progress['status'] = 'error'
            training_progress['message'] = str(e)
    
    thread = threading.Thread(target=train_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Training started in background'})


@app.route('/api/training-status', methods=['GET'])
def training_status():
    return jsonify(training_progress)


@app.route('/api/predict', methods=['POST'])
def predict_api():
    global model, vectorizer
    
    if model is None or vectorizer is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded. Train first!'}), 400
    
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'status': 'error', 'message': 'Text is required'}), 400
    
    try:
        prediction = predict_emotion(text, model, vectorizer)
        return jsonify({'status': 'success', 'prediction': prediction, 'text': text})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/model-status', methods=['GET'])
def model_status():
    global model, vectorizer
    loaded = model is not None and vectorizer is not None
    return jsonify({'loaded': loaded})


if __name__ == '__main__':
    print("🚀 Starting Emotion Classifier Web Server...")
    print("📱 Open your browser at: http://localhost:5000")
    app.run(debug=True, port=5000)
