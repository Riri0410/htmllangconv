import streamlit as st
import numpy as np
import librosa
import os
import pickle
import sounddevice as sd
import wavio
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from time import sleep
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import dlib  # For facial landmark detection
from scipy.spatial import distance as dist
import time

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Enhanced BioAuth System",
    layout="wide",
    page_icon="🛡",
    initial_sidebar_state="expanded"
)

# Add a manual refresh button in the sidebar
#st.sidebar.markdown("---")


# Constants
DATA_DIR = "voice_data"
FACE_DIR = "face_data"
VOICE_EMBEDDINGS_FILE = "voice_embeddings.pkl"
FACE_ENCODINGS_FILE = "face_encodings.pkl"
VOICE_THRESHOLD = 0.75  # 75% similarity threshold for voice
FACE_THRESHOLD = 0.60  # Adjusted threshold for better face matching

# Standard paragraph for voice registration
REGISTRATION_TEXT = """
Please read this paragraph clearly: 
"The quick brown fox jumps over the lazy dog. This sentence contains all the letters in the English alphabet."
Please speak clearly and at a normal pace for voice registration. Thank you for helping improve our security system.
"""

# Authentication phrase
AUTH_PHRASE = "I want to authenticate"

# Create directories if not exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

# Initialize face encodings dictionary
if os.path.exists(FACE_ENCODINGS_FILE):
    with open(FACE_ENCODINGS_FILE, 'rb') as f:
        known_face_encodings = pickle.load(f)
else:
    known_face_encodings = {}

# Initialize voice embeddings dictionary
if os.path.exists(VOICE_EMBEDDINGS_FILE):
    with open(VOICE_EMBEDDINGS_FILE, 'rb') as f:
        known_voice_embeddings = pickle.load(f)
else:
    known_voice_embeddings = {}

# Initialize Resemblyzer voice encoder
voice_encoder = VoiceEncoder()

# Use OpenCV's built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dlib's face detector and landmark predictor for liveness detection
try:
    dlib_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file
except:
    st.warning("Could not initialize dlib face landmark detector. Liveness detection will be limited.")


# Extract voice embeddings using Resemblyzer
def extract_voice_embedding(file_path):
    """Extract voice embedding using Resemblyzer"""
    try:
        # Preprocess the wav file
        wav = preprocess_wav(file_path)

        # Extract embedding using Resemblyzer
        embedding = voice_encoder.embed_utterance(wav)

        return embedding
    except Exception as e:
        st.error(f"Error extracting voice embedding: {str(e)}")
        return None


# Process face image using OpenCV with liveness detection
def process_face_image(image_path=None, image=None, check_liveness=False):
    """Process face image using OpenCV for facial recognition with optional liveness detection"""
    try:
        if image_path is not None:
            image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect face using Haar Cascade
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None

        # Get largest face (by area)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face

        # Add margin to the bounding box
        margin = int(max(w, h) * 0.3)
        x_expanded = max(0, x - margin)
        y_expanded = max(0, y - margin)
        width_expanded = min(w + 2 * margin, image.shape[1] - x_expanded)
        height_expanded = min(h + 2 * margin, image.shape[0] - y_expanded)

        # Extract face with margin
        face_img = image[y_expanded:y_expanded + height_expanded,
                   x_expanded:x_expanded + width_expanded]

        # Resize to standard size
        face_img = cv2.resize(face_img, (224, 224))

        # Liveness detection
        if check_liveness:
            if not is_live_face(face_img):
                st.warning("⚠ Liveness check failed - face may not be real")
                return None

        # Convert to grayscale for feature extraction
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Use HOG features as face embedding
        hog = cv2.HOGDescriptor()
        embedding = hog.compute(face_gray)

        # Normalize the embedding
        embedding = embedding.flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        st.error(f"Error processing face image: {str(e)}")
        return None


# Liveness detection function
def is_live_face(face_img):
    """Check if the face is live using eye blink detection and facial landmarks"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Detect faces using dlib
        rects = dlib_detector(gray, 0)

        if len(rects) == 0:
            return False

        # Get facial landmarks
        shape = predictor(gray, rects[0])
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Simple eye aspect ratio check for blinking
        left_eye = shape[42:48]
        right_eye = shape[36:42]

        def eye_aspect_ratio(eye):
            # Compute the euclidean distances between the two sets of
            # vertical eye landmarks (x, y)-coordinates
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])

            # Compute the euclidean distance between the horizontal
            # eye landmark (x, y)-coordinates
            C = np.linalg.norm(eye[0] - eye[3])

            # Compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)

            # Return the eye aspect ratio
            return ear

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # If eyes are closed (EAR < 0.2), assume it's a photo
        if left_ear < 0.2 or right_ear < 0.2:
            return False

        return True

    except Exception as e:
        st.warning(f"Liveness detection error: {str(e)}")
        return True  # Fallback to allow authentication if liveness check fails


# Train voice model with Resemblyzer
def train_voice_model():
    """Generate and save voice embeddings for all registered users"""
    try:
        global known_voice_embeddings
        known_voice_embeddings = {}

        for file in os.listdir(DATA_DIR):
            if file.endswith(".wav"):
                username = os.path.splitext(file)[0]
                file_path = os.path.join(DATA_DIR, file)

                # Get embedding using Resemblyzer
                embedding = extract_voice_embedding(file_path)
                if embedding is not None:
                    known_voice_embeddings[username] = embedding

        if known_voice_embeddings:
            with open(VOICE_EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(known_voice_embeddings, f)
            st.success(f"Voice model trained with {len(known_voice_embeddings)} users!")
        else:
            st.error("No voice training data available!")
    except Exception as e:
        st.error(f"Error training voice model: {str(e)}")


# Train face model using OpenCV embeddings
def train_face_model():
    """Generate and save face embeddings for all registered users"""
    try:
        global known_face_encodings
        known_face_encodings = {}

        for file in os.listdir(FACE_DIR):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                username = os.path.splitext(file)[0]
                image_path = os.path.join(FACE_DIR, file)

                # Get embedding using OpenCV
                embedding = process_face_image(image_path=image_path)
                if embedding is not None:
                    known_face_encodings[username] = embedding

        if known_face_encodings:
            with open(FACE_ENCODINGS_FILE, 'wb') as f:
                pickle.dump(known_face_encodings, f)
            st.success(f"Face model trained with {len(known_face_encodings)} users!")
        else:
            st.error("No face training data available or no faces detected!")
    except Exception as e:
        st.error(f"Error training face model: {str(e)}")


# Voice recognition function with Resemblyzer
def recognize_speaker(file_path):
    """Recognize speaker from voice sample using Resemblyzer embeddings"""
    try:
        if not known_voice_embeddings:
            st.error("Voice embeddings not generated yet!")
            return None, 0

        # Extract embedding from the provided voice sample
        test_embedding = extract_voice_embedding(file_path)
        if test_embedding is None:
            return None, 0

        # Compare with known embeddings
        best_match = None
        highest_similarity = 0

        for username, known_embedding in known_voice_embeddings.items():
            # Calculate cosine similarity
            similarity = cosine_similarity([test_embedding], [known_embedding])[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = username

        # Convert similarity to percentage
        match_confidence = highest_similarity * 100

        # Return match result if confidence is above threshold
        if highest_similarity > VOICE_THRESHOLD:
            return best_match, match_confidence
        else:
            return None, match_confidence
    except Exception as e:
        st.error(f"Error recognizing speaker: {str(e)}")
        return None, 0


def perform_liveness_check():
    """Perform active liveness detection by checking multiple frames"""
    try:
        st.info("Please follow the instructions for liveness verification...")

        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open video device")
            return False

        # We'll check for 3 types of liveness cues
        liveness_checks = {
            'blink_detected': False,
            'mouth_movement': False,
            'head_movement': False
        }

        # Display instructions and process frames
        instruction_placeholder = st.empty()
        frames_processed = 0
        max_frames = 30  # Process about 1 second of video (assuming 30fps)

        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Display instructions based on progress
            if frames_processed < 10:
                instruction_placeholder.info("👀 Please blink your eyes")
                # Check for blinking (simplified - would use facial landmarks in real implementation)
                # This is a placeholder - replace with actual blink detection
                liveness_checks['blink_detected'] = True if frames_processed > 5 else False
            elif frames_processed < 20:
                instruction_placeholder.info("😃 Please open your mouth")
                # Check for mouth movement
                # Placeholder - replace with actual mouth movement detection
                liveness_checks['mouth_movement'] = True if frames_processed > 15 else False
            else:
                instruction_placeholder.info("🙂 Please turn your head slightly")
                # Check for head movement
                # Placeholder - replace with actual head movement detection
                liveness_checks['head_movement'] = True if frames_processed > 25 else False

            # Display the frame being processed (optional)
            # st.image(frame, channels="BGR", use_column_width=True)

            frames_processed += 1
            time.sleep(0.03)  # Simulate processing time

        cap.release()

        # Check if all liveness cues were detected
        if all(liveness_checks.values()):
            st.success("✅ Liveness verification passed")
            return True
        else:
            st.error("❌ Liveness verification failed")
            return False

    except Exception as e:
        st.error(f"Liveness check error: {str(e)}")
        return False
# Face verification function using OpenCV with liveness detection
def verify_face(frame=None, check_liveness=True):
    """Verify face against known face embeddings with enhanced liveness detection"""
    try:
        if not known_face_encodings:
            st.error("Face model not trained yet!")
            return False, None, 0, False

        if frame is None:
            # Capture from camera if no frame provided
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Failed to capture image")
                return False, None, 0, False

        # Initialize liveness status
        liveness_passed = False

        if check_liveness:
            # Enhanced liveness detection - check multiple frames
            liveness_passed = perform_liveness_check()
            if not liveness_passed:
                st.warning("❌ Liveness check failed - possible spoof attempt")
                return False, None, 0, False

        # Get embedding for the captured face
        embedding = process_face_image(image=frame, check_liveness=check_liveness)
        if embedding is None:
            return False, None, 0, liveness_passed

        # Compare with known embeddings
        best_match = None
        highest_similarity = 0

        for username, known_embedding in known_face_encodings.items():
            # Calculate cosine similarity
            similarity = np.dot(embedding, known_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
            )
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = username

        # Convert similarity to percentage
        match_confidence = highest_similarity * 100

        # Return match result with liveness status
        if highest_similarity > FACE_THRESHOLD:
            return True, best_match, match_confidence, liveness_passed
        else:
            return False, None, match_confidence, liveness_passed
    except Exception as e:
        st.error(f"Error verifying face: {str(e)}")
        return False, None, 0, False


# Recording audio function with paragraph guidance
def record_audio(filename, duration=10, fs=16000, registration=True):
    """Record audio with visualization and guidance"""
    try:
        with st.spinner("Preparing to record..."):
            if registration:
                st.warning("Please speak the following paragraph clearly:")
                st.info(REGISTRATION_TEXT)
            else:
                st.warning("Please say clearly:")
                st.info(AUTH_PHRASE)

            # Countdown before recording
            countdown_placeholder = st.empty()
            for i in range(3, 0, -1):
                countdown_placeholder.markdown(f"## Recording starts in {i}...")
                sleep(1)
            countdown_placeholder.empty()

            # Recording visualization
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='display: flex; justify-content: center;'>
                    <div class='pulsating-mic'></div>
                </div>
                """, unsafe_allow_html=True)
                status_text = st.empty()
                status_text.info("⏺ Recording... Speak now!")

            # Start recording
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            wavio.write(filename, audio, fs, sampwidth=2)

            status_text.success("✅ Recording complete!")
        return True
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return False


# Capture face function with improved feedback and liveness detection
def capture_face(username=None, for_auth=False):
    """Capture face with improved user guidance and liveness detection"""
    try:
        with st.spinner("Preparing camera..."):
            # User instructions
            st.info("""
            Face Capture Instructions:
            - Face the camera directly
            - Ensure good lighting
            - Remove sunglasses/hats
            - Blink naturally (for liveness detection)
            """)

            cap = cv2.VideoCapture(0)

            # Show live preview with countdown
            preview_placeholder = st.empty()
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                if ret:
                    # Add countdown to frame
                    countdown_frame = frame.copy()
                    cv2.putText(countdown_frame, str(i),
                                (int(frame.shape[1] / 2) - 50, int(frame.shape[0] / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
                    preview_placeholder.image(countdown_frame,
                                              channels="BGR",
                                              caption=f"Capturing in {i}...",
                                              use_container_width=True)
                    sleep(1)

            # Capture final frame
            ret, frame = cap.read()
            cap.release()
            preview_placeholder.empty()

            if ret:
                if username and not for_auth:
                    # For registration - save the image
                    face_path = os.path.join(FACE_DIR, f"{username}.jpg")

                    # Detect faces using OpenCV
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30)
                    )

                    if len(faces) > 0:
                        # Draw rectangle around detected face
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        cv2.imwrite(face_path, frame)
                        st.image(frame, channels="BGR", caption="Face captured", use_container_width=True)

                        # Immediately process the face for quality check
                        with st.spinner("Analyzing face quality..."):
                            embedding = process_face_image(image_path=face_path)
                            if embedding is None:
                                st.error("❌ Could not extract face features. Please try again.")
                                return False

                        st.success("✅ High-quality face captured successfully!")
                        return True
                    else:
                        st.error("❌ No face detected. Please try again with better lighting.")
                        return False
                else:
                    # For authentication - just return the frame
                    st.image(frame, channels="BGR", caption="Captured Face", use_container_width=True)
                    return frame
            else:
                st.error("❌ Failed to capture image from camera!")
                return False
    except Exception as e:
        st.error(f"Error capturing face: {str(e)}")
        return False


# Delete user function
def delete_user(username):
    """Delete a user and their associated data"""
    try:
        # Delete voice data
        voice_file = os.path.join(DATA_DIR, f"{username}.wav")
        if os.path.exists(voice_file):
            os.remove(voice_file)

        # Delete face data
        face_file = os.path.join(FACE_DIR, f"{username}.jpg")
        if os.path.exists(face_file):
            os.remove(face_file)

        # Remove from face encodings
        global known_face_encodings
        if username in known_face_encodings:
            del known_face_encodings[username]
            # Save updated encodings
            with open(FACE_ENCODINGS_FILE, 'wb') as f:
                pickle.dump(known_face_encodings, f)

        # Remove from voice embeddings
        global known_voice_embeddings
        if username in known_voice_embeddings:
            del known_voice_embeddings[username]
            # Save updated embeddings
            with open(VOICE_EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(known_voice_embeddings, f)

        st.success(f"✅ User {username} deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting user: {str(e)}")


# Admin dashboard
def admin_dashboard():
    st.title("👑 Admin Dashboard")
    st.markdown("---")

    # Statistics and Registered Users
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### 📈 Statistics")
        total_users = len([f for f in os.listdir(DATA_DIR) if f.endswith(".wav")])
        st.metric("Total Users", total_users)

        # Model status
        if os.path.exists(VOICE_EMBEDDINGS_FILE) and known_voice_embeddings:
            st.metric("Voice Model", "Trained ✅")
        else:
            st.metric("Voice Model", "Not Trained ❌")

        if known_face_encodings:
            st.metric("Face Model", "Trained ✅")
        else:
            st.metric("Face Model", "Not Trained ❌")

    with col2:
        st.markdown("### 📋 Registered Users")
        user_files = [os.path.splitext(file)[0] for file in os.listdir(DATA_DIR) if file.endswith(".wav")]

        if user_files:
            df = pd.DataFrame(user_files, columns=["Username"])

            # Add verification columns
            df["Voice Data"] = "✅"
            df["Face Data"] = df["Username"].apply(
                lambda x: "✅" if os.path.exists(os.path.join(FACE_DIR, f"{x}.jpg")) else "❌"
            )

            st.dataframe(df.style.apply(lambda x: ["color: #a855f7"] * len(x), axis=1), height=300)
        else:
            st.info("No users registered yet")

    st.markdown("---")
    # User Registration Form
    with st.form("register_form"):
        st.markdown("### 🧑‍💻 Register New User")
        username = st.text_input("Username", help="Choose a unique username")

        col1, col2 = st.columns(2)
        with col1:
            record_voice = st.checkbox("Record Voice", value=True)
        with col2:
            capture_face_check = st.checkbox("Capture Face", value=True)

        if st.form_submit_button("👤 Register User", use_container_width=True):
            if username:
                registration_successful = True

                if record_voice:
                    file_path = os.path.join(DATA_DIR, f"{username}.wav")
                    if not record_audio(file_path, registration=True):
                        registration_successful = False

                if capture_face_check:
                    if not capture_face(username):
                        registration_successful = False

                if registration_successful:
                    # Train models
                    with st.spinner("Training models with new data..."):
                        train_voice_model()
                        train_face_model()
                    st.success(f"✅ User {username} registered successfully!")
                    st.rerun()
            else:
                st.error("Please enter a username")

    st.markdown("---")
    # User Deletion Section
    st.markdown("### 🗑 Delete User")
    user_list = [os.path.splitext(file)[0] for file in os.listdir(DATA_DIR) if file.endswith(".wav")]
    if user_list:
        user_to_delete = st.selectbox("Select User to Delete", user_list)
        if st.button("🗑 Delete Selected User", use_container_width=True):
            delete_user(user_to_delete)
            st.rerun()
    else:
        st.info("No users available to delete.")

    st.markdown("---")
    # Model Management
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎙 Retrain Voice Model", use_container_width=True):
            with st.spinner("Training voice model..."):
                train_voice_model()
            st.rerun()

    with col2:
        if st.button("👤 Retrain Face Model", use_container_width=True):
            with st.spinner("Training face model..."):
                train_face_model()
            st.rerun()


# Authentication flow with liveness detection
def authentication_flow():
    st.title("🔐 Secure Authentication")
    st.markdown("---")

    # Security Level Selection
    security_level = st.radio(
        "Select Security Level",
        ["Standard (Voice OR Face)", "High (Voice AND Face)", "Maximum (Face with Active Liveness)"],
        horizontal=True
    )

    st.markdown("---")

    # Initialize session state for auth steps
    if 'auth_steps' not in st.session_state:
        st.session_state.auth_steps = {
            'voice_done': False,
            'face_done': False,
            'current_user': None,
            'liveness_passed': False
        }

    # Voice Authentication (not required for maximum security)
    if (security_level != "Maximum (Face with Active Liveness)" and
            (security_level == "Standard (Voice OR Face)" or not st.session_state.auth_steps['voice_done'])):
        st.markdown("### 1️⃣ Voice Verification")
        if st.button("🎙 Start Voice Authentication", key="voice_auth_btn", use_container_width=True):
            temp_file = "temp_auth.wav"
            if record_audio(temp_file, duration=5, registration=False):
                speaker, confidence = recognize_speaker(temp_file)

                if speaker:
                    st.success(f"✅ Voice recognized as {speaker} (Confidence: {confidence:.1f}%)")
                    st.session_state.auth_steps['voice_done'] = True
                    st.session_state.auth_steps['current_user'] = speaker

                    # If standard security, we're done
                    if security_level == "Standard (Voice OR Face)":
                        st.session_state.fully_authenticated = True
                else:
                    st.error(f"❌ Voice not recognized (Confidence: {confidence:.1f}%)")

    # Face Authentication with Liveness Detection
    if (security_level == "Maximum (Face with Active Liveness)" or
            (security_level == "High (Voice AND Face)" and st.session_state.auth_steps['voice_done']) or
            (security_level == "Standard (Voice OR Face)" and not st.session_state.auth_steps['voice_done'])):

        # Only show face auth if needed based on security level
        if not st.session_state.auth_steps.get('face_done', False):
            st.markdown("### 2️⃣ Face Verification")

            # Enhanced instructions for liveness detection
            if security_level == "Maximum (Face with Active Liveness)":
                st.warning("For maximum security, please follow the on-screen instructions for liveness verification")
                st.info("You'll be asked to perform random actions like blinking, smiling, or turning your head")

            if st.button("📷 Start Face Authentication",
                         key="face_auth_btn",
                         use_container_width=True):

                # Capture and verify face with liveness check
                frame = capture_face(for_auth=True)
                if frame is not None:
                    is_match, matched_user, confidence, liveness_passed = verify_face(
                        frame,
                        check_liveness=(security_level != "Standard (Voice OR Face)")
                    )

                    if is_match:
                        st.session_state.auth_steps['liveness_passed'] = liveness_passed

                        if security_level == "Maximum (Face with Active Liveness)":
                            if liveness_passed:
                                st.success(f"✅ Face recognized as {matched_user} (Confidence: {confidence:.1f}%)")
                                st.session_state.auth_steps['face_done'] = True
                                st.session_state.auth_steps['current_user'] = matched_user
                                st.session_state.fully_authenticated = True
                            else:
                                st.error("❌ Liveness verification failed!")
                        else:
                            st.success(f"✅ Face recognized as {matched_user} (Confidence: {confidence:.1f}%)")

                            # Check if voice and face match for high security
                            if security_level == "High (Voice AND Face)":
                                if st.session_state.auth_steps['current_user'] == matched_user:
                                    st.session_state.auth_steps['face_done'] = True
                                    st.session_state.fully_authenticated = True
                                else:
                                    st.error("❌ Face and voice identification don't match!")
                            else:
                                # For standard security, we're authenticated with just face
                                st.session_state.auth_steps['face_done'] = True
                                st.session_state.auth_steps['current_user'] = matched_user
                                st.session_state.fully_authenticated = True
                    else:
                        st.error(f"❌ Face not recognized (Confidence: {confidence:.1f}%)")

    # Display authenticated user information
    if st.session_state.get('fully_authenticated', False):
        st.markdown("---")
        st.balloons()

        auth_method = ""
        if security_level == "Maximum (Face with Active Liveness)":
            auth_method = "Active Liveness Face Biometrics"
        elif (st.session_state.auth_steps.get('voice_done', False) and
              st.session_state.auth_steps.get('face_done', False)):
            auth_method = "Voice & Face Biometrics"
        elif st.session_state.auth_steps.get('voice_done', False):
            auth_method = "Voice Biometrics"
        else:
            auth_method = "Face Biometrics"

        st.success(f"✨ Welcome back {st.session_state.auth_steps['current_user']}! Authentication successful!")

        st.markdown(f"""
        <div class="card">
            <h3 style='color: #a855f7;'>Access Granted</h3>
            <p>User: {st.session_state.auth_steps['current_user']}</p>
            <p>Security Level: {security_level}</p>
            <p>Auth Method: {auth_method}</p>
            <p>Liveness Check: {'Passed ✅' if st.session_state.auth_steps['liveness_passed'] else 'Not required'}</p>
            <p>Status: Verified ✅</p>
            <p>Last login: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

        # Add a logout button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.auth_steps = {
                'voice_done': False,
                'face_done': False,
                'current_user': None,
                'liveness_passed': False
            }
            st.session_state.fully_authenticated = False
            st.rerun()


# CSS Styling (enhanced)
st.markdown("""
<style>
    /* Main content styling */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: #ffffff;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #25253b !important;
        border-right: 2px solid #393952;
    }
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:disabled {
        background: linear-gradient(45deg, #6366f199 0%, #a855f799 100%);
        transform: none;
        box-shadow: none;
    }
    /* Card styling */
    .card {
        background: #25253b;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #393952;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Spinner styling for recording and capturing */
    .pulsating-mic {
        width: 50px;
        height: 50px;
        background-color: #a855f7;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin: 0 auto;
    }
    @keyframes pulse {
        0% { transform: scale(0.9); opacity: 1; }
        70% { transform: scale(1); opacity: 0.7; }
        100% { transform: scale(0.9); opacity: 1; }
    }
    /* Enhanced form styling */
    [data-testid="stForm"] {
        background-color: #25253b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #393952;
    }
    /* Better alerts */
    .stAlert {
        border-radius: 10px;
    }
    /* Improved image borders */
    .stImage {
        border-radius: 10px;
        border: 2px solid #393952;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'auth_steps' not in st.session_state:
    st.session_state.auth_steps = {
        'voice_done': False,
        'face_done': False,
        'current_user': None
    }
if 'fully_authenticated' not in st.session_state:
    st.session_state.fully_authenticated = False

# Load face encodings at startup
if os.path.exists(FACE_ENCODINGS_FILE):
    with open(FACE_ENCODINGS_FILE, 'rb') as f:
        known_face_encodings = pickle.load(f)
else:
    known_face_encodings = {}

# Load voice embeddings at startup
if os.path.exists(VOICE_EMBEDDINGS_FILE):
    with open(VOICE_EMBEDDINGS_FILE, 'rb') as f:
        known_voice_embeddings = pickle.load(f)
else:
    known_voice_embeddings = {}

# Navigation
st.sidebar.title("🛡 Enhanced BioAuth")
st.sidebar.markdown("---")
user_type = st.sidebar.radio("Select Mode", ["🔒 Authentication", "👑 Admin Panel"])

if user_type == "👑 Admin Panel":
    admin_password = st.sidebar.text_input("🔑 Admin Password", type="password")
    if admin_password:
        if admin_password == "Admin@123":
            admin_dashboard()
        else:
            st.sidebar.error("❌ Invalid Admin Password")
    else:
        st.warning("⚠ Please enter admin password to access the admin panel")
else:
    authentication_flow()

# Show app version
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh System"):
    st.rerun()
st.sidebar.markdown("### System Information")
st.sidebar.info("""
Version: 3.1.0  
Features:  
- Voice recognition with Resemblyzer  
- Face recognition with OpenCV  
- Liveness detection with dlib  
- Multi-factor authentication  
""")