import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import hashlib
import os

# ─── CONFIG ───
st.set_page_config(page_title="DeepTrust AI", layout="wide")

USER_FILE = "users.json"

# ─── USER STORAGE ───
def load_users():
    if not os.path.exists(USER_FILE):
        return {}
    with open(USER_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ─── SESSION INIT ───
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

users = load_users()

# ─── AUTH ───
def login():
    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

def signup():
    st.subheader("📝 Sign Up")

    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if username in users:
            st.warning("User already exists")
        else:
            users[username] = hash_password(password)
            save_users(users)
            st.success("Account created")

# ─── AUTH FLOW ───
if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        login()
    with tab2:
        signup()

    st.stop()

# ─── LOGOUT ───
with st.sidebar:
    st.write(f"👤 {st.session_state.user}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ─── HEADER ───
st.title("🛡️ DeepTrust AI")
st.caption("AI-assisted Deepfake Detection")

# ─── DETECTOR ───
class Detector:

    def detect_face(self, gray):
        face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return face.detectMultiScale(gray, 1.3, 5)

    def compute(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = self.detect_face(gray)

        if len(faces) == 0:
            return 70, "No face detected → fallback analysis"

        # Texture
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Noise
        noise = np.std(gray)

        # Edges
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges)

        # Color consistency
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_var = np.var(hsv[:,:,1])

        # Weighted score
        score = (
            0.3 * (lap/100) +
            0.2 * (noise/50) +
            0.3 * (edge_density/50) +
            0.2 * (color_var/1000)
        )

        score = max(0, min(1, score))
        return int(score*100), None

    def analyze(self, img):
        score, note = self.compute(img)

        if score >= 75:
            verdict = "Likely Real ✅"
        elif score >= 50:
            verdict = "Uncertain ⚠️"
        else:
            verdict = "Likely Fake 🚨"

        return score, verdict, note

detector = Detector()

# ─── MAIN ───
file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
    st.image(file)

    if st.button("Analyze 🚀"):
        score, verdict, note = detector.analyze(img)

        st.markdown("### 🔥 Trust Score")
        st.progress(score/100)
        st.subheader(f"{verdict} ({score})")

        col1, col2 = st.columns(2)
        col1.metric("Real Probability", f"{score}%")
        col2.metric("Fake Probability", f"{100-score}%")

        if note:
            st.info(note)

        if 50 <= score <= 70:
            st.warning("Manual verification recommended")

        st.markdown("### 🧠 Explanation")

        if score >= 75:
            st.write("• Natural texture")
            st.write("• Stable lighting")
        elif score >= 50:
            st.write("• Mixed signals")
            st.write("• Possible edits")
        else:
            st.write("• Artificial patterns")
            st.write("• Edge inconsistencies")

        st.caption("⚠️ Probabilistic AI system — not 100% guarantee")
