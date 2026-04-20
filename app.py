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

users = load_users()

# ─── SESSION ───
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

# ─── LOGIN ───
def login():
    st.subheader("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u] == hash_password(p):
            st.session_state.logged_in = True
            st.session_state.user = u
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ─── SIGNUP ───
def signup():
    st.subheader("📝 Sign Up")
    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if u in users:
            st.warning("User exists")
        else:
            users[u] = hash_password(p)
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

st.title("🛡️ DeepTrust AI")
st.caption("Deepfake Detection System")

# ─── DETECTOR ───
class Detector:

    def detect_face(self, gray):
        face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return face.detectMultiScale(gray, 1.3, 5)

    def analyze(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detect_face(gray)

        if len(faces) == 0:
            return 65, "Non-face content"

        scores = []
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            texture = cv2.Laplacian(g, cv2.CV_64F).var()
            noise = np.std(g)
            edges = np.mean(cv2.Canny(g,100,200))

            score = (texture*0.4 + noise*0.3 + edges*0.3)/8000
            scores.append(min(1, score))

        final = int(np.mean(scores)*100)

        if final >= 75:
            verdict = "Likely Real ✅"
        elif final >= 50:
            verdict = "Uncertain ⚠️"
        else:
            verdict = "Likely Fake 🚨"

        return final, verdict

detector = Detector()

# ─── MODE ───
mode = st.sidebar.radio("Mode", ["Upload", "Compare"])

# ─── UPLOAD ───
if mode == "Upload":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        st.image(file)

        if st.button("Analyze 🚀"):
            score, verdict = detector.analyze(img)

            st.progress(score/100)
            st.subheader(f"{verdict} ({score})")

            col1, col2 = st.columns(2)
            col1.metric("Real", f"{score}%")
            col2.metric("Fake", f"{100-score}%")

# ─── COMPARE ───
elif mode == "Compare":

    col1, col2 = st.columns(2)

    f1 = col1.file_uploader("Image 1", key="1")
    f2 = col2.file_uploader("Image 2", key="2")

    if f1 and f2:
        img1 = cv2.cvtColor(np.array(Image.open(f1)), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(Image.open(f2)), cv2.COLOR_RGB2BGR)

        col1.image(f1)
        col2.image(f2)

        if st.button("Compare 🚀"):
            s1, v1 = detector.analyze(img1)
            s2, v2 = detector.analyze(img2)

            col1.subheader(f"{v1} ({s1})")
            col2.subheader(f"{v2} ({s2})")

            st.markdown("## 🔍 Insight")

            if abs(s1 - s2) < 10:
                st.warning("Both images look similar")
            elif s1 > s2:
                st.success("Image 1 is more authentic")
            else:
                st.success("Image 2 is more authentic")
