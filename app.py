import streamlit as st
import numpy as np
from PIL import Image
import cv2

# ─── PAGE CONFIG ───
st.set_page_config(page_title="DeepTrust AI", layout="wide")

# ─── SESSION INIT ───
if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}  # default user

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ─── LOGIN FUNCTION ───
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ─── SIGNUP FUNCTION ───
def signup():
    st.title("📝 Sign Up")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if new_user in st.session_state.users:
            st.warning("User already exists")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("Account created! Go to login.")

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
    st.write(f"👤 {st.session_state.current_user}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.rerun()

# ─── HEADER ───
st.title("🛡️ DeepTrust AI")
st.caption("Deepfake Detection System")

# ─── DETECTOR ───
class Detector:
    def compute(self, gray):
        var = np.var(gray)
        noise = np.std(gray)
        edges = cv2.Canny(gray, 100, 200)
        return min(1.0, (var*0.4 + noise*0.3 + np.mean(edges)*0.3)/7000)

    def analyze(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = int(self.compute(gray)*100)

        if score >= 75:
            verdict = "REAL ✅"
        elif score >= 50:
            verdict = "UNCERTAIN ⚠️"
        else:
            verdict = "FAKE 🚨"

        return score, verdict

detector = Detector()

# ─── MODES ───
mode = st.sidebar.radio("Mode", ["Upload", "Compare"])

# ─── UPLOAD MODE ───
if mode == "Upload":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        st.image(file)

        if st.button("Analyze 🚀"):
            score, verdict = detector.analyze(img)

            st.progress(score/100)
            st.subheader(f"{verdict} ({score})")

# ─── COMPARE MODE ───
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

            if s1 > s2:
                st.success("Image 1 is more authentic")
            else:
                st.success("Image 2 is more authentic")

# ─── FOOTER ───
st.markdown("---")
st.caption("🚀 DeepTrust AI | Hackathon Edition")
