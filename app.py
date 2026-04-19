import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import tempfile
from io import BytesIO

try:
    import cv2
    CV2 = True
except:
    CV2 = False

# ─── PAGE CONFIG ───
st.set_page_config(page_title="DeepTrust AI", layout="wide")

# ─── CUSTOM UI (🔥 MODERN LOOK) ───
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.glass {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ─── LOGIN SYSTEM ───
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<div class='title'>🔐 Login to DeepTrust</div>", unsafe_allow_html=True)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
        else:
            st.error("Invalid credentials")

def signup():
    st.markdown("<div class='title'>📝 Create Account</div>", unsafe_allow_html=True)
    st.text_input("New Username")
    st.text_input("New Password", type="password")
    st.button("Sign Up (Demo)")

# ─── AUTH FLOW ───
if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        login()
    with tab2:
        signup()
    st.stop()

# ─── MAIN APP ───

# 🔥 HEADER
st.markdown("<div class='title'>🛡️ DeepTrust AI</div>", unsafe_allow_html=True)
st.caption("Advanced Deepfake Detection System")

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

# ─── SIDEBAR ───
mode = st.sidebar.radio("Mode", ["Upload", "Compare"])

# ─── UPLOAD ───
if mode == "Upload":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        st.image(file)

        if st.button("Analyze 🚀"):
            score, verdict = detector.analyze(img)

            st.markdown("### Result")
            st.progress(score/100)
            st.subheader(f"{verdict} ({score})")

# ─── COMPARE ───
elif mode == "Compare":
    col1, col2 = st.columns(2)

    f1 = col1.file_uploader("Image 1")
    f2 = col2.file_uploader("Image 2")

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

# ─── FOOTER ───
st.markdown("---")
st.caption("🚀 DeepTrust AI | Hackathon Edition")
