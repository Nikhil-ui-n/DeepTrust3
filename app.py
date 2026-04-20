import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="DeepTrust AI", layout="wide")

# ─── UI STYLE ───
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color:white;
}
.big-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🛡️ DeepTrust AI</div>", unsafe_allow_html=True)

# ─── DETECTOR ───
class Detector:

    def detect_face(self, gray):
        face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return face.detectMultiScale(gray, 1.3, 5)

    def score_face(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        texture = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise = np.std(gray)
        edges = np.mean(cv2.Canny(gray,100,200))

        score = (texture*0.4 + noise*0.3 + edges*0.3)/8000
        return min(1, score)

    def analyze(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detect_face(gray)

        if len(faces) == 0:
            return 65, "Non-face image → limited analysis"

        scores = []

        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            scores.append(self.score_face(face))

        final = int(np.mean(scores)*100)

        if final >= 75:
            verdict = "Likely Real ✅"
        elif final >= 50:
            verdict = "Uncertain ⚠️"
        else:
            verdict = "Likely Fake 🚨"

        return final, verdict

detector = Detector()

# ─── SIDEBAR ───
mode = st.sidebar.radio("Mode", ["Upload", "Compare"])

# ─── UPLOAD ───
if mode == "Upload":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
        st.image(file, use_column_width=True)

        if st.button("Analyze 🚀"):
            score, verdict = detector.analyze(img)

            st.progress(score/100)
            st.subheader(f"{verdict} ({score})")

            col1, col2 = st.columns(2)
            col1.metric("Real", f"{score}%")
            col2.metric("Fake", f"{100-score}%")

# ─── 🔥 FIXED COMPARE MODE ───
elif mode == "Compare":

    col1, col2 = st.columns(2)

    file1 = col1.file_uploader("Image 1", key="1")
    file2 = col2.file_uploader("Image 2", key="2")

    if file1 and file2:
        img1 = cv2.cvtColor(np.array(Image.open(file1)), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(Image.open(file2)), cv2.COLOR_RGB2BGR)

        col1.image(file1)
        col2.image(file2)

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
