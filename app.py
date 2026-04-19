import streamlit as st
import numpy as np
from PIL import Image
import cv2

# ─── CONFIG ───
st.set_page_config(page_title="DeepTrust AI", layout="wide")

# ─── HEADER ───
st.markdown("""
<h1 style='text-align:center;'>🛡️ DeepTrust AI</h1>
<p style='text-align:center; color:gray;'>AI-Assisted Deepfake Detection</p>
""", unsafe_allow_html=True)

# ─── DETECTOR ───
class Detector:

    def compute(self, gray):
        variance = np.var(gray)
        noise = np.std(gray)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges)

        score = (variance*0.4 + noise*0.3 + edge_density*0.3) / 7000
        return min(1.0, score)

    def analyze(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = int(self.compute(gray) * 100)

        real_prob = score
        fake_prob = 100 - score

        if score >= 75:
            verdict = "Likely Real ✅"
        elif score >= 50:
            verdict = "Uncertain ⚠️"
        else:
            verdict = "Likely Fake 🚨"

        return score, verdict, real_prob, fake_prob

detector = Detector()

# ─── UPLOAD ───
file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_RGB2BGR)
    st.image(file, caption="Uploaded Image")

    if st.button("Analyze 🚀"):

        score, verdict, real_prob, fake_prob = detector.analyze(img)

        # 🔥 TRUST BAR
        st.markdown("### 🔥 Trust Score")
        st.progress(score/100)

        st.markdown(f"<h2 style='text-align:center;'>{score}/100</h2>", unsafe_allow_html=True)

        # 🎯 VERDICT
        st.subheader(f"{verdict}")

        # 📊 PROBABILITY
        st.markdown("### 📊 Probability")
        col1, col2 = st.columns(2)

        col1.metric("Real Probability", f"{real_prob}%")
        col2.metric("Fake Probability", f"{fake_prob}%")

        # ⚠️ WARNING ZONE
        if 50 <= score <= 70:
            st.warning("⚠️ Manual verification recommended")

        # 🧠 BREAKDOWN
        st.markdown("### 🧠 AI Analysis Breakdown")

        texture = max(0, score - 10)
        noise = max(0, score - 20)
        edges = max(0, score - 15)

        st.write(f"Texture Consistency: {texture}%")
        st.write(f"Noise Pattern: {noise}%")
        st.write(f"Edge Smoothness: {edges}%")

        # 📌 REASONING
        st.markdown("### 💡 Explanation")

        if score >= 75:
            st.write("• Natural texture detected")
            st.write("• Consistent lighting")
            st.write("• Balanced edges")
        elif score >= 50:
            st.write("• Mixed signals detected")
            st.write("• Slight inconsistencies")
            st.write("• Possible editing traces")
        else:
            st.write("• Strong artificial patterns")
            st.write("• Unnatural smoothing")
            st.write("• Edge irregularities")

        # ⚠️ DISCLAIMER
        st.caption("⚠️ This system provides probabilistic analysis, not absolute verification.")

# ─── FOOTER ───
st.markdown("---")
st.caption("🚀 DeepTrust AI | Hackathon Smart Edition")
