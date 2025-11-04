import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils.preprocessing import *

# 🧭 Konfigurasi halaman
st.set_page_config(
    page_title="🐾Preprocessing Demo",
    layout="wide",
    page_icon="🐾"
)


st.markdown("""
<style>
    body {
        background-color: #f7f9fb;
        color: #1e1e1e;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }
    .card {
        background-color: white;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
    }
    .stImage > img {
        border-radius: 12px;
    }
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }
    .caption {
        text-align: center;
        color: gray;
        font-size: 0.9em;
        margin-top: -0.5rem;
    }
    .stage-title {
        color: #2563eb;
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 🌟 Header
st.markdown("<h1 style='text-align:center;'>🐾Preprocessing</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:gray;'>Unggah foto dan lihat proses transformasi citra.</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# 📤 Upload Gambar
uploaded_file = st.file_uploader("📤 Unggah Foto ", type=["jpg", "jpeg", "png"])

# Fungsi bantu tampil before-after dengan desain rapi
def show_step(title, before, after, caption):
    st.markdown(f"<div class='card'><div class='stage-title'>{title}</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.image(before, caption="Before", use_container_width=True)
    with col2:
        st.image(after, caption=caption, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 🚀 Proses utama
if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(image, caption="📸 Gambar Asli", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🔧 Tahapan Preprocessing")

    gray = step1_grayscale(image)
    show_step("1️⃣ Grayscale Conversion", image, gray, "Hasil Grayscale")

    denoised = step2_denoise(gray)
    show_step("2️⃣ Noise Removal (Median Filter)", gray, denoised, "Setelah Noise Dihapus")

    clahe_img = step3_clahe(denoised)
    show_step("3️⃣ CLAHE Enhancement (Peningkatan Kontras)", denoised, clahe_img, "Kontras Ditingkatkan")

    thresh = step4_threshold(clahe_img)
    show_step("4️⃣ Thresholding (Otsu)", clahe_img, thresh, "Segmentasi Biner")

    edges = step5_edges(clahe_img)
    show_step("5️⃣ Edge Detection (Canny)", clahe_img, edges, "Deteksi Tepi")

    morph = step6_morph(thresh)
    show_step("6️⃣ Morphological Operation (Closing)", thresh, morph, "Noise Dihapus & Struktur Diperjelas")

    final_img = step7_resize_normalize(morph)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image(final_img, caption="7️⃣ Resize + Normalization (224×224)", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.success("🎉 Semua tahap preprocessing selesai!")
else:
    st.info("📤 Silakan unggah gambar untuk memulai proses.")
