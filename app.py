import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

# Konfigurasi Halaman Web
st.set_page_config(page_title="Pencocokan Wajah", layout="centered")

st.title("📷 Aplikasi Pencocokan Wajah")
st.write("Unggah dua foto untuk menghitung tingkat kemiripan wajah dalam persentase.")

# Layout 2 Kolom untuk Upload Gambar
col1, col2 = st.columns(2)


with col1:
    st.subheader("Gambar 1")
    uploaded_file1 = st.file_uploader("Pilih Foto Pertama", type=["jpg", "jpeg", "png"], key="1")

with col2:
    st.subheader("Gambar 2")
    uploaded_file2 = st.file_uploader("Pilih Foto Kedua", type=["jpg", "jpeg", "png"], key="2")

# Tampilkan gambar jika sudah diunggah
if uploaded_file1 and uploaded_file2:
    img1 = Image.open(uploaded_file1)
    img2 = Image.open(uploaded_file2)

    col1.image(img1, caption="Foto 1", use_container_width=True)
    col2.image(img2, caption="Foto 2", use_container_width=True)

    st.markdown("---")

    # Tombol untuk Proses Pencocokan
    if st.button("🔍 Cocokkan Wajah", use_container_width=True):
        with st.spinner("Menganalisis dan menghitung kemiripan..."):
            # Konversi gambar ke array NumPy (RGB) yang dibaca oleh face_recognition
            img1_np = np.array(img1.convert("RGB"))
            img2_np = np.array(img2.convert("RGB"))

            # Dapatkan face encoding dari kedua gambar
            encodings1 = face_recognition.face_encodings(img1_np)
            encodings2 = face_recognition.face_encodings(img2_np)

            # Validasi apakah wajah terdeteksi
            if len(encodings1) == 0:
                st.error("❌ Wajah tidak terdeteksi pada Gambar 1.")
            elif len(encodings2) == 0:
                st.error("❌ Wajah tidak terdeteksi pada Gambar 2.")
            else:
                # Hitung jarak Euclidean antara dua encoding wajah
                jarak = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]

                # Konversi jarak ke persentase kemiripan
                persentase = max(0.0, (1.0 - jarak) * 100)

                # Tampilkan Hasil
                st.subheader("Hasil Analisis:")
                st.metric(label="Tingkat Kemiripan", value=f"{persentase:.2f}%")

                if jarak < 0.6:
                    st.success("✅ **Status:** Wajah COCOK (Orang yang sama)")
                else:
                    st.warning("⚠️ **Status:** Wajah TIDAK COCOK (Orang yang berbeda)")

elif uploaded_file1 or uploaded_file2:
    st.info("Silakan unggah kedua foto untuk memulai pencocokan.")