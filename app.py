import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบ AI คัดแยกขยะรีไซเคิล", 
    page_icon="♻️", 
    layout="wide"
)

# ตกแต่ง CSS เพิ่มเติม (ขยายความกว้าง Sidebar ให้กว้างขึ้นตามต้องการ)
st.markdown("""
    <style>
    /* ขยายความกว้างของ Sidebar ให้กว้างขึ้นเป็น 400px */
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 400px !important;
    }
    .main-title {
        font-size: 2.3rem;
        color: #4CAF50;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    .custom-result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1b4d3e !important;
        border-left: 6px solid #4CAF50;
        color: #ffffff !important;
        margin-bottom: 10px;
    }
    .custom-result-box span {
        color: #ffffff !important;
        font-size: 1.4rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 2. ส่วนหัวของเว็บ
st.markdown('<p class="main-title">♻️ ระบบ AI คัดแยกขยะรีไซเคิลอัตโนมัติ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">โปรเจกต์ต้นแบบสำหรับการคัดแยกขยะด้วย Deep Learning (MobileNetV2)</p>', unsafe_allow_html=True)

# 3. ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trash_classifier_model.keras")

with st.spinner("กำลังโหลดโมเดล AI... กรุณารอสักครู่"):
    model = load_model()

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

advice = {
    'cardboard': "📦 **ลังกระดาษ:** พับให้แบนเพื่อประหยัดพื้นที่ แล้วทิ้งในถังรีไซเคิลกระดาษ",
    'glass': "🍾 **ขวดแก้ว:** เทของเหลวออก ล้างให้สะอาด แล้วทิ้งถังรีไซเคิลแก้วด้วยความระมัดระวัง",
    'metal': "🥫 **โลหะ/กระป๋อง:** บีบให้แบน (ถ้าทำได้) แล้วทิ้งถังรีไซเคิลโลหะ",
    'paper': "📄 **กระดาษ:** แยกกระดาษเปื้อนอาหารหรือน้ำมันออก ทิ้งเฉพาะกระดาษแห้งในถังรีไซเคิล",
    'plastic': "🥤 **พลาสติก:** เทของเหลวออก ล้างพอสะอาด บีบให้แบน แล้วทิ้งถังรีไซเคิลพลาสติก",
    'trash': "🗑 **ขยะทั่วไป:** ไม่สามารถรีไซเคิลได้ ให้ทิ้งลงถังขยะทั่วไป"
}

# 4. แถบเมนูด้านซ้าย (Sidebar)
st.sidebar.markdown("### 🗂️ แผงควบคุมการทดสอบ")
st.sidebar.markdown("---")

option = st.sidebar.radio("📌 **เลือกวิธีทดสอบระบบ:**", ["📤 อัปโหลดรูปภาพเอง", "🖼️ ใช้รูปตัวอย่างในระบบ"])

uploaded_file = None

if option == "📤 อัปโหลดรูปภาพเอง":
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("เลือกไฟล์รูปภาพขยะ (.jpg, .png)", type=["jpg", "jpeg", "png"])
else:
    st.sidebar.markdown("---")
    sample_images = {
        "📦 ลังกระดาษ 1": "cardboard1.jpg",
        "📦 ลังกระดาษ 2": "cardboard2.jpg",
        "🍾 ขวดแก้ว 1": "glass1.jpg",
        "🍾 ขวดแก้ว 2": "glass2.jpg",
        "🥫 กระป๋องโลหะ 1": "metal1.jpg",
        "🥫 กระป๋องโลหะ 2": "metal2.jpg",
        "📄 กระดาษ 1": "paper1.jpg",
        "📄 กระดาษ 2": "paper2.jpg",
        "🥤 พลาสติก 1": "plastic1.jpg",
        "🥤 พลาสติก 2": "plastic2.jpg"
    }
    selected_sample = st.sidebar.selectbox("🎯 **เลือกรูปตัวอย่าง:**", list(sample_images.keys()))
    uploaded_file = sample_images[selected_sample]

# 5. ส่วนแสดงผลหลัก (แบ่งเป็น 2 คอลัมน์ซ้าย-ขวา)
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("📷 รูปภาพที่ตรวจสอบ")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔍 ผลการวิเคราะห์จาก AI")
            with st.spinner("กำลังประมวลผลภาพ..."):
                img_resized = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model.predict(img_array)
                confidence = np.max(predictions[0]) * 100
                predicted_class = class_names[np.argmax(predictions[0])]
            
            # แสดงผลลัพธ์ด้วย HTML แบบกำหนดสีเอง
            st.markdown(f"""
                <div class="custom-result-box">
                    ประเภทขยะ: <span>{predicted_class.upper()}</span>
                </div>
            """, unsafe_allow_html=True)
            
            st.write(f"ระดับความมั่นใจ (Confidence):")
            st.progress(int(confidence))
            st.caption(f"**{confidence:.2f}%**")
            
            st.markdown("---")
            st.info(advice[predicted_class])
            
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์รูปภาพ '{uploaded_file}' ในโฟลเดอร์โปรเจกต์ กรุณาตรวจสอบว่าได้ย้ายไฟล์รูปมาไว้หน้าหลักหรือยังครับ")
else:
    st.info("💡 **คำแนะนำ:** กรุณาอัปโหลดรูปภาพขยะ หรือเลือกรูปตัวอย่างจากแถบเมนูด้านซ้าย เพื่อเริ่มต้นการทำงานของระบบครับ")