import cv2
from PIL import Image
import streamlit as st
import numpy as np
from ultralytics import YOLO
import os

def app():
    global uploaded_file, source
    source = None  # ค่าเริ่มต้นสำหรับ source
    uploaded_file = None
    st.header('วินิฉัยโรคต้อกระจก')
    st.subheader('ทำงานด้วย AI ')
    st.subheader('จัดทำโดย')
    st.write('นายวชิรภัทร แก้วมะลัง')
    st.write('นายอัฟฟาน เจ๊ะมะ')
    st.write('อัปโหลดรูปภาพหน้าตรง')
    st.image('requirement.png')

    # ซ่อน UI ที่ไม่ต้องการ
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # ปุ่มเปิดกล้อง
    if st.button("Start Camera", key="start_camera_button"):
        st.session_state.cam = True  # ตั้งค่าสถานะการเปิดกล้อง

    # ตรวจสอบสถานะการเปิดกล้อง
    if "cam" not in st.session_state:
        st.session_state.cam = False
        with st.form("my_form"):
            uploaded_file = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
            submit_button = st.form_submit_button(label='Submit')

    
    # ถ้ากล้องถูกเปิด ให้เรียกฟังก์ชันจับภาพจากกล้อง
    if st.session_state.cam:
        capture_image()
        
    # โหลดโมเดล YOLO
    model = YOLO("model/eye-detect.pt")
    model_cls = YOLO("model/cataract-cls.pt")

    # การประมวลผลรูปภาพที่อัปโหลดหรือรูปภาพจากกล้อง
    if (uploaded_file is not None or source is not None):
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # แปลงภาพที่อัปโหลด
        
        if source is not None:
            st.image(source, channels="BGR")  # แสดงภาพ

            # เริ่มการตรวจจับดวงตา
            st.write("--Start EyesDetection--")
            results = model(source)
            names = model.names
            eyes = {}
            eyes_image = {}
            boxes = results[0].boxes.xyxy.tolist()

            # แสดงผลลัพธ์
            for r in results:
                im_bgr = r.plot()
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # แปลง BGR เป็น RGB
                r.save(filename=f"results.jpg")

                for c in r.boxes.cls:
                    st.write(names[int(c)])

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    ultralytics_crop_object = source[int(y1):int(y2), int(x1):int(x2)]  # ตัดภาพดวงตา
                    st.write(f"--Start Classification for Eye {i+1}--")
                    results_cls = model_cls(ultralytics_crop_object)  # ทำการจำแนกประเภทของตา
                    names_cls = model_cls.names
                    eyes[f'eye{i}'] = names_cls[1]
                    eyes_image[f'eye{i}'] = ultralytics_crop_object

                    # แสดงภาพดวงตาที่ถูกตัด
                    cv2.imwrite('ultralytics_crop_' + str(i) + '.jpg', ultralytics_crop_object)
                    st.image(ultralytics_crop_object, channels="BGR", caption=f'Eye {i+1}')
                    st.write(f'Eye {i+1} is {names_cls[1]}')

            st.write("--All Results--")
            st.write(eyes)

            st.title("--Finished--")

def capture_image():
    global source
    cap = st.camera_input("Take a picture", label_visibility="hidden")

    if cap is not None:
        file_bytes = np.frombuffer(cap.read(), np.uint8)
        source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # แปลงภาพที่ถ่ายจากกล้อง

if __name__ == "__main__":
    app()
