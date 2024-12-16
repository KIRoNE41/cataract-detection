import cv2
from PIL import Image
import streamlit as st
import numpy as np
from ultralytics import YOLO
import os

def app():
    global uploaded_file, source
    source = None  # Default value for source
    uploaded_file = None
    st.header('วินิฉัยโรคต้อกระจก')
    st.subheader('ทำงานด้วย AI ')
    st.subheader('จัดทำโดย')
    st.write('นายวชิรภัทร แก้วมะลัง')
    st.write('นายอัฟฟาน เจ๊ะมะ')
    st.write('อัปโหลดรูปภาพหน้าตรง')
    st.image('requirement.png')
    
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Hide unnecessary elements from the Streamlit UI
    startbt = st.button("Start Camera", key="start_camera_button", on_click=cam_button)

    if "cam" not in st.session_state:
        st.session_state.cam = False
        st.session_state.upload = False

    # Check if the camera is started
    if st.session_state.cam:
        capture_image()  # Start capturing from the camera

    # File upload section inside a form
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
        submit_button = st.form_submit_button(label='Submit')

    # Load YOLO models
    model = YOLO("model/eye-detect.pt")
    model_cls = YOLO("model/cataract-cls.pt")  # load a custom model

    # Processing uploaded file or camera image
    if (uploaded_file is not None or source is not None):
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode image from the binary stream
        
        if source is not None:
            st.image(source, channels="BGR")  # Display the uploaded image

            # Start Eye Detection
            st.write("--Start EyesDetection--")
            results = model(source)  # Run YOLO inference on the image
            names = model.names
            eyes = {}
            eyes_image = {}
            boxes = results[0].boxes.xyxy.tolist()

            # Visualize results
            for r in results:
                im_bgr = r.plot()  # Plot the result image
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # Convert BGR to RGB for display
                r.save(filename=f"results.jpg")
                for c in r.boxes.cls:
                    st.write(names[int(c)])

                # Loop through each detected box (eye)
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    ultralytics_crop_object = source[int(y1):int(y2), int(x1):int(x2)]  # Crop the eye region
                    st.write(f"--Start Classification for Eye {i+1}--")
                    results_cls = model_cls(ultralytics_crop_object)  # Classify the eye
                    names_cls = model_cls.names
                    eyes[f'eye{i}'] = names_cls[1]
                    eyes_image[f'eye{i}'] = ultralytics_crop_object

                    # Save and display the cropped eye image
                    cv2.imwrite('ultralytics_crop_' + str(i) + '.jpg', ultralytics_crop_object)
                    st.image(ultralytics_crop_object, channels="BGR", caption=f'Eye {i+1}')
                    st.write(f'Eye {i+1} is {names_cls[1]}')

            st.write("--All Results--")
            st.write(eyes)

            st.title("--Finished--")

def cam_button():
    st.session_state.cam = True

def capture_image():
    frame_img_path = "frame3.png"
    global source

    # Capture image from camera
    cap = st.camera_input("Take a picture", label_visibility="hidden")
    
    if cap is not None:
        file_bytes = np.frombuffer(cap.read(), np.uint8)
        source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the captured image

if __name__ == "__main__":
    app()
