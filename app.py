import cv2
from PIL import Image
import streamlit as st
import numpy as np
from ultralytics import YOLO
import os


def app():
    global uploaded_file,source
    source = None  # กำหนดค่าเริ่มต้นให้กับ source
    uploaded_file = None
    st.header('วินิฉัยโรคต้อกระจก')
    st.subheader('ทำงานด้วย AI ')
    st.subheader('จัดทำโดย')
    st.write('นายวชิรภัทร แก้วมะลัง')
    st.write('นายอัฟฟาน เจ๊ะมะ')
    st.write('อัปโหลดรูปภาพหน้าตรง')
    st.image('requirement.png')
    upbutton = st.empty
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Streamlit app title
    # Start the camera on button click
    startbt=st.button("Start Camera", key="start_camera_button",on_click=cam_button,use_container_width=True)
    
    if "cam" in st.session_state:
        
        capture_image()

    if "upload" in st.session_state :
        st.session_state.cam = False
        st.session_state.upload = False

    if "cam" not in st.session_state:

        st.session_state.starting = False
        st.session_state.cam = False
        
        with st.form("my_form"):
            uploaded_file = st.file_uploader("Upload video", type=("jpg", "jpeg", "png"))
            #min_confidence = st.slider('Confidence score', 0.0, 1.0,0.5)
            upbt= st.form_submit_button(label='Submit')


    # Load a pretrained YOLO11n model
    model = YOLO("model/eye-detect.pt")
    model_cls = YOLO("model/cataract-cls.pt")  # load a custom model

    
    if (uploaded_file is not None )or (source is not None):
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the image from the binary stream
        if source is not None:
            pass
        st.image(source, channels="BGR")  # Display the uploaded image


        # Read an image using OpenCV
        #source = cv2.imread("X:/NKWK/SET/ALL PROJECT/AI-EYES/YoloWithFlatter/2.png")
        print("--Start EyesDetection--")
        # Run inference on the source
        results = model(source)  # list of Results objects
        names = model.names
        eyes = {}
        eyes_image = {}
        boxes = results[0].boxes.xyxy.tolist()

        # Visualize the results
        for r in results:
            # Plot results image
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                # Show results to screen (in supported environments)
            # r.show()
                # Save results to disk
            r.save(filename=f"results.jpg") 
            for c in r.boxes.cls:
                print(names[int(c)])

            for i, box in enumerate(boxes):
                print(f'Round {i}')
                x1, y1, x2, y2 = box
                # Crop the object using the bounding box coordinates
                ultralytics_crop_object = source[int(y1):int(y2), int(x1):int(x2)] #image crop
                #model_cls
                print("--Start Classification--")
                print(f'---image{str(i)}---')
                results_cls = model_cls(ultralytics_crop_object)  # predict on an image
                names_cls = model_cls.names
                # print(results_cls)
                print(names_cls[1])
                eyes[f'eye{i}'] = names_cls[1] 
                eyes_image[f'eye{i}'] = ultralytics_crop_object
                # Save the cropped object as an image
                cv2.imwrite('ultralytics_crop_' + str(i) + '.jpg', ultralytics_crop_object)
                st.image(ultralytics_crop_object, channels="BGR",caption=f'Eye {i+1}')
                st.write(f'Eye {i+1} is {names_cls[1]}')
                
        print("All Result")
        print(eyes)

        print("--Finished--")
        st.title("--Finished--")

def cap_button():
    st.session_state.cap = True

def cam_button():
    st.session_state.cam = True
    
def up_button():
    st.session_state.upload = True


def capture_image():
    frame_img_path = "frame3.png"
    global uploaded_file,source
    
    cap = None
    # st.button("Upload", key="upload_open",on_click=up_button,use_container_width=True)
    cap = st.camera_input("TEST",label_visibility="hidden")
    if cap is None:
        st.info("Take a picture.")
        return

    if cap is not None:
            file_bytes = np.frombuffer(cap.read(), np.uint8)
            source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the image from the binary stream

if __name__ == "__main__":
    app()
