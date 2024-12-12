import cv2
from PIL import Image
import streamlit as st
import numpy as np
from ultralytics import YOLO


def app():
    st.header('วินิฉัยโรคต้อกระจก')
    st.subheader('ทำงานด้วย AI ')
    st.write('อัปโหลดรูปภาพหน้าตรง')
    st.image('requirement.png')
    # Load a pretrained YOLO11n model
    model = YOLO("model/eye-detect.pt")
    model_cls = YOLO("model/cataract-cls.pt")  # load a custom model

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=("jpg", "jpeg", "png"))
        #min_confidence = st.slider('Confidence score', 0.0, 1.0,0.5)
        st.form_submit_button(label='Submit')

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        source = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode the image from the binary stream
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


if __name__ == "__main__":
    app()