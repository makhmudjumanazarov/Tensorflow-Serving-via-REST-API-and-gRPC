import streamlit as st
from tensorflow_video import *
import cv2
import tempfile
from PIL import Image
import numpy as np

st.write("""
### Face Detection via Tensorflow
""")

# Upload a video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
model_name = "/home/airi/yolo/Tensorflow_Serving/saved_models/model_1"
# Load the model
loaded_model = model_load(model_name)
conf_threshold = 0.5
iou_thres = 0.1

if video_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())

    # Open the video file for reading
    cap = cv2.VideoCapture(tfile.name)

    if cap.isOpened():
        st.write("Video Playback:")
        prev_time = 0
        curr_time = 0
        fps_out = st.empty()
        image_out = st.empty()

        (success, image) = cap.read()

        while success:

            prev_time = time.time()
            
            input_shape = (640, 640)  # Replace with the expected input shape of your YOLO model
            bgr, ratio, dwdh = letterbox(np.array(image), input_shape)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            tensor = np.ascontiguousarray(tensor)

            # Get the serving function
            serving_fn = loaded_model.signatures["serving_default"]

            # Prepare the input tensor
            input_name = list(serving_fn.structured_input_signature[1].keys())[0]
            input_tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)

            # Make a prediction
            result = serving_fn(**{input_name: input_tensor})['output']
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            fps_out.write(f"FPS:{fps}")

            # result = model_load(model_name, tensor)['output']
            predictions = np.array(result).reshape((5, 8400))
            predictions = predictions.T 

            # Filter out object confidence scores below threshold
            scores = np.max(predictions[:, 4:], axis=1)
            predictions = predictions[scores > conf_threshold, :]
            scores = scores[scores > conf_threshold] 
            class_ids = np.argmax(predictions[:, 4:], axis=1)

            # Get bounding boxes for each object
            boxes = predictions[:, :4]

            #rescale box
            input_shape = np.array([640, 640, 640, 640])
            boxes = np.divide(boxes, input_shape, dtype=np.float32)
            boxes *= np.array([640, 640, 640, 640])
            boxes = boxes.astype(np.int32)

            indices = nms(boxes, scores, iou_thres)
            image_draw = rgb.copy()

            for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
                bbox = bbox.round().astype(np.int32).tolist()
                cls_id = int(label)
                color = (0,255,0)
                cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                cv2.putText(image_draw, f'face:{int(score*100)}',
                            (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, [225, 255, 255],
                            thickness=1)
                
            

            output_image = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
            # Display the frame in Streamlit
            image_out.image(output_image, channels="BGR", use_column_width=True)
            # cv2.imwrite(f"./detected_images/rasm.png", output_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()
                # Release everything after the job is finished
        cap.release()
        # out.release()
        cv2.destroyAllWindows()
    else:
        st.write("Error: Unable to open the video file.")
else:
    st.write("Please upload a video file to display.")
