import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from numpy import ndarray
from typing import List, Optional, Tuple, Union
import tempfile
import time

np.random.seed(20)
tf.keras.backend.clear_session()

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (0, 0, 0)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:

    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im
    
def nms(boxes, scores, iou_threshold):

    # Convert to xyxy
    boxes = xywh2xyxy(boxes)
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest 
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over threshold
        keep_indices = np.where(ious < iou_threshold)[0] + 1
        sorted_indices = sorted_indices[keep_indices]

    return keep_boxes

def compute_iou(box, boxes):

    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou

def xywh2xyxy(x):

    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def model_load(model_name):

    # Load the model
    loaded_model = tf.saved_model.load(model_name)
    return loaded_model


imagePath = '/home/airi/yolo/Tensorflow_Serving/bus.jpg'
model_name = "/home/airi/yolo/Tensorflow_Serving/saved_models/1"

# Load the model
loaded_model = model_load(model_name)

def predictVideo(videoPath, conf_threshold = 0.5, iou_thres = 0.1):
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        print("Error opening file...")
        return
    
    (success, image) = cap.read()

    prev_time = 0
    curr_time = 0

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
        cv2.imwrite(f"./detected_images/rasm.png", output_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        (success, image) = cap.read()

# def main():
#     videoPath = '/home/airi/yolo/Tensorflow_Serving/supermodel.mp4'
#     predictVideo(videoPath)

# if __name__ == "__main__":
#     main()
