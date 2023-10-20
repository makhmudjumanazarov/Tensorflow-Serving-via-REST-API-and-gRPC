import requests
import json 
import numpy as np 

from tensorflow_video import *

url = 'http://localhost:2000/v1/models/yolov8_tensorflow/versions/1:predict'

imagePath = '/home/airi/yolo/Tensorflow_Serving/bus.jpg'
conf_threshold = 0.5
iou_thres = 0.1
    
input_shape = (640, 640)  # Replace with the expected input shape of your YOLO model
image = cv2.imread(imagePath)
bgr, ratio, dwdh = letterbox(np.array(image), input_shape)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
tensor = blob(rgb, return_seg=False)
tensor = np.ascontiguousarray(tensor)

data = json.dumps({"instances": tensor.tolist()})

# Send an HTTP POST request to the specified URL with the JSON payload
response = requests.post(url=url, data=data)
print(response)

# # Retrieve the predictions from the REST API response
# array = np.array(json.loads(response.text)['predictions'])

# array = array[0]['output']
# predictions = np.array(array).reshape((5, 8400))
# predictions = predictions.T 

# # Filter out object confidence scores below threshold
# scores = np.max(predictions[:, 4:], axis=1)
# predictions = predictions[scores > conf_threshold, :]
# scores = scores[scores > conf_threshold] 
# class_ids = np.argmax(predictions[:, 4:], axis=1)

# # Get bounding boxes for each object
# boxes = predictions[:, :4]

# #rescale box
# input_shape = np.array([640, 640, 640, 640])
# boxes = np.divide(boxes, input_shape, dtype=np.float32)
# boxes *= np.array([640, 640, 640, 640])
# boxes = boxes.astype(np.int32)

# indices = nms(boxes, scores, iou_thres)
# image_draw = rgb.copy()

# for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
#     bbox = bbox.round().astype(np.int32).tolist()
#     cls_id = int(label)
#     color = (0,255,0)
#     cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
#     cv2.putText(image_draw, f'face:{int(score*100)}',
#                     (bbox[0], bbox[1] - 2),
#                 cv2.FONT_HERSHEY_PLAIN,
#                 1, [225, 255, 255],
#                 thickness=1)
    
    
# output_image = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
# cv2.imwrite(f"./detected_images/rasm.png", output_image)

