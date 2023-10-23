import grpc
import numpy as np
import tensorflow as tf
from tensorflow_video import *

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

channel = grpc.insecure_channel('localhost:2000')  # Replace with the gRPC server address
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'tf_serving_grpc'
request.model_spec.signature_name = 'serving_default'

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

        # Set the input tensor in the gRPC request
        request.inputs['input'].CopyFrom(tf.make_tensor_proto(tensor))
        # Send the gRPC request
        response = stub.Predict(request)

        # Retrieve the predictions from the gRPC response
        array = tf.make_ndarray(response.outputs['output'])
        predictions = np.array(array).reshape((5, 8400))
        predictions = predictions.T

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        print(fps)

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
            cv2.putText(image_draw, f'face:{int(score*100)}' ,
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

def main():
    videoPath = '/home/airi/yolo/Tensorflow_Serving/supermodel.mp4'
    predictVideo(videoPath)

if __name__ == "__main__":
    main()

