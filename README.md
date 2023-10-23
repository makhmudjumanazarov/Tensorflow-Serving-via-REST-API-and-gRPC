### Tensorflow Serving via REST API and gRPC with Docker
Model is an YOLOv8 model. This model was trained for Face Detection and converted Tensorflow model format.
#### 1. Tensorflow Serving via REST API with Docker
- Download the TensorFlow Serving Docker image
<pre>
docker pull tensorflow/serving
</pre> 
- To start docker container
<pre>
docker run -it -v /home/airi/yolo/Tensorflow_Serving:/Tensorflow_Serving -p 8601:8601 --entrypoint /bin/bash tensorflow/serving
</pre> 
- To serve only latest model
<pre>
tensorflow_model_server --rest_api_port=8601 --model_name=yolov8_tf_serving --model_base_path=/Tensorflow_Serving/saved_models/
</pre> 
- To serve models using model config file
<pre>
tensorflow_model_server --rest_api_port=8601 --model_config_file=/Tensorflow_Serving/model.config.a
</pre> 
- TF Serving Installation Instructions & Config File Help
  https://www.tensorflow.org/tfx/serving/docker https://www.tensorflow.org/tfx/serving/serving_config

#### 2. Tensorflow Serving via gRPC with Docker
- To start docker container
<pre>
docker run -it -v /home/airi/yolo/Tensorflow_Serving:/Tensorflow_Serving -p 2000:2000 --entrypoint /bin/bash tensorflow/serving
</pre> 
- To serve only latest model
<pre>
tensorflow_model_server --port=2000 --model_name=tf_serving_grpc --model_base_path=/Tensorflow_Serving/saved_models
</pre> 
