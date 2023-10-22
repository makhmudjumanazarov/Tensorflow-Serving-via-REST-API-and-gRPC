### Tensorflow Serving via REST API and gRPC with Docker
#### 1. Tensorflow Serving via REST API with Docker
- Download the TensorFlow Serving Docker image
<pre>
docker pull tensorflow/serving
</pre> 
- To start docker container
<pre>
docker run -it -v /home/airi/yolo/Tensorflow_Serving:/Tensorflow_Serving -p 8601:8601 --entrypoint /bin/bash tensorflow/serving
</pre> 
