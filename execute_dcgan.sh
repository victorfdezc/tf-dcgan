# sudo docker build -t my_tensorflow_image .
sudo docker run --gpus all -v $(pwd):/app -v $(pwd)/tf-dcgan.py:/app/tf-dcgan.py my_tensorflow_image
