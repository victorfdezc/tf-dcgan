# Use the official Tensorflow GPU image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy the Python script into the container
COPY tf-dcgan.py /app/

# Install required Python packages
RUN pip install imageio[ffmpeg]
RUN pip install ipython
RUN pip install matplotlib
RUN pip install --upgrade keras
RUN pip install tf-keras
RUN pip install tensorflow-docs

# Command to run your script when the container starts
CMD ["python", "tf-dcgan.py"]

