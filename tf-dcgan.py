from IPython import display
from PIL import Image
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from models.dcgan import DCGAN
from tf_utils.manage_data import *
from tf_utils.process_images import *


'''
####
#### TODO mirar esto para la gan truncation
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)
'''



############################################################################################################
# MAIN SCRIPT a finalizar usando CVAE como ejemplo
############################################################################################################
batch_size = 2
epochs = 100
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 320
num_examples_to_generate = 4
img_shape = 200
img_channels = 3

load_model = True
start_epoch = 390


# # keeping the random vector constant for generation (prediction) so
# # it will be easier to see the improvement.
# random_vector_for_generation = tf.random.normal(
#     shape=[num_examples_to_generate, latent_dim])
# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

train_images = extract_video_frames("data_in/video_cortito.mp4", img_shape=img_shape, img_channels=img_channels)
# Preprocess data
train_images = shuffle_dataset(train_images)
train_images, test_images = split_dataset(train_images)
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)


train_size = len(train_images)
test_size = len(test_images)
print("Shape of train images:", np.shape(train_images))
print("Shape of test images:", np.shape(test_images))


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
print("MAXIMUM TRAIN VALUE: " + str(np.max(train_images)))
print("MINIMMUM TRAIN VALUE: " + str(np.min(train_images)))

generator_checkpoint_path = "training_1/cp-generator-{epoch:04d}.weights.h5"
discriminator_checkpoint_path = "training_1/cp-discriminator-{epoch:04d}.weights.h5"

model = DCGAN(latent_dim, img_shape, img_channels)
if load_model:
  model.generator.load_weights(generator_checkpoint_path.format(epoch=start_epoch))
  model.discriminator.load_weights(discriminator_checkpoint_path.format(epoch=start_epoch))
noise = tf.random.normal([1, latent_dim])
generated_image = model.generator(noise, training=False)

save_image(generated_image[0], "test")
decision = model.discriminator(generated_image)
print (decision)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, latent_dim])


def train(dataset, epochs):
  for epoch in range(start_epoch, epochs+start_epoch):
    start = time.time()

    mean_gen_loss = 0
    mean_disc_loss = 0
    for image_batch in dataset:
      gen_loss,disc_loss, generated_images = model.train_step(image_batch, batch_size)
      # print("MAXIMUM TRAIN VALUE: " + str(np.max(image_batch.numpy())))
      # print("MINIMMUM TRAIN VALUE: " + str(np.min(image_batch.numpy())))

      # print("MAXIMUM TRAIN VALUE: " + str(np.max(generated_images.numpy())))
      # print("MINIMMUM TRAIN VALUE: " + str(np.min(generated_images.numpy())))
      mean_gen_loss += gen_loss
      mean_disc_loss += disc_loss
    
    mean_gen_loss = mean_gen_loss.numpy()
    mean_disc_loss = mean_disc_loss.numpy()
    print("Generator LOSS: " + str(mean_gen_loss/len(dataset)))
    print("Discriminator LOSS: " + str(mean_disc_loss/len(dataset)))

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    save_image_matrix(model.generate_images(seed), img_path ='data_out/image_at_epoch_{:04d}'.format(epoch))

    # Save the model every 15 epochs
    if (epoch) % 15 == 0:
        model.generator.save_weights(generator_checkpoint_path.format(epoch=epoch))
        model.discriminator.save_weights(discriminator_checkpoint_path.format(epoch=epoch))

    print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))


  model.generator.save_weights(generator_checkpoint_path.format(epoch=epochs))
  model.discriminator.save_weights(discriminator_checkpoint_path.format(epoch=epochs))

  # Generate after the final epoch
  # save_image_matrix(model.generate_images(seed), img_path ='data_out/image_at_epoch_{:04d}'.format(start_epoch+epochs))


# model.generator.load_weights(generator_checkpoint_path.format(epoch=240))
# model.discriminator.load_weights(discriminator_checkpoint_path.format(epoch=240))
train(train_dataset, epochs)
save_gif('data_out/dcgan', re_images_name='data_out/image*.png')
save_mp4('data_out/dcgan', re_images_name='data_out/image*.png')