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

# @tf.function
# def train_step(model, x, optimizer):
#   """Executes one training step and returns the loss.

#   This function computes the loss and gradients, and uses the latter to
#   update the model's parameters.
#   """
#   with tf.GradientTape() as tape:
#     loss = compute_loss(model, x)
#   gradients = tape.gradient(loss, model.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  

# def plot_latent_images(model, n, digit_size=28):
#   """Plots n x n digit images decoded from the latent space."""

#   norm = tfp.distributions.Normal(0, 1)
#   grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
#   grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
#   image_width = digit_size*n
#   image_height = image_width
#   image = np.zeros((image_height, image_width))

#   for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#       z = np.array([[xi, yi]])
#       x_decoded = model.sample(z)
#       digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
#       image[i * digit_size: (i + 1) * digit_size,
#             j * digit_size: (j + 1) * digit_size] = digit.numpy()

#   plt.figure(figsize=(10, 10))
#   plt.imshow(image, cmap='Greys_r')
#   plt.axis('Off')
#   # plt.show()


# def inference_image(model, image):
#   reshaped_image = tf.expand_dims(image, axis=0)
#   mean, logvar = model.encode(reshaped_image)
#   z = model.reparameterize(mean, logvar)
#   predictions = model.sample(z)
#   return predictions[0, :, :, :]

def save_image(image,img_name):
  fig = plt.figure()
  plt.imshow(image)
  plt.axis('off')
  plt.savefig(img_name + '.png')
  

# def split_frames(frames, num_images_set1):
#     set1 = frames[:num_images_set1]
#     set2 = frames[num_images_set1:]
#     return set1, set2


############################################################################################################
# MAIN SCRIPT a finalizar usando CVAE como ejemplo
############################################################################################################
batch_size = 32
epochs = 100
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 100
num_examples_to_generate = 1
img_shape = 200
img_channels = 3


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


model = DCGAN(latent_dim, img_shape, img_channels)
noise = tf.random.normal([1, 100])
generated_image = model.generator(noise, training=False)

save_image(generated_image[0], "test")
decision = model.discriminator(generated_image)
print (decision)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, latent_dim])

generator_checkpoint_path = "training_1/cp-generator-{epoch:04d}.weights.h5"
discriminator_checkpoint_path = "training_1/cp-discriminator-{epoch:04d}.weights.h5"

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(int(np.sqrt(predictions.shape[0])), int(np.sqrt(predictions.shape[0])), i + 1)
    plt.imshow(predictions[i, :, :, :])
    plt.axis('off')

  plt.savefig('data_out/image_at_epoch_{:04d}.png'.format(epoch))
  # plt.show()


@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = model.generator(noise, training=True)

      real_output = model.discriminator(images, training=True)
      fake_output = model.discriminator(generated_images, training=True)

      gen_loss = model.generator_loss(fake_output)
      disc_loss = model.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, model.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminator.trainable_variables)

    model.generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generator.trainable_variables))
    model.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(model.generator,
                             epoch + 1,
                             seed)



    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        model.generator.save_weights(generator_checkpoint_path.format(epoch=epoch))
        model.discriminator.save_weights(discriminator_checkpoint_path.format(epoch=epoch))

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


train(train_dataset, epochs)

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(epochs)


anim_file = 'data_out/dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('data_out/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
  
embed.embed_file(anim_file)

# Make some predictions
save_img(inference_image(model, test_sample[0]),"test_sample_predicted_trained")
save_img(inference_image(model, random_images[0]),"chino_predicted_trained")
save_img(inference_image(model, random_images[1]),"test_image_predicted_trained")