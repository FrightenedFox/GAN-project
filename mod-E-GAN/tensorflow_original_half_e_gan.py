import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    # Note: None is the batch size
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1),
                                     padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False,
                                     activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


class TensorHalfEGAN:

    def __init__(self,
                 buffer_size=60000,
                 batch_size=256,
                 epochs=50,
                 noise_dim=100,
                 num_examples_to_generate=16):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_dim = noise_dim
        self.num_examples_to_generate = num_examples_to_generate
        self.anim_file = 'images/dcgan.gif'
        self.show_images = False

        self.seed = tf.random.normal([num_examples_to_generate, noise_dim])
        self.train_dataset = self.load_data()

        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    def load_data(self):
        (train_imgs, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

        train_imgs = train_imgs.reshape(
            train_imgs.shape[0], 28, 28, 1).astype('float32')
        # Normalize the images to [-1, 1]
        train_imgs = (train_imgs - 127.5) / 127.5

        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(
            train_imgs).shuffle(self.buffer_size).batch(self.batch_size)
        return train_dataset

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss,
            self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(zip(
            gradients_of_generator,
            self.generator.trainable_variables
        ))
        self.discriminator_optimizer.apply_gradients(zip(
            gradients_of_discriminator,
            self.discriminator.trainable_variables
        ))

    def generate_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator(test_input, training=False)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'images/image_at_epoch_{epoch:04d}.png')
        if self.show_images:
            plt.show()

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()

            for image_batch in self.train_dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as you go
            self.generate_and_save_images(epoch + 1,
                                          self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

        # Generate after the final epoch
        self.generate_and_save_images(self.epochs,
                                      self.seed)

    def make_animation(self):
        with imageio.get_writer(self.anim_file, mode='I') as writer:
            filenames = glob.glob('images/image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)


if __name__ == "__main__":
    gan = TensorHalfEGAN(epochs=5,)
    gan.train()
    gan.make_animation()
