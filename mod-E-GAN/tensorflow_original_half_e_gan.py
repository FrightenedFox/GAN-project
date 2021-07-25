import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
from datetime import datetime
import time
from bit_operations import BitOps


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
                 enable_mutations=True,
                 num_examples_to_generate=16,
                 n_mut=50,
                 mut_prob=0.002):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_dim = noise_dim
        self.num_examples_to_generate = num_examples_to_generate
        self.show_images = False

        self.enable_mutations = enable_mutations
        self.n_mut = n_mut
        self.mut_prob = mut_prob
        self.n_successful_mutations, self.n_mutations = 0, 0
        self.layers_to_mutate = [
            "dense",
            "conv2d_transpose",
            "conv2d_transpose_1",
            "conv2d_transpose_2",
        ]

        self.model_name_suffix = (f"_m{self.enable_mutations}"
                                  f"_ep{self.epochs}_nm{self.n_mut}"
                                  f"_mp{self.mut_prob}")
        self.unique_model_name = (
            f"TF_orig_t{datetime.now().strftime('%m%d_%H%M%S')}"
            f"{self.model_name_suffix}"
        )
        os.makedirs(f"images/{self.unique_model_name}", exist_ok=True)
        self.anim_file = f"images/{self.unique_model_name}/dcgan.gif"

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

    def create_mutations(self):
        print(f"Calculating mutations with p = {self.mut_prob}")
        for layer_id, layer in enumerate(self.generator.layers):
            if layer.name in self.layers_to_mutate:
                weights = layer.get_weights()
                weights_len = len(weights)
                print(f"Layer # {layer_id} --- {layer.name}")
                template = [None for _ in range(weights_len)]
                mutated_layer_params = [template.copy()
                                        for _ in range(self.n_mut)]
                for param_id, param in enumerate(weights):
                    in_shape = param.shape
                    mut_engine = BitOps(param.flatten())
                    mut_engine.mutate(n_mut=self.n_mut, prob=self.mut_prob)
                    for mut_id in range(self.n_mut):
                        mutated_layer_params[mut_id][param_id] = mut_engine. \
                            mutations[mut_id].reshape(in_shape)
                    del mut_engine
                self.compare_mutations(mutated_layer_params, layer_id)

    def compare_mutations(self, mutated_weights, layer_id, n_tests=16):
        mut_loss_res = np.empty(len(mutated_weights))
        noise = tf.random.normal([n_tests, self.noise_dim])
        original_weights = self.generator.layers[layer_id].get_weights()
        gen_imgs = self.generator(noise, training=False)
        fake_out = self.discriminator(gen_imgs, training=False)
        curr_loss = self.cross_entropy(tf.zeros_like(fake_out), fake_out)
        for mut_id, weights in enumerate(mutated_weights):
            self.generator.layers[layer_id].set_weights(weights)
            gen_imgs = self.generator(noise, training=False)
            fake_out = self.discriminator(gen_imgs, training=False)
            mut_loss_res[mut_id] = self.cross_entropy(tf.zeros_like(fake_out),
                                                      fake_out)
        max_id = np.argmax(mut_loss_res)
        to_print_and_log = np.copy(mut_loss_res)
        to_print_and_log[::-1].sort()

        print(f"Current loss {curr_loss:.4}\n"
              f"Best mutations (larger is better):\t",
              to_print_and_log[:5])
        self.n_mutations += 1
        if mut_loss_res[max_id] > curr_loss:
            print("Applying new parameters!\n")
            self.generator.layers[layer_id].set_weights(
                mutated_weights[int(max_id)])
            self.n_successful_mutations += 1
        else:
            print("Mutation unsuccessful, keeping old parameters.\n")
            self.generator.layers[layer_id].set_weights(original_weights)

    def generate_and_save_images(self, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator(test_input, training=False)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'images/{self.unique_model_name}/image_{epoch:04d}.png')
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

            if self.enable_mutations:
                self.create_mutations()
                print(f"Mutation success rate: {self.n_successful_mutations}/"
                      f"{self.n_mutations}")

            print(f"{8*'-'}\tEpoch {epoch + 1} time: {time.time() - start} sec")


        # Generate after the final epoch
        self.generate_and_save_images(self.epochs,
                                      self.seed)

    def make_animation(self):
        with imageio.get_writer(self.anim_file, mode='I') as writer:
            filenames = glob.glob(f"images/{self.unique_model_name}/image*.png")
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)


if __name__ == "__main__":
    gan = TensorHalfEGAN(epochs=50,
                         mut_prob=0.002,
                         n_mut=50,
                         enable_mutations=True)
    gan.train()
    gan.make_animation()
