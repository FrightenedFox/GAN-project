from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

from bit_operations import BitOps


class ModEGAN:

    UNIQUE_MODEL_NAME = f"Mod-E-GAN-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def __init__(self,
                 batch_size=128,
                 epochs=30_000,
                 sample_interval=50,
                 mutation_interval=3000,
                 n_mut=10,
                 mutation_prob=0.02):
        # General attributes
        self.batch_size = batch_size
        self.epochs = epochs
        self.sample_interval = sample_interval

        # Mutations attributes
        self.enable_mutations = True
        self.n_mut = n_mut
        self.mutation_prob = mutation_prob
        self.mutation_interval = mutation_interval

        self.img_rows, self.img_cols, self.channels = self.img_shape = (
            28, 28, 1
        )
        self.latent_dim = 100
        self.lr = 2e-4  # learning rate
        self.beta1 = 0.5
        self.mut_prob_reducer = 40
        self.sample_image_rows, self.sample_image_cols = 5, 5

        os.makedirs("images", exist_ok=True)

        # Adding logging variables
        self.log_ar = np.array([
            np.empty((epochs, 2)),  # d_loss_real
            np.empty((epochs, 2)),  # d_loss_fake
            np.empty((epochs, 2)),  # average d_loss
            np.empty((epochs, 2)),  # g_loss
            np.empty((epochs, 2)),  # mut success rate
        ])
        self.n_of_mut_res_to_log = np.ceil(n_mut / 10 + 1).astype("int32")
        number_of_mut = np.ceil(epochs / mutation_interval).astype("int32")
        self.mutations_log = np.empty(
            (number_of_mut, self.n_of_mut_res_to_log)
        )
        self.log_df, self.mutations_log_df = None, None

        optimizer = Adam(self.lr, self.beta1)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images
        # as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Initializing the generator to compare future mutations.
        self.mut_compare_generator = self.build_generator(summary=False)
        self.mut_compare_generator.trainable = False

        self.train_step = 0
        self.train_writer = tf.summary.create_file_writer(
            f"logs/TensorBoard/{UNIQUE_MODEL_NAME}"
        )

    def build_generator(self, summary=True):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        if summary:
            model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def create_mutations(self, **kwargs):
        layers = self.generator.layers[1].get_weights()
        template = [None for _ in range(len(layers))]
        mutated_layers = [template.copy() for _ in range(self.n_mut)]
        for layer_ind, layer in enumerate(layers):
            mut_engine = BitOps(layer.flatten())
            mut_engine.mutate(n_mut=self.n_mut, **kwargs)
            for mut_ind in range(self.n_mut):
                mutated_layers[mut_ind][layer_ind] = mut_engine.\
                    mutations[mut_ind].reshape(layer.shape)
        return mutated_layers

    def compare_mutations(self, mutations, mut_log_number=0, n_tests=20):
        mut_loss_res = np.empty(len(mutations))
        fake = np.zeros((n_tests, 1))
        noise = np.random.normal(0, 1, (n_tests, self.latent_dim))
        for mut_ind, mutation in enumerate(mutations):
            self.mut_compare_generator.layers[1].set_weights(mutation)
            gen_eval_imgs = self.mut_compare_generator.predict(noise)
            mut_loss_res[mut_ind] = self.discriminator.\
                test_on_batch(gen_eval_imgs, fake)[0]

        curr_loss = self.discriminator.test_on_batch(
            self.generator(noise),
            fake
        )[0]
        max_ind = np.argmax(mut_loss_res)
        to_print_and_log = np.copy(mut_loss_res)
        to_print_and_log[::-1].sort()
        self.mutations_log[mut_log_number][0] = curr_loss
        self.mutations_log[mut_log_number][1:] = to_print_and_log[
            0:self.n_of_mut_res_to_log - 1
        ]
        print(f"Current loss {curr_loss:.4}\n"
              f"Best mutations (larger is better):\n\t",
              to_print_and_log[:6])
        if mut_loss_res[max_ind] > curr_loss:
            self.log_discriminator_tf_summary(mut_loss_res[max_ind])
            return mutations[int(max_ind)]
        return None

    def train(self):
        # Load the dataset
        (x_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        mut_success_rate = np.zeros(2, dtype="int32")
        for epoch in range(self.epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

            # Train the generator
            # (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Collect log info
            self.log_ar[0][epoch] = d_loss_real
            self.log_ar[1][epoch] = d_loss_fake
            self.log_ar[2][epoch] = d_loss
            self.log_ar[3][epoch] = g_loss
            self.log_ar[4][epoch] = mut_success_rate
            self.log_discriminator_tf_summary(*d_loss)

            # If at mutation interval => create and verify new mutations
            if epoch % self.mutation_interval == 0 and self.enable_mutations:
                mut_success_rate[1] += 1

                # If mutation probability reducer is not None, then
                # reduce default mutation probability
                mut_probability = self.mutation_prob
                if self.mut_prob_reducer is not None:
                    mut_probability /= epoch / self.mut_prob_reducer + 1

                print(f"\nCalculating mutations with p = {mut_probability:.3%}")
                mutations = self.create_mutations(prob=mut_probability)
                new_params = self.compare_mutations(mutations)

                # Use mutated parameters of the model
                # if they better obfuscate the discriminator
                if new_params is not None:
                    mut_success_rate[0] += 1
                    print("Applying new parameters!")
                    self.generator.layers[1].set_weights(new_params)
                else:
                    print("Mutation unsuccessful, keeping old parameters.\n")

            # If at save interval => save generated image samples
            if epoch % self.sample_interval == 0:
                # Plot the progress
                print(
                    f"## {epoch} ## "
                    f"[D loss: {d_loss[0]:.4}, acc.: {d_loss[1]:.2%}] "
                    f"[G loss: {g_loss:.4}] [Mutations success rate "
                    f"{mut_success_rate[0]}/{mut_success_rate[1]}]"
                )
                self.sample_images(epoch)

    def sample_images(self, epoch):
        noise = np.random.normal(
            0, 1,
            (self.sample_image_rows * self.sample_image_cols, self.latent_dim)
        )
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(self.sample_image_rows, self.sample_image_cols)
        cnt = 0
        for i in range(self.sample_image_rows):
            for j in range(self.sample_image_cols):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"images/ker_{epoch}_m{self.enable_mutations}.png")
        plt.close()

    def log_discriminator_tf_summary(self, loss, accuracy=None):
        with self.train_writer.as_default():
            tf.summary.scalar("Loss", loss, step=self.train_step)
            if accuracy is not None:
                tf.summary.scalar("Accuracy", accuracy, step=self.train_step)
            self.train_step += 1

    def save_log_info(self, path="logs/"):
        # Creating DataFrames from the collected log info
        self.log_df = pd.DataFrame({
            "d_loss_real": self.log_ar[0, :, 0],
            "d_loss_real_acc": self.log_ar[0, :, 1],
            "d_loss_fake": self.log_ar[1, :, 0],
            "d_loss_fake_acc": self.log_ar[1, :, 1],
            "average_d_loss": self.log_ar[2, :, 0],
            "average_d_loss_acc": self.log_ar[2, :, 1],
            "g_loss": self.log_ar[3, :, 0],
            "g_loss_acc": self.log_ar[3, :, 1],
            "successful_mut": self.log_ar[4, :, 0],
            "mut_counter": self.log_ar[4, :, 1],
        })
        if self.mutation_interval == 1 and self.enable_mutations:
            self.log_df["original_d_loss"] = self.mutations_log[:, 0]
            self.log_df["best_mut_d_loss"] = self.mutations_log[:, 1]

        self.mutations_log_df = pd.DataFrame(self.mutations_log)
        filename_suffix = (f"_ep{self.epochs}_bs{self.batch_size}"
                           f"_nm{self.n_mut}_mp{self.mutation_prob}"
                           f"_mi{self.mutation_interval}"
                           f"_m{self.enable_mutations}")

        # Writing down collected log info
        os.makedirs(path, exist_ok=True)
        self.log_df.to_csv(f"{path}loss_log{filename_suffix}.csv")
        if self.enable_mutations:
            self.mutations_log_df.to_csv(f"{path}mut_log{filename_suffix}.csv")


if __name__ == '__main__':
    gan = ModEGAN(
        epochs=30_000,
        batch_size=32,
        sample_interval=200,
        n_mut=150,
        mutation_prob=0.02,
        mutation_interval=1000,
    )
    gan.enable_mutations = True
    gan.train()
    gan.save_log_info()
