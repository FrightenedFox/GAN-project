from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

from bit_operations import BitOps


class ModEGAN:

    def __init__(self,
                 batch_size=128,
                 epochs=30_000,
                 sample_interval=50,
                 enable_mutations=True,
                 mutation_interval=3000,
                 n_mut=10,
                 mutation_prob=0.02,
                 combined_mutation_mode=True):

        # General attributes
        self.batch_size = batch_size
        self.epochs = epochs

        # Mutations and selection attributes
        self.enable_mutations = enable_mutations
        self.mutation_interval = mutation_interval
        self.n_mut = n_mut
        self.mutation_prob = mutation_prob
        self.mut_prob_reducer = 40
        self.enable_selection = False
        self.n_selections = 1
        # Define whether to update parameters on the self.combined or
        # the self.generator:
        self._combined_mutation_mode = combined_mutation_mode

        # Input attributes
        self.img_rows, self.img_cols, self.channels = self.img_shape = (
            28, 28, 1
        )
        self.latent_dim = 100

        # Training tuning attributes
        self.lr = 2e-4  # learning rate
        self.beta1 = 0.5

        # Samples and logs
        self.sample_interval = sample_interval
        self.sample_image_rows, self.sample_image_cols = 5, 5
        self.collect_logs = True
        self.log_interval = 10

        self.model_name_suffix = (f"_m{self.enable_mutations}"
                                  f"_ep{self.epochs}_bs{self.batch_size}"
                                  f"_nm{self.n_mut}_mp{self.mutation_prob}"
                                  f"_mi{self.mutation_interval}")
        self.UNIQUE_MODEL_NAME = (
            f"Mod-E-GAN-Keras-{self.model_name_suffix}"
            f"_t{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

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
        self.train_step, self.train_writer = None, None

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

    def get_model_name(self):
        self.update_model_name()
        return self.UNIQUE_MODEL_NAME

    def update_model_name(self):
        """ Updates model name if it has been changed after model
        initialization
        """
        self.update_model_suffix()
        self.UNIQUE_MODEL_NAME = (
            f"Mod_E_GAN_t{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f"{self.model_name_suffix}"
        )

    def get_model_suffix(self):
        self.update_model_suffix()
        return self.model_name_suffix

    def update_model_suffix(self):
        """ Updates model suffix if it has been changed after model
        initialization
        """
        self.model_name_suffix = (f"_m{self.enable_mutations}"
                                  f"_ep{self.epochs}_bs{self.batch_size}"
                                  f"_nm{self.n_mut}_mp{self.mutation_prob}"
                                  f"_mi{self.mutation_interval}")

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

    def create_mutations(self, weights, **kwargs):
        template = [None for _ in range(len(weights))]
        n_mut_and_sel = self.n_mut
        if self.enable_selection:
            n_mut_and_sel *= (self.n_selections + 1)
        mutated_params = [template.copy() for _ in range(n_mut_and_sel)]
        for param_ind, param in enumerate(weights):
            mut_engine = BitOps(param.flatten())
            mut_engine.mutate(n_mut=self.n_mut,
                              apply_selection=self.enable_selection,
                              n_selections=self.n_selections,
                              **kwargs)
            for mut_ind in range(n_mut_and_sel):
                mutated_params[mut_ind][param_ind] = mut_engine.\
                    mutations[mut_ind].reshape(param.shape)
        del mut_engine
        return mutated_params

    def compare_mutations(self, mutations, mut_log_number=0, n_tests=16):
        mut_loss_res = np.empty(len(mutations))
        noise = np.random.normal(0, 1, (n_tests, self.latent_dim))
        if self._combined_mutation_mode:
            valid = np.ones((n_tests, 1))
            original_parameters = self.combined.layers[1].get_weights()
            curr_loss = self.combined.test_on_batch(noise, valid)
            for mut_ind, mutation in enumerate(mutations):
                self.combined.layers[1].set_weights(mutation)
                mut_loss_res[mut_ind] = self.combined.test_on_batch(noise,
                                                                    valid)

            self.combined.layers[1].set_weights(original_parameters)
            extreme_direction = "smaller"
            extreme_ind = np.argmin(mut_loss_res)
            to_print_and_log = np.copy(mut_loss_res)
            to_print_and_log.sort()

        else:
            fake = np.zeros((n_tests, 1))
            curr_loss = self.discriminator.test_on_batch(
                self.generator(noise),
                fake
            )[0]
            original_parameters = self.discriminator.layers[1].get_weights()
            for mut_ind, mutation in enumerate(mutations):
                self.discriminator.layers[1].set_weights(mutation)
                gen_eval_imgs = self.discriminator.predict(noise)
                mut_loss_res[mut_ind] = self.discriminator. \
                    test_on_batch(gen_eval_imgs, fake)[0]

            self.generator.layers[1].set_weights(original_parameters)
            extreme_direction = "larger"
            extreme_ind = np.argmax(mut_loss_res)
            to_print_and_log = np.copy(mut_loss_res)
            to_print_and_log[::-1].sort()

        self.mutations_log[mut_log_number][0] = curr_loss
        self.mutations_log[mut_log_number][1:] = to_print_and_log[
            0:self.n_of_mut_res_to_log - 1
        ]
        print(f"Current loss {curr_loss:.4}\n"
              f"Best mutations ({extreme_direction} is better):\n\t",
              to_print_and_log[:3], to_print_and_log[-3:])
        if ((mut_loss_res[extreme_ind] > curr_loss) ^
                self._combined_mutation_mode):
            return mutations[int(extreme_ind)], mut_loss_res[extreme_ind]
        return None, None

    def train(self):
        self.update_model_name()
        imgs_folder_name = f"{self.UNIQUE_MODEL_NAME}"
        os.makedirs(f"images/{imgs_folder_name}", exist_ok=True)

        # If collect logs => generate folders and writer for the logs
        if self.collect_logs:
            self.train_step = 0

            self.train_writer = tf.summary.create_file_writer(
                f"logs/TensorBoard/{self.UNIQUE_MODEL_NAME}"
            )
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
            if epoch % self.log_interval == 0 and self.collect_logs:
                self.log_ar[0][epoch] = d_loss_real
                self.log_ar[1][epoch] = d_loss_fake
                self.log_ar[2][epoch] = d_loss
                self.log_ar[3][epoch] = g_loss
                self.log_ar[4][epoch] = mut_success_rate
                self.log_tf_summary(g_loss, *d_loss)

            # If at mutation interval => create and verify new mutations
            logs_were_collected_during_mut = False
            if epoch % self.mutation_interval == 0 and self.enable_mutations \
                    and epoch:  # equivalent to epoch != 0
                mut_success_rate[1] += 1

                # If mutation probability reducer is not None, then
                # reduce default mutation probability
                mut_probability = self.mutation_prob
                if self.mut_prob_reducer is not None:
                    mut_probability /= epoch / self.mut_prob_reducer + 1

                if self._combined_mutation_mode:
                    weights = self.combined.layers[1].get_weights()
                else:
                    weights = self.generator.layers[1].get_weights()

                print(f"\nCalculating mutations with p = {mut_probability:.4%}")
                new_params, new_d_loss_fake = self.compare_mutations(
                    self.create_mutations(weights, prob=mut_probability)
                )

                # Use mutated parameters of the model
                # if they better obfuscate the discriminator
                if new_params is not None:
                    mut_success_rate[0] += 1
                    print("Applying new parameters!\n")
                    if self._combined_mutation_mode:
                        self.combined.layers[1].set_weights(new_params)
                    else:
                        self.generator.layers[1].set_weights(new_params)

                    # Rewriting logs with new values after mutation
                    new_d_loss = 0.5 * np.add(d_loss_real[0], new_d_loss_fake)
                    if self.collect_logs:
                        logs_were_collected_during_mut = True
                        self.log_tf_summary(g_loss, new_d_loss,
                                            include_layers=True,
                                            add_step=False)
                        self.train_writer.flush()
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

                # Update TensorBoard stats
                if self.collect_logs and not logs_were_collected_during_mut:
                    self.log_tf_summary(g_loss, *d_loss,
                                        include_layers=True,
                                        include_image_sample=True,
                                        add_step=False)
                    self.train_writer.flush()

                # Generate the image
                image_sample = self.generate_image_sample()
                self.save_images(image_sample, imgs_folder_name, epoch)

    def generate_image_sample(self):
        noise = np.random.normal(
            0, 1,
            (self.sample_image_rows * self.sample_image_cols, self.latent_dim)
        )
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        return 0.5 * gen_imgs + 0.5

    def save_images(self, image_sample, folder, epoch=-1):
        fig, axs = plt.subplots(self.sample_image_rows, self.sample_image_cols)
        cnt = 0
        for i in range(self.sample_image_rows):
            for j in range(self.sample_image_cols):
                axs[i, j].imshow(image_sample[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(f"images/{folder}/{epoch}.png")
        plt.close()

    def log_tf_summary(self, g_loss, d_loss, d_accuracy=None,
                       include_layers=False, include_image_sample=False,
                       add_step=True):
        if add_step:
            self.train_step += 1
        with self.train_writer.as_default():
            tf.summary.scalar("Generator loss",
                              g_loss, step=self.train_step)
            tf.summary.scalar("Discriminator loss",
                              d_loss, step=self.train_step)
            if d_accuracy is not None:
                tf.summary.scalar("Discriminator accuracy",
                                  d_accuracy, step=self.train_step)

            if include_image_sample:
                image_sample = self.generate_image_sample()
                tf.summary.image("Image Sample", image_sample, step=0)

            if include_layers:
                dict_of_layers = {
                    "Gen": self.generator.layers,
                    "Disc": self.discriminator.layers,
                    "Comb": self.combined.layers,
                }
                for model_name, model_layers in dict_of_layers.items():
                    for layer_ind in range(1, len(model_layers)):
                        for param_ind, param in enumerate(
                                model_layers[layer_ind].get_weights()
                        ):
                            tf.summary.histogram(
                                f"{model_name}_lid{layer_ind}_pid{param_ind}",
                                data=param,
                                step=self.train_step
                            )

    def save_log_info(self, path="logs/"):
        # Creating DataFrames from the collected log info
        self.update_model_suffix()
        self.log_df = pd.DataFrame({
            "d_loss_real":          self.log_ar[0, :, 0],
            "d_loss_real_acc":      self.log_ar[0, :, 1],
            "d_loss_fake":          self.log_ar[1, :, 0],
            "d_loss_fake_acc":      self.log_ar[1, :, 1],
            "average_d_loss":       self.log_ar[2, :, 0],
            "average_d_loss_acc":   self.log_ar[2, :, 1],
            "g_loss":               self.log_ar[3, :, 0],
            "g_loss_acc":           self.log_ar[3, :, 1],
            "successful_mut":       self.log_ar[4, :, 0],
            "mut_counter":          self.log_ar[4, :, 1],
        })
        if self.mutation_interval == 1 and self.enable_mutations:
            self.log_df["original_d_loss"] = self.mutations_log[:, 0]
            self.log_df["best_mut_d_loss"] = self.mutations_log[:, 1]

        self.mutations_log_df = pd.DataFrame(self.mutations_log)

        # Writing down collected log info
        os.makedirs(path, exist_ok=True)
        self.log_df.to_csv(f"{path}loss_log{self.model_name_suffix}.csv")
        if self.enable_mutations:
            self.mutations_log_df.to_csv(f"{path}mut_log"
                                         f"{self.model_name_suffix}.csv")


if __name__ == '__main__':
    gan = ModEGAN(
        epochs=30_000,
        batch_size=32,
        sample_interval=200,
        enable_mutations=True,
        n_mut=250,
        mutation_prob=0.0008,
        mutation_interval=1000,
        combined_mutation_mode=True
    )
    gan.mut_prob_reducer = None
    # gan.collect_logs = False
    # gan.enable_selection = True
    gan.train()
