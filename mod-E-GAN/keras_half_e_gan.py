from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np

from bit_operations import BitOps


class ModEGAN:
    def __init__(self, batch_size=128, n_mut=10):
        self.batch_size = batch_size
        self.n_mut = n_mut
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

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

    def create_mutations(self, model, **kwargs):

        layers = model.layers[1].get_weights()
        template = [None for _ in range(len(layers))]
        mutated_layers = [template for _ in range(self.n_mut)]
        for ind, layer in enumerate(layers):
            mut_engine = BitOps(layer.flatten())
            mut_engine.mutate(n_mut=self.n_mut, **kwargs)
            for mut_ind in range(self.n_mut):
                mutated_layers[mut_ind][ind] = mut_engine.\
                    mutations[mut_ind].reshape(layer.shape)
        return mutated_layers

    def compare_mutations(self, mutations, curr_loss, n_tests=10):
        mut_loss_res = np.empty(len(mutations))
        fake = np.zeros((n_tests, 1))
        noise = np.random.normal(0, 1, (n_tests, self.latent_dim))
        for mut_ind, mutation in enumerate(mutations):
            gen_eval = self.build_generator(summary=False)
            gen_eval.layers[1].set_weights(mutation)
            gen_eval_imgs = gen_eval.predict(noise)
            mut_loss_res[mut_ind] = self.discriminator.\
                test_on_batch(gen_eval_imgs, fake)[0]

        max_ind = np.argmax(mut_loss_res)
        to_print = np.copy(mut_loss_res)
        to_print[::-1].sort()
        print("\nCurrent loss %4f\nBest mutations:" % curr_loss[0],
              to_print[:6])
        if mut_loss_res[max_ind] > curr_loss[0]:
            return mutations[int(max_ind)]
        return None

    def train(self, epochs,
              sample_interval=50, mutation_interval=100, mutation_prob=0.05):
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

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

            # If at save interval => save generated image samples
            mut_counter, good_mut = 0, 0
            if epoch % mutation_interval == 0:
                mut_counter += 1
                print("Calculating mutations with p = %5f... " %
                      (mutation_prob / (np.sqrt(epoch) + 1)), end="")
                mutations = self.create_mutations(
                    self.generator,
                    prob=mutation_prob / (np.sqrt(epoch) + 1)
                )
                new_params = self.compare_mutations(mutations, d_loss_fake)
                if new_params is not None:
                    good_mut += 1
                    print("Applying new params!")
                    self.generator.layers[1].set_weights(new_params)
                else:
                    print("Mutation unsuccessful, keeping old params.")

            if epoch % sample_interval == 0:
                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], g_loss),
                      "[Mutations success rate %d/%d]" % (
                          good_mut, mut_counter))
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/ker_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = ModEGAN(batch_size=32)
    gan.train(epochs=1000, sample_interval=200, mutation_prob=0.02)
