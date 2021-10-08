import os
import glob
import argparse
from copy import deepcopy
from datetime import datetime
from collections import OrderedDict

import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from bit_operations import BitOps

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int,     default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int,   default=64, help="size of the batches")
parser.add_argument("--lr", type=float,         default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float,         default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float,         default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int,        default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int,   default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")

parser.add_argument("--enable_mut", type=bool,  default=False, help="whether to generate mutations")
parser.add_argument("--n_mut", type=int,        default=20, help="number of the mutations per parameter")
parser.add_argument("--mut_prob", type=float,   default=0.02, help="probability of the mutation")
parser.add_argument("--mut_interval", type=int, default=3, help="interval between mutations")
opt = parser.parse_args()
print(opt)

IMG_SIZE = 28
IMG_CHANNELS = 1
IMG_SHAPE = (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

# 12345 seed is added for reproducibility
# TODO: add project seed multiplier so that we could generate different
#       results if we want to do so


def initialize_weights_for_tanh(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight.data, 1)
        nn.init.constant_(model.bias.data, 0)


def initialize_weights_for_sigmoid(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight.data, 1)
        nn.init.constant_(model.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(IMG_SHAPE))),
            nn.Tanh()
        )

    def forward(self, z_):
        img = self.model(z_)
        img = img.view(img.size(0), *IMG_SHAPE)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(IMG_SHAPE)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Mutator:
    def __init__(self, gen_state_dict, disc_state_dict, n_mut=5, on_gpu=True):
        self.gen_state_dict = self.params2device(gen_state_dict, out_device="cpu")
        self.disc_state_dict = self.params2device(disc_state_dict, out_device="cpu")
        self.n_mut = n_mut
        self.mutated_dicts = [OrderedDict() for _ in range(n_mut)]
        self.on_gpu = on_gpu

    def create_mutations(self, **kwargs):
        """ Creates a list of mutated parameters of the model

        Parameters
        ----------
        kwargs:
            prob : float, optional
                Probability of mutation. Default 0.05.

            length : int, optional
                Length of the bitstring. Default 56.

            chunk_s : int, optional
                Size of the single chunk. Default 8.

        Returns
        -------
        List[OrderDict]
        """
        for key, param in self.gen_state_dict.items():
            mut_engine = BitOps(param.numpy().flatten())
            mut_engine.mutate(n_mut=self.n_mut, **kwargs)
            for mut_ind in range(self.n_mut):
                self.mutated_dicts[mut_ind][key] = torch.from_numpy(mut_engine.mutations[mut_ind].reshape(param.shape))
            del mut_engine
        return self.mutated_dicts

    def compare_mutations(self, mutated_params=None, test_batch=128):
        """ Compares mutations and returns either the best one or None.

        Parameters
        ----------
        mutated_params : list
            The list of the mutated parameters
        test_batch : int
            The size of the batch on which each mutation is tested

        Returns
        -------
        OrderedDict or None
            None if all mutations appeared to be worse then the original
            parameters and OrderedDict if there was at least one successful
            mutation.
        """
        # TODO: rewrite comparison so that there is no need for extra instance
        #  of the generator and discriminator
        gen_eval = Generator()
        disc_eval = Discriminator()
        adversarial_loss_eval = torch.nn.BCELoss()
        tensor_eval = torch.FloatTensor

        disc_eval.load_state_dict(self.disc_state_dict)
        valid_eval = Variable(tensor_eval(test_batch, 1).fill_(1.0), requires_grad=False)
        z_eval = Variable(tensor_eval(np.random.normal(0, 1, (test_batch, opt.latent_dim))))

        if mutated_params is None:
            mutated_params = self.mutated_dicts
        mut_res = np.empty(len(mutated_params))
        for mut_ind in range(len(mutated_params)):
            gen_eval.load_state_dict(mutated_params[mut_ind])
            with torch.no_grad():
                gen_imgs_eval = gen_eval(z_eval)
                loss_eval = adversarial_loss_eval(disc_eval(gen_imgs_eval), valid_eval)
            mut_res[mut_ind] = loss_eval
        gen_eval.load_state_dict(self.gen_state_dict)
        with torch.no_grad():
            curr_imgs = gen_eval(z_eval)
            curr_loss = adversarial_loss_eval(disc_eval(curr_imgs), valid_eval)
        min_ind = np.argmin(mut_res)
        to_print = np.copy(mut_res)
        to_print.sort()
        print(f"\nCurrent loss {curr_loss.numpy():.4f}\nBest mutations:", to_print[:6])
        if mut_res[min_ind] < curr_loss.numpy():
            return self.params2device(mutated_params[int(min_ind)], out_device="cuda:0" if self.on_gpu else "cpu")
        return None

    @staticmethod
    def params2device(in_params, out_device="cuda:0") -> OrderedDict:
        """ Converts state dictionary from CPU to GPU and vice versa.

        Parameters
        ----------
        in_params : OrderedDict[str, torch.self.Tensor]
            Input model state dictionary.
        out_device : {"cpu", "cuda:0"}, optional
            Device to which input has to be converted. Default is "cuda:0".

        Returns
        -------
        out_params : OrderedDict[str, Any]
            Converted state dictionary to the appropriate device.
        """
        out_params = OrderedDict()
        for key, param in in_params.items():
            out_params[key] = param.to(out_device)
        return out_params


# noinspection PyUnresolvedReferences
class ModelTraining:
    SAVE_STATE_DICT_AFTER_EACH_EPOCH = False
    mut_probabilities_list = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

    def __init__(self):
        # Folders and files naming handling
        model_name_suffix = (f"_m{opt.enable_mut}"
                             f"_ep{opt.n_epochs}_bs{opt.batch_size}"
                             f"_nm{opt.n_mut}_mp{opt.mut_prob}"
                             f"_mi{opt.mut_interval}")
        self.unique_model_name = f"PyTorch_t{datetime.now().strftime('%m%d_%H%M%S')}{model_name_suffix}"
        os.makedirs(f"images/{self.unique_model_name}/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/media/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/samples/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/logs/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/models/", exist_ok=True)

        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        # Initialize models
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.apply(initialize_weights_for_tanh)
        self.discriminator.apply(initialize_weights_for_sigmoid)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

        # Handling training on GPU
        self.cuda = True if torch.cuda.is_available() else False
        print("Is cuda enabled?", "YES" if self.cuda else "NO")
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()

        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Generate seed for consistent images
        # TODO: multiply 12345 by the "project seed multiplier"
        self.rng = np.random.default_rng(12345)
        self.z_seed = Variable(self.Tensor(self.rng.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Initialize other training properties
        self.epoch = 0
        self.batches_done = 0
        self.good_mut = 0
        self.mut_counter = 0
        self.g_loss = None
        self.d_loss = None
        self.stats = {
            "g_loss": [],
            "d_loss": [],
        }
        # NOTE: consider making a list of specific values, not just a single one
        self._return_specific_epoch_state = False
        self._specific_epoch_state = 0
        self._model_suffix = "n"
        self._have_state_dicts = [self.generator, self.discriminator, self.optimizer_G, self.optimizer_D]
        self._enable_comparison_sequence = False

    def train_epoch(self, imgs):
        # Adversarial ground truths
        valid = Variable(self.Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(self.Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(self.Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        self.optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(self.Tensor(self.rng.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = self.generator(z)

        # Loss measures generator's ability to fool the discriminator
        self.g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

        self.g_loss.backward()
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        self.d_loss = (real_loss + fake_loss) / 2

        self.d_loss.backward()
        self.optimizer_D.step()

    def trainer(self):
        specific_epoch_state = None
        # Configure data loader
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "..",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(IMG_SIZE),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

        # Start training
        while self.epoch < opt.n_epochs:
            self.epoch += 1
            for i, (imgs, _) in enumerate(dataloader):
                self.batches_done = (self.epoch - 1) * len(dataloader) + i
                self.train_epoch(imgs)
                self.collect_stats()

                if self.batches_done % opt.sample_interval == 0:
                    self.save_image()
                    self.print_stats(len(dataloader))

            if self.epoch % opt.mut_interval == 0 and self.epoch != 0:
                if self._enable_comparison_sequence:
                    self.rng = np.random.default_rng(self.epoch)
                    torch.manual_seed(self.epoch)
                if opt.enable_mut:
                    self.mutation_handler()

            if self._return_specific_epoch_state and self.epoch == self._specific_epoch_state:
                specific_epoch_state = [deepcopy(m.state_dict()) for m in self._have_state_dicts]

            if self.SAVE_STATE_DICT_AFTER_EACH_EPOCH:
                self.save_model_state(self._have_state_dicts,
                                      [f"gen_ep{self.epoch}", f"disc_ep{self.epoch}",
                                       f"optimG_ep{self.epoch}", f"optimD_ep{self.epoch}"])
        return specific_epoch_state

    def mutation_handler(self, old_gen_state_dict=None, disc_state_dict=None):
        self.mut_counter += 1
        if old_gen_state_dict is None:
            old_gen_state_dict = self.generator.state_dict()
        if disc_state_dict is None:
            disc_state_dict = self.discriminator.state_dict()
        em = Mutator(old_gen_state_dict, disc_state_dict, n_mut=opt.n_mut)
        print("Starting mutation sequence:")
        best_of_the_best = []

        # Generate and choose the best mutation for each probability
        for mut_prob in self.mut_probabilities_list:
            print(f"prob = {mut_prob:.5%}", end="\t")
            em.create_mutations(prob=mut_prob)
            new_params = em.compare_mutations()
            if new_params is not None:
                print("Adding the best mutation to the final comparison list")
                best_of_the_best.append(new_params)
            else:
                print("Mutation unsuccessful, skipping.")

        # Compare the best mutations and choose 'the best of the best'
        if len(best_of_the_best) > 1:
            print("Starting to compare best mutations:")
            best_params = em.compare_mutations(best_of_the_best)
            if best_params is not None:
                print("Applying new params!")
                self.good_mut += 1
                return best_params
            else:
                print("Mutation unsuccessful, keeping old params.")
        elif len(best_of_the_best) == 1:
            print("Applying the only mutation which passed.")
            self.good_mut += 1
            return best_of_the_best[0]
        else:
            print("Mutation unsuccessful, keeping old params.")
            return None

    def apply_mutation(self, new_state_dict):
        self.save_image("b")
        self.generator.load_state_dict(new_state_dict)
        self.save_image("a")

    def save_image(self, prefix=""):
        gen_imgs = self.generator(self.z_seed)
        save_image(gen_imgs.data[:36],
                   f"images/{self.unique_model_name}/samples/"
                   f"{prefix}{self.batches_done}_{self._model_suffix}.png",
                   nrow=6,
                   normalize=True)

    def print_stats(self, dataloader_length):
        print(f"[Epoch {self.epoch:d}/{opt.n_epochs}] "
              f"[Batch {self.batches_done:d}/{opt.n_epochs * dataloader_length:d}] "
              f"[D loss: {self.d_loss.item():.3f}] "
              f"[G loss: {self.g_loss.item():.3f}] "
              f"[Mutations success rate {self.good_mut:d}/{self.mut_counter:d}]")

    def collect_stats(self):
        self.stats["d_loss"].append(self.d_loss.item())
        self.stats["g_loss"].append(self.g_loss.item())

    def plot_stats(self, smoothing=40, filename="plot"):
        plt.style.use("ggplot")
        smoothed_stats_df = pd.DataFrame(self.stats).rolling(smoothing, center=True).mean()

        fig, ax = plt.subplots()
        ax.plot(smoothed_stats_df.g_loss, label="Generator")
        ax.plot(smoothed_stats_df.d_loss, label="Discriminator")
        ax.legend()
        ax.set_title("Generator and discriminator loss plot")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Batch")

        plt.savefig(f"images/{mt.unique_model_name}/media/{filename}.png", dpi=300)

    def make_animation(self, filename="animation"):
        output_file = f"images/{self.unique_model_name}/media/{filename}.gif"
        input_data = f"images/{self.unique_model_name}/samples/*.png"
        with imageio.get_writer(output_file, mode='I') as writer:
            filenames = glob.glob(input_data)
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    def save_stats(self, filename="stats"):
        stats_df = pd.DataFrame(self.stats)
        stats_df.to_json(f"images/{self.unique_model_name}/logs/{filename}.json")
        del stats_df

    def save_model_training_info(self):
        # TODO: make model info file
        pass

    def save_model_state(self, models, filenames):
        for model, filename in zip(models, filenames):
            torch.save(model.state_dict(), f"images/{self.unique_model_name}/models/{filename}.pth")

    def comparison_sequence(self, compare_mutations=True):
        self._enable_comparison_sequence = True
        self._return_specific_epoch_state = True
        # TODO: raise exception if mutations are enabled
        key_epochs = list(range(opt.mut_interval, opt.n_epochs + 1, opt.mut_interval))

        for epoch_num in key_epochs:
            print(f"{10 * '-'} Starting a new sequence {10 * '-'}")
            self._specific_epoch_state = epoch_num
            self._model_suffix = str(epoch_num)

            # Train and save all stats
            states_from_i = mt.trainer()
            mt.plot_stats(filename=f"plot_{epoch_num}")
            mt.save_stats(filename=f"stats_{epoch_num}")
            self.save_model_state(self._have_state_dicts,
                                  [f"gen_finish{epoch_num}", f"disc_finish{epoch_num}",
                                   f"optimG_finish{epoch_num}", f"optimD_finish{epoch_num}"])

            if compare_mutations:
                mutated_gen_state_dict = self.mutation_handler(old_gen_state_dict=states_from_i[0],
                                                               disc_state_dict=states_from_i[1])
                if mutated_gen_state_dict is not None:
                    states_from_i[0] = mutated_gen_state_dict

            # Prepare for the next training
            for m, state in zip(self._have_state_dicts, states_from_i):
                m.load_state_dict(state)
            self.rng = np.random.default_rng(epoch_num)
            torch.manual_seed(epoch_num)
            self.epoch = epoch_num
            self.stats["g_loss"], self.stats["d_loss"] = [], []


if __name__ == "__main__":
    mt = ModelTraining()
    mt.comparison_sequence()
