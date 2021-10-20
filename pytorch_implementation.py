import os
import glob
import logging
import argparse
from sys import stdout
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

IMG_SIZE = 28
IMG_CHANNELS = 1
IMG_SHAPE = (IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
MODEL_SEED = 23456

# Setup logging so that messages are printed to both stdout and logfile
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(stdout)
logger.addHandler(stdout_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int,     default=80, help="number of epochs of training")
parser.add_argument("--batch_size", type=int,   default=64, help="size of the batches")
parser.add_argument("--lr", type=float,         default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float,         default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float,         default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int,        default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int,   default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")

parser.add_argument("--enable_mut", type=bool,  default=False, help="whether to generate mutations")
parser.add_argument("--n_mut", type=int,        default=30, help="number of the mutations per parameter")
parser.add_argument("--mut_interval", type=int, default=4, help="interval between mutations")
opt = parser.parse_args()
logger.info(opt)


def initialize_weights_for_tanh(model):
    """Initializes model parameters using xavier uniform distribution"""
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight.data, 1)
        nn.init.constant_(model.bias.data, 0)


def initialize_weights_for_sigmoid(model):
    """Initializes model parameters using xavier normal distribution"""
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight.data, 1)
        nn.init.constant_(model.bias.data, 0)


def x_cross_comparison(disc_v1_dict, gen_v1_dict,
                       disc_v2_dict, gen_v2_dict,
                       optimD_v1_dict, optimG_v1_dict,
                       optimD_v2_dict, optimG_v2_dict,
                       comparison_seed=MODEL_SEED,
                       batch_size=opt.batch_size,
                       n_batches=5, start_epoch=0,
                       n_epochs=20):
    # Discriminator and generator weights are SHUFFLED
    xccmt1 = ModelTraining(mt_seed=comparison_seed, mt_id=1)
    xccmt1_state = [
        gen_v1_dict,    disc_v2_dict,
        optimG_v1_dict, optimD_v2_dict,
    ]
    xccmt1.load_model_state(xccmt1_state)

    xccmt2 = ModelTraining(mt_seed=comparison_seed, mt_id=2)
    xccmt2_state = [
        gen_v2_dict, disc_v1_dict,
        optimG_v2_dict, optimD_v1_dict,
    ]
    xccmt2.load_model_state(xccmt2_state)

    g1_losses, g2_losses = [], []
    valid = Variable(xccmt1.Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
    with torch.no_grad():
        for _ in range(n_batches):
            z = Variable(xccmt1.Tensor(xccmt1.rng.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_imgs_v1 = xccmt1.generator(z)
            gen_imgs_v2 = xccmt2.generator(z)
            g1_losses.append(xccmt1.adversarial_loss(xccmt1.discriminator(gen_imgs_v1), valid).item())
            g2_losses.append(xccmt2.adversarial_loss(xccmt2.discriminator(gen_imgs_v2), valid).item())

    logger.info(f"G1 vs D2 mean = {np.mean(g1_losses)}\n\tG1 vs D2 losses: {g1_losses}.")
    logger.info(f"G2 vs D1 mean = {np.mean(g2_losses)}\n\tG2 vs D1 losses: {g2_losses}.")

    for mt in [xccmt1, xccmt2]:
        mt.epoch = start_epoch
        mt.n_epochs = start_epoch + n_epochs
        mt.reset_seeds()
        mt.trainer()
        mt.save_stats()
        mt.final_result_show_off()
        mt.save_model_state(mt.have_state_dicts,
                            [f"gen_finish{mt.epoch}", f"disc_finish{mt.epoch}",
                             f"optimG_finish{mt.epoch}", f"optimD_finish{mt.epoch}"])


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
        logger.info(f"\nCurrent loss {curr_loss.numpy():.4f}\nBest mutations: {to_print[:6]}")
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


class ModelTraining:
    SAVE_STATE_DICT_AFTER_EACH_EPOCH = False
    mut_probabilities_list = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]

    def __init__(self, mt_seed=MODEL_SEED, mt_id="_"):
        # Folders and files naming handling
        self.unique_model_name = f"PyTorch_t{datetime.now().strftime('%m%d_%H%M%S')}_id{mt_id}"
        os.makedirs(f"images/{self.unique_model_name}/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/media/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/samples/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/logs/", exist_ok=True)
        os.makedirs(f"images/{self.unique_model_name}/models/", exist_ok=True)

        # Start logging
        output_file_handler = logging.FileHandler(f"images/{self.unique_model_name}/logs/output.log")
        logger.addHandler(output_file_handler)

        # Consistent seeds for reproducibility
        self.mt_seed = mt_seed
        self.rng = np.random.default_rng(self.mt_seed)
        torch.manual_seed(self.mt_seed)

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
        logger.info(f"Is cuda enabled? {'YES' if self.cuda else 'NO'}")
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        # Initialize other training properties
        self.mt_id = mt_id
        self.epoch = 0
        self.n_epochs = opt.n_epochs
        self.batches_done = 0
        self.good_mut = 0
        self.mut_counter = 0
        self.g_loss = None
        self.d_loss = None
        self.sample_scaling = 2
        self.images_per_sample = 36
        self.rows_per_sample = 6
        self.number_of_final_show_off_samples = 5
        self.stats = {
            "time":   [],
            "batch":  [],
            "g_loss": [],
            "d_loss": [],
        }
        self._return_specific_epoch_state = False
        self._specific_epoch_state = 0
        self._model_suffix = "n"
        self.have_state_dicts = [self.generator, self.discriminator, self.optimizer_G, self.optimizer_D]
        self._enable_comparison_sequence = False

        # Generate seed for consistent images
        self.z_seed = Variable(self.Tensor(self.rng.normal(0, 1, (self.images_per_sample, opt.latent_dim))))
        self.final_show_off_z_seeds = [
            Variable(self.Tensor(self.rng.normal(0, 1, (self.images_per_sample, opt.latent_dim))))
            for _ in range(self.number_of_final_show_off_samples)
        ]

        # Create a description of the model in the log file
        self.save_model_training_info()

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
        specific_epoch_state = None
        while self.epoch < self.n_epochs:
            self.epoch += 1
            for i, (imgs, _) in enumerate(dataloader):
                self.train_epoch(imgs)
                self.batches_done = (self.epoch - 1) * len(dataloader) + i
                self.collect_stats()

                if self.batches_done % opt.sample_interval == 0:
                    self.save_image()
                    self.print_stats(len(dataloader))

            if self.epoch % opt.mut_interval == 0 and self.epoch != 0:
                if self._enable_comparison_sequence:
                    self.reset_seeds(self.epoch * self.mt_seed)
                if opt.enable_mut:
                    mutated_gen_state_dict = self.mutation_handler()
                    if mutated_gen_state_dict is not None:
                        self.apply_mutation(mutated_gen_state_dict)

            if self._return_specific_epoch_state and self.epoch == self._specific_epoch_state:
                specific_epoch_state = [deepcopy(m.state_dict()) for m in self.have_state_dicts]

            if self.SAVE_STATE_DICT_AFTER_EACH_EPOCH:
                self.save_model_state(self.have_state_dicts,
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
        logger.info("Starting mutation sequence:")
        best_of_the_best = []

        # Generate and choose the best mutation for each probability
        for mut_prob in self.mut_probabilities_list:
            logger.info(f"prob = {mut_prob:.5%}\t")
            em.create_mutations(prob=mut_prob)
            new_params = em.compare_mutations()
            if new_params is not None:
                logger.info("Adding the best mutation to the final comparison list")
                best_of_the_best.append(new_params)
            else:
                logger.info("Mutation unsuccessful, skipping.")

        # Compare the best mutations and choose 'the best of the best'
        if len(best_of_the_best) > 1:
            logger.info("Starting to compare best mutations:")
            best_params = em.compare_mutations(best_of_the_best)
            if best_params is not None:
                logger.info("Applying new params!")
                self.good_mut += 1
                return best_params
            else:
                logger.info("Mutation unsuccessful, keeping old params.")
        elif len(best_of_the_best) == 1:
            logger.info("Applying the only mutation which passed.")
            self.good_mut += 1
            return best_of_the_best[0]
        else:
            logger.info("Mutation unsuccessful, keeping old params.")
            return None

    def apply_mutation(self, new_state_dict):
        self.save_image("b")
        self.generator.load_state_dict(new_state_dict)
        self.save_image("a")

    def reset_seeds(self, seed=None):
        if seed is None:
            seed = self.mt_seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

    def save_image(self, prefix="", z_seed=None):
        if z_seed is None:
            z_seed = self.z_seed
        gen_imgs = self.generator(z_seed)
        scaled_imgs = transforms.Resize(size=gen_imgs.shape[0] * self.sample_scaling)(gen_imgs)
        save_image(scaled_imgs,
                   f"images/{self.unique_model_name}/samples/"
                   f"{prefix}{self.batches_done}_{self._model_suffix}.png",
                   nrow=self.rows_per_sample,
                   normalize=True)

    def print_stats(self, dataloader_length):
        logger.info(f"[Epoch {self.epoch:d}/{self.n_epochs}] "
                    f"[Batch {self.batches_done:d}/{self.n_epochs * dataloader_length:d}] "
                    f"[D loss: {self.d_loss.item():.3f}] "
                    f"[G loss: {self.g_loss.item():.3f}] "
                    f"[Mutations success rate {self.good_mut:d}/{self.mut_counter:d}]")

    def collect_stats(self):
        self.stats["time"].append(datetime.now())
        self.stats["batch"].append(self.batches_done)
        self.stats["d_loss"].append(self.d_loss.item())
        self.stats["g_loss"].append(self.g_loss.item())

    def plot_stats(self, smoothing=40, filename="plot"):
        plt.style.use("ggplot")
        stats_df = pd.DataFrame(self.stats)
        smoothed_stats_df = stats_df[["d_loss", "g_loss"]].rolling(smoothing, center=True).mean()

        fig, ax = plt.subplots()
        ax.plot(smoothed_stats_df.g_loss, label="Generator")
        ax.plot(smoothed_stats_df.d_loss, label="Discriminator")
        ax.legend()
        ax.set_title("Generator and discriminator loss plot")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Batch")

        plt.savefig(f"images/{self.unique_model_name}/media/{filename}.png", dpi=300)

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
        stats_df.to_csv(f"images/{self.unique_model_name}/logs/{filename}.csv", index=False)

    def save_model_training_info(self):
        mes = (f"Model iD:{self.mt_id}; start time:{datetime.now()}; "
               f"model seed:{self.mt_seed}; epochs:{self.n_epochs}; "
               f"batch size:{opt.batch_size}; comparison sequence:{self._enable_comparison_sequence};\n"
               f"mutations:{opt.enable_mut}; mutations interval:{opt.mut_interval}; "
               f"number of mutations:{opt.n_mut}; probabilities:{self.mut_probabilities_list};\n")
        logger.info(mes)

    def save_model_state(self, models, filenames):
        for model, filename in zip(models, filenames):
            torch.save(model.state_dict(), f"images/{self.unique_model_name}/models/{filename}.pth")

    def load_model_state(self, states):
        for m, state in zip(self.have_state_dicts, states):
            m.load_state_dict(state)

    def final_result_show_off(self, prefix=""):
        for i, fz_seed in enumerate(self.final_show_off_z_seeds):
            self.save_image(prefix=f"{prefix}final_zID{i+1}_", z_seed=fz_seed)

    def comparison_sequence(self, compare_mutations=True):
        if opt.enable_mut:  # Comparison won't make any sense if mutations are enabled
            raise RuntimeError("Comparison sequence can't be executed when 'enable_mut' parameter is True.")

        self._enable_comparison_sequence = True
        self._return_specific_epoch_state = True
        key_epochs = list(range(opt.mut_interval, self.n_epochs + 1, opt.mut_interval))
        logger.info("Comparison sequence is enabled now.")

        for epoch_num in key_epochs:
            logger.info(f"{10 * '-'} Starting a new sequence {10 * '-'}")
            self._specific_epoch_state = epoch_num
            self._model_suffix = str(epoch_num)

            # Train and save all stats
            states_from_i = self.trainer()
            self.plot_stats(filename=f"plot_{epoch_num}")
            self.save_stats(filename=f"stats_{epoch_num}")
            self.final_result_show_off()
            self.save_model_state(self.have_state_dicts,
                                  [f"gen_finish{epoch_num}", f"disc_finish{epoch_num}",
                                   f"optimG_finish{epoch_num}", f"optimD_finish{epoch_num}"])

            if compare_mutations:
                mutated_gen_state_dict = self.mutation_handler(old_gen_state_dict=states_from_i[0],
                                                               disc_state_dict=states_from_i[1])
                if mutated_gen_state_dict is not None:
                    states_from_i[0] = mutated_gen_state_dict

            # Prepare for the next training
            self.load_model_state(states_from_i)
            self.reset_seeds(epoch_num * self.mt_seed)
            self.epoch = epoch_num
            for stats_key in ["time", "batch", "g_loss", "d_loss"]:
                self.stats[stats_key] = []


if __name__ == "__main__":
    make_comparison = True
    mt = ModelTraining()
    if make_comparison:
        mt.comparison_sequence()
    else:
        mt.trainer()
        mt.plot_stats()
        mt.make_animation()
        mt.save_stats()
        mt.final_result_show_off()
    # dir_path = "images\\PyTorch_t1010_100858_important_version\\models\\"
    # x_cross_comparison(
    #     disc_v1_dict=torch.load(f"{dir_path}disc_finish4.pth"),
    #     gen_v1_dict=torch.load(f"{dir_path}gen_finish4.pth"),
    #     disc_v2_dict=torch.load(f"{dir_path}disc_finish8.pth"),
    #     gen_v2_dict=torch.load(f"{dir_path}gen_finish8.pth"),
    #     optimD_v1_dict=torch.load(f"{dir_path}optimD_finish4.pth"),
    #     optimG_v1_dict=torch.load(f"{dir_path}optimG_finish4.pth"),
    #     optimD_v2_dict=torch.load(f"{dir_path}optimD_finish8.pth"),
    #     optimG_v2_dict=torch.load(f"{dir_path}optimG_finish8.pth"),
    #     n_batches=1000,
    #     n_epochs=40,
    #     start_epoch=80,
    #     comparison_seed=23456
    # )
