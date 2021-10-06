import argparse
import os
import numpy as np
from collections import OrderedDict

import torchvision.transforms as transforms
from torchvision.utils import save_image

import glob
import imageio
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

from bit_operations import BitOps

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400,
                    help="interval between image samples")

parser.add_argument("--enable_mutations", type=bool, default=False,
                    help="whether to generate mutations")
parser.add_argument("--n_mutations", type=int, default=20,
                    help="number of the mutations per parameter")
parser.add_argument("--mutation_prob", type=float, default=0.02,
                    help="probability of the mutation")
parser.add_argument("--mutation_interval", type=int, default=6000,
                    help="interval between mutations")
opt = parser.parse_args()
print(opt)


def make_animation(folder_with_imgs, output_path):
    with imageio.get_writer(output_path, mode='I') as writer:
        filenames = glob.glob(f"{folder_with_imgs}/*.png")
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


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
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z_):
        img = self.model(z_)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
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
    def __init__(self, gen_state_dict, disc_state_dict, n_mut=5):
        self.gen_state_dict = self.params2device(gen_state_dict,
                                                 out_device="cpu")
        self.disc_state_dict = self.params2device(disc_state_dict,
                                                  out_device="cpu")
        self.n_mut = n_mut
        self.mutated_dicts = [OrderedDict() for _ in range(n_mut)]

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
                self.mutated_dicts[mut_ind][key] = torch.from_numpy(
                    mut_engine.mutations[mut_ind].reshape(param.shape)
                )
            del mut_engine
        return self.mutated_dicts

    def compare_mutations(self, mutated_dicts=None, test_batch=128):
        """ Compares mutations and returns either the best one or None.

        Parameters
        ----------
        test_batch : int
            The size of the batch on which each mutation is tested

        Returns
        -------
        OrderedDict or None
            None if all mutations appeared to be worse then the original
            parameters and OrderedDict if there was at least one successful
            mutation.
        """
        gen_eval = Generator()
        disc_eval = Discriminator()
        adversarial_loss_eval = torch.nn.BCELoss()
        tensor_eval = torch.FloatTensor

        disc_eval.load_state_dict(self.disc_state_dict)
        valid_eval = Variable(tensor_eval(test_batch, 1).fill_(1.0),
                              requires_grad=False)
        z_eval = Variable(tensor_eval(np.random.normal(
            0, 1, (test_batch, opt.latent_dim))))

        if mutated_dicts is None:
            mutated_dicts = self.mutated_dicts
        mut_res = np.empty(len(mutated_dicts))
        for mut_ind in range(len(mutated_dicts)):
            gen_eval.load_state_dict(mutated_dicts[mut_ind])
            with torch.no_grad():
                gen_imgs_eval = gen_eval(z_eval)
                loss_eval = adversarial_loss_eval(disc_eval(gen_imgs_eval),
                                                  valid_eval)
            mut_res[mut_ind] = loss_eval
        gen_eval.load_state_dict(self.gen_state_dict)
        with torch.no_grad():
            curr_imgs = gen_eval(z_eval)
            curr_loss = adversarial_loss_eval(disc_eval(curr_imgs),
                                              valid_eval)
        min_ind = np.argmin(mut_res)
        to_print = np.copy(mut_res)
        to_print.sort()
        print(f"\nCurrent loss {curr_loss.numpy():.4f}\nBest mutations:",
              to_print[:6])
        if mut_res[min_ind] < curr_loss.numpy():
            return self.params2device(mutated_dicts[int(min_ind)],
                                      out_device="cuda:0" if cuda else "cpu")
        return None

    @staticmethod
    def params2device(in_params, out_device="cuda:0") -> OrderedDict:
        """ Converts state dictionary from CPU to GPU and vice versa.

        Parameters
        ----------
        in_params : OrderedDict[str, torch.Tensor]
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

    def __init__(self):
        # TODO move all opt parameters as a class attributes
        img_shape = (opt.channels, opt.img_size, opt.img_size)
        cuda = True if torch.cuda.is_available() else False
        print("Is cuda enabled?", "YES" if cuda else "NO")
        model_name_suffix = (f"_m{opt.enable_mutations}"
                             f"_ep{opt.n_epochs}_bs{opt.batch_size}"
                             f"_nm{opt.n_mutations}_mp{opt.mutation_prob}"
                             f"_mi{opt.mutation_interval}")
        UNIQUE_MODEL_NAME = (f"PyTorch_t{datetime.now().strftime('%m%d_%H%M%S')}"
                             f"{model_name_suffix}")
        mut_probabilities_list = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
        os.makedirs(f"images/{UNIQUE_MODEL_NAME}/", exist_ok=True)

        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        generator = Generator()
        discriminator = Discriminator()
        generator.apply(initialize_weights_for_tanh)
        discriminator.apply(initialize_weights_for_sigmoid)

        if cuda:
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        # Configure data loader
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "..",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(),
                                       lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                       lr=opt.lr, betas=(opt.b1, opt.b2))

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        seed = Variable(
            Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

    def train_epoch(self):

# ----------
#  Training
# ----------
mut_counter, good_mut = 0, 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0),
                         requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(
            0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability
        # to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.mutation_interval == 0 and batches_done != 0 and opt.enable_mutations:
            for mut_rep_ind in range(10):
                print(f"\n\n-------Mut rep ind = {mut_rep_ind}-------\n")
                # print(f"Calculating mutations with "
                #       f"p = {opt.mutation_prob / (epoch + 1):.5%}")
                mut_counter += 1
                em = Mutator(generator.state_dict(),
                             discriminator.state_dict(),
                             n_mut=opt.n_mutations)
                print("Starting mutation sequence:")
                best_of_the_best = []
                for mut_prob in mut_probabilities_list:
                    print(f"prob = {mut_prob:.5%}", end="\t")
                    # TODO: revert changes for the final model
                    em.create_mutations(prob=mut_prob)
                    new_params = em.compare_mutations()
                    if new_params is not None:
                        print("Adding the best mutation to the final comparison list")
                        best_of_the_best.append(new_params)
                    else:
                        print("Mutation unsuccessful, skipping.")

                if len(best_of_the_best) > 1:
                    print("Starting to compare best mutations:")
                    best_params = em.compare_mutations(best_of_the_best)
                    if best_params is not None:
                        gen_imgs = generator(seed)
                        save_image(gen_imgs.data[:36],
                                   f"images/{UNIQUE_MODEL_NAME}/m{batches_done + mut_rep_ind}.png",
                                   nrow=6, normalize=True)
                        good_mut += 1
                        print("Applying new params!")
                        generator.load_state_dict(best_params)

                        # make images before and after mutation
                        z = Variable(Tensor(np.random.normal(
                            0, 1, (imgs.shape[0], opt.latent_dim))))
                        gen_imgs = generator(seed)
                        save_image(gen_imgs.data[:36],
                                   f"images/{UNIQUE_MODEL_NAME}/m{batches_done + 1 + mut_rep_ind}.png",
                                   nrow=6, normalize=True)
                    else:
                        print("Mutation unsuccessful, keeping old params.")
                elif len(best_of_the_best) == 1:
                    print("Applying the only mutation which passed.")
                    gen_imgs = generator(seed)
                    save_image(gen_imgs.data[:36],
                               f"images/{UNIQUE_MODEL_NAME}/m{batches_done + mut_rep_ind}.png",
                               nrow=6, normalize=True)
                    good_mut += 1
                    generator.load_state_dict(best_of_the_best[0])

                    # make images before and after mutation
                    z = Variable(Tensor(np.random.normal(
                        0, 1, (imgs.shape[0], opt.latent_dim))))
                    gen_imgs = generator(seed)
                    save_image(gen_imgs.data[:36],
                               f"images/{UNIQUE_MODEL_NAME}/m{batches_done + 1 + mut_rep_ind}.png",
                               nrow=6, normalize=True)
                else:
                    print("Mutation unsuccessful, keeping old params.")

                # em.create_mutations(prob=opt.mutation_prob / (epoch + 1))
                # new_params = em.compare_mutations()
                # if new_params is not None:
                #     good_mut += 1
                #     print("Applying new params!")
                #     generator.load_state_dict(new_params)
                # else:
                #     print("Mutation unsuccessful, keeping old params.")

        if batches_done % opt.sample_interval == 0:
            gen_imgs = generator(seed)
            save_image(gen_imgs.data[:36],
                       f"images/{UNIQUE_MODEL_NAME}/{batches_done}.png",
                       nrow=6, normalize=True)
            print(f"[Epoch {epoch:d}/{opt.n_epochs}] "
                  f"[Batch {i:d}/{len(dataloader):d}] "
                  f"[D loss: {d_loss.item():.3f}] "
                  f"[G loss: {g_loss.item():.3f}] "
                  f"[Mutations success rate {good_mut:d}/{mut_counter:d}]")

make_animation(f"images/{UNIQUE_MODEL_NAME}/",
               f"images/{UNIQUE_MODEL_NAME}/animation.gif")
