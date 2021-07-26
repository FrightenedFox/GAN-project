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
parser.add_argument("--batch_size", type=int, default=128,
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

parser.add_argument("--enable_mutations", type=bool, default=True,
                    help="whether to generate mutations")
parser.add_argument("--n_mutations", type=int, default=25,
                    help="number of the mutations per parameter")
parser.add_argument("--mutation_prob", type=float, default=0.02,
                    help="probability of the mutation")
parser.add_argument("--mutation_interval", type=int, default=12000,
                    help="interval between mutations")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print("Is cuda enabled?", "YES" if cuda else "NO")

model_name_suffix = (f"_m{opt.enable_mutations}"
                     f"_ep{opt.n_epochs}_bs{opt.batch_size}"
                     f"_nm{opt.n_mutations}_mp{opt.mutation_prob}"
                     f"_mi{opt.mutation_interval}")
UNIQUE_MODEL_NAME = (f"PyTorch_t{datetime.now().strftime('%m%d_%H%M%S')}"
                     f"{model_name_suffix}")

os.makedirs(f"images/{UNIQUE_MODEL_NAME}/", exist_ok=True)


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


class EvoMod:
    def __init__(self, gen_state_dict, disc_state_dict, n_mut=5):
        self.gen_state_dict = self.params2device(gen_state_dict,
                                                 out_device="cpu")
        self.disc_state_dict = self.params2device(disc_state_dict,
                                                  out_device="cpu")
        self.n_mut = n_mut
        self.mutated_dicts = [OrderedDict() for _ in range(n_mut)]
        self.mut_res = np.empty(n_mut)

    def create_mutations(self, **kwargs):
        for key, param in self.gen_state_dict.items():
            mut_engine = BitOps(param.numpy().flatten())
            mut_engine.mutate(n_mut=self.n_mut, **kwargs)
            for mut_ind in range(self.n_mut):
                self.mutated_dicts[mut_ind][key] = torch.from_numpy(
                    mut_engine.mutations[mut_ind].reshape(param.shape)
                )
            del mut_engine
        return self.mutated_dicts

    def compare_mutations(self, n_tests=10):
        gen_eval = Generator()
        disc_eval = Discriminator()
        adversarial_loss_eval = torch.nn.BCELoss()
        tensor_eval = torch.FloatTensor

        disc_eval.load_state_dict(self.disc_state_dict)
        valid_eval = Variable(tensor_eval(n_tests, 1).fill_(1.0),
                              requires_grad=False)
        z_eval = Variable(tensor_eval(np.random.normal(
            0, 1, (n_tests, opt.latent_dim)
        )))
        for mut_ind in range(self.n_mut):
            gen_eval.load_state_dict(self.mutated_dicts[mut_ind])
            with torch.no_grad():
                gen_imgs_eval = gen_eval(z_eval)
                loss_eval = adversarial_loss_eval(disc_eval(gen_imgs_eval),
                                                  valid_eval)
            self.mut_res[mut_ind] = loss_eval
        gen_eval.load_state_dict(self.gen_state_dict)
        with torch.no_grad():
            curr_imgs = gen_eval(z_eval)
            curr_loss = adversarial_loss_eval(disc_eval(curr_imgs),
                                              valid_eval)
        min_ind = np.argmin(self.mut_res)
        to_print = np.copy(self.mut_res)
        to_print.sort()
        print("\nCurrent loss %4f\nBest mutations:" % (curr_loss.numpy()),
              to_print[:6])
        if self.mut_res[min_ind] < curr_loss.numpy():
            return self.params2device(
                self.mutated_dicts[int(min_ind)],
                out_device="cuda:0" if cuda else "cpu"
            )
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


def make_animation(folder_with_imgs, output_path):
    with imageio.get_writer(output_path, mode='I') as writer:
        filenames = glob.glob(f"{folder_with_imgs}/*.png")
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

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
        if batches_done % opt.mutation_interval == 0 and opt.enable_mutations:
            mut_counter += 1
            print(f"Calculating mutations with "
                  f"p = {opt.mutation_prob / (epoch + 1):.4%}")
            em = EvoMod(generator.state_dict(),
                        discriminator.state_dict(),
                        n_mut=opt.n_mutations)
            em.create_mutations(prob=opt.mutation_prob / (epoch + 1))
            new_params = em.compare_mutations()
            if new_params is not None:
                good_mut += 1
                print("Applying new params!")
                generator.load_state_dict(new_params)
            else:
                print("Mutation unsuccessful, keeping old params.")

        if batches_done % opt.sample_interval == 0:
            gen_imgs = generator(seed)
            save_image(gen_imgs.data[:25],
                       f"images/{UNIQUE_MODEL_NAME}/{batches_done}.png",
                       nrow=5, normalize=True)
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(),
                   g_loss.item()),
                "[Mutations success rate %d/%d]" % (good_mut, mut_counter)
            )

make_animation(f"images/{UNIQUE_MODEL_NAME}/",
               f"images/{UNIQUE_MODEL_NAME}/animation.gif")

