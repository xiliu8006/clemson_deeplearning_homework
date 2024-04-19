import torch
import torchvision
from torchvision import utils
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from pytorch_gan_metrics import get_inception_score
from tqdm import tqdm
import numpy as np
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
batch_size = 128


class ACGAN_Generator(nn.Module):
    def __init__(self): 
        super(ACGAN_Generator, self).__init__()
        self.emb = nn.Embedding(10, 100)
        self.net_seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, True),
            # nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(num_features=64),
            # nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x, lables):
        x = torch.mul(self.emb(lables), x)
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return self.net_seq(x)

class ACGAN_Discriminator(nn.Module):
    def __init__(self):
        super(ACGAN_Discriminator, self).__init__()
        self.net_seq = nn.Sequential(
        # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
        # nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True))

        self.adv_layer = nn.Sequential(nn.Linear(512 * 4 ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(512 * 4 ** 2, 10), nn.Softmax())

    def forward(self, x):
        x = self.net_seq(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)
        label = self.aux_layer(x)
        return validity, label

def train(generator, discriminator, train_dataloader, learning_rate=0.0002, epochs=50):
    source_criterion = nn.BCELoss()
    class_criterion = nn.NLLLoss()
    optim_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if not os.path.exists('train_generated_images_acgan_real/'): 
        os.makedirs('train_generated_images_acgan_real')
    if not os.path.exists('train_generated_images_acgan_fake/'): 
        os.makedirs('train_generated_images_acgan_fake')
        
    inception_score_file = open("inception_score_acgan.csv", "w")
    inception_score_file.write('epoch, inception_score \n')

    for epoch in tqdm(range(epochs)): 
        for images, labels in train_dataloader:
            batch_size = images.shape[0]
            real_images = Variable(images.type(torch.cuda.FloatTensor)).to(device)
            real_labels = Variable(labels.type(torch.cuda.LongTensor)).to(device)

            fake = torch.zeros(batch_size).to(device)
            valid = torch.ones(batch_size).to(device)

            optim_generator.zero_grad()
            z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, 100))))
            generated_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, 10, batch_size)))

            generated_images = generator(z, generated_labels)

            validity, predicted_label = discriminator(generated_images)
            gen_loss = 0.5 * (source_criterion(validity, valid.unsqueeze(1)) + class_criterion(predicted_label, generated_labels))
            gen_loss.backward()
            optim_generator.step()

            optim_discriminator.zero_grad()

            # compute real images loss
            real_pred, real_aux = discriminator(real_images)
            disc_loss_real = 0.5 * (source_criterion(real_pred, valid.unsqueeze(1)) + class_criterion(real_aux, real_labels))

            # compute fake images loss
            fake_pred, fake_aux = discriminator(generated_images.detach())
            disc_loss_fake = 0.5 * (source_criterion(fake_pred, fake.unsqueeze(1)) + class_criterion(fake_aux, generated_labels))

            # compute overall discriminator loss, optimize discriminator
            disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
            disc_loss.backward()
            optim_discriminator.step()


        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (1000, 100))))
        generated_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, 10, 1000)))
        samples = generator(z, generated_labels)

        # normalize to [0, 1]
        samples = samples.mul(0.5).add(0.5)
        
        assert 0 <= samples.min() and samples.max() <= 1
        inception_score, inception_score_std = get_inception_score(samples)
        print("epoch: " + str(epoch) + ', inception score: ' + str(round(inception_score, 2)) + ' ± ' + str(round(inception_score_std, 2)))

        samples = samples.data.cpu()
        utils.save_image(samples, 'train_generated_images_acgan_fake/epoch_{}.png'.format(str(epoch)))
        utils.save_image(real_images, 'train_generated_images_acgan_real/epoch_{}.png'.format(str(epoch)))
        
        inception_score_file.write(str(epoch) + ', ' + str(round(inception_score, 2)) + '\n')

    inception_score_file.close()

def generate_images(generator, data_loader, batch_size=128):
    inception_scores = []
    samples_list = []
    for images, labels in data_loader:
        z = torch.randn(batch_size, 100).to(device)
        samples = generator(z, labels.to(device))
        samples = samples.mul(0.5).add(0.5)
        inception_score, inception_score_std = get_inception_score(samples)
        print(inception_score)
        inception_scores.append(inception_score)
        samples_list.append(samples)

    top_scores, top_indices = torch.topk(torch.tensor(inception_scores), k=10)
    samples = samples_list[top_indices[0]]
    random_indices = random.sample(range(samples.size(0)), 10)
    samples = samples[random_indices].data.cpu()
    grid = utils.make_grid(samples, nrow=5)
    print(top_scores)
    utils.save_image(grid, 'ACGAN_generated_images_top1.png')

def load_model(model, model_filename): 
    model.load_state_dict(torch.load(model_filename))

if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Resize(32), 
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_CIFAR10_set = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True, transform=transform)
    test_CIFAR10_set = torchvision.datasets.CIFAR10(root='./cifar10/', train=False, download=True, transform=transform)

    train_CIFAR10_dataloader = DataLoader(train_CIFAR10_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_CIFAR10_dataloader = DataLoader(test_CIFAR10_set, batch_size=batch_size, shuffle=True, drop_last=True)

    ACGAN_generator = ACGAN_Generator()
    ACGAN_discriminator = ACGAN_Discriminator()
    ACGAN_generator.to(device)
    ACGAN_discriminator.to(device)
    train(ACGAN_generator, ACGAN_discriminator, train_CIFAR10_dataloader)
    torch.save(ACGAN_generator.state_dict(), 'ACGAN_generator.pkl')
    torch.save(ACGAN_discriminator.state_dict(), 'ACGAN_discriminator.pkl')
    # load_model(ACGAN_generator, 'ACGAN_generator.pkl')
    # load_model(ACGAN_discriminator, 'ACGAN_discriminator.pkl')
    # generate_images(ACGAN_generator, test_CIFAR10_dataloader)