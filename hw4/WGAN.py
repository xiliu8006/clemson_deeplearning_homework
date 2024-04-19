import torch
import torchvision
from torchvision import utils
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from pytorch_gan_metrics import get_inception_score
from tqdm import tqdm
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
batch_size = 128

class WGAN_Generator(nn.Module):
    def __init__(self): 
        super(WGAN_Generator, self).__init__()
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

    def forward(self, x):
        return self.net_seq(x)

class WGAN_Discriminator(nn.Module):
    def __init__(self):
        super(WGAN_Discriminator, self).__init__()
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
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        return self.net_seq(x)

def train(generator, discriminator, train_dataloader, learning_rate = 0.00005, epochs = 50):
    optim_generator = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
    optim_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)
    weight_cliping_limit = 0.01
        
    inception_score_file = open("inception_score_WGAN.csv", "w")
    inception_score_file.write('epoch, inception_score \n')
    one = torch.FloatTensor([1])
    mone = one * -1

    one = one.to(device)
    mone = mone.to(device)
    for epoch in tqdm(range(epochs)): 
        for real_images, _ in train_dataloader:
            real_images = real_images.to(device)
            
            z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
            preds = discriminator(real_images)
            fake_images = generator(z).detach()
            disc_loss = -torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_images))
            
            # optimize discriminator
            optim_discriminator.zero_grad()
            disc_loss.backward()
            optim_discriminator.step()

            for p in discriminator.parameters():
                p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

            z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
            fake_images = generator(z)
            preds = discriminator(fake_images)
            gen_loss = -torch.mean(preds)

            # optimize generator 
            optim_generator.zero_grad()
            gen_loss.backward()
            optim_generator.step()

        z = Variable(torch.randn(1000, 100, 1, 1)).to(device)
        samples = generator(z)

        samples = samples.mul(0.5).add(0.5)
        
        assert 0 <= samples.min() and samples.max() <= 1
        inception_score, inception_score_std = get_inception_score(samples)
        print("epoch: " + str(epoch) + ', inception score: ' + str(round(inception_score, 2)) + ' ± ' + str(round(inception_score_std, 2)))

        samples = samples[:64].data.cpu()
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'train_generated_images_WGAN/epoch_{}.png'.format(str(epoch)))
        inception_score_file.write(str(epoch) + ', ' + str(round(inception_score, 2)) + '\n')

    inception_score_file.close()

def generate_images(generator, batch_size=128, num_infer = 100):
    inception_scores = []
    samples_list = []
    for i in range(num_infer):
        z = torch.randn(batch_size, 100, 1, 1).to(device)
        samples = generator(z)
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
    utils.save_image(grid, 'wgan_generated_images_top1.png')

def load_model(model, model_filename): 
    model.load_state_dict(torch.load(model_filename))
    
if __name__ == "__main__":
    transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Resize(32), 
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_CIFAR10_set = torchvision.datasets.CIFAR10(root='./cifar10/', train=True, download=True, transform=transform)
    train_CIFAR10_dataloader = DataLoader(train_CIFAR10_set, batch_size=batch_size, shuffle=True, drop_last=True)

    WGAN_generator = WGAN_Generator()
    WGAN_discriminator = WGAN_Discriminator()
    WGAN_generator.to(device)
    WGAN_discriminator.to(device)
    # train(WGAN_generator, WGAN_discriminator, train_CIFAR10_dataloader)
    # torch.save(WGAN_generator.state_dict(), 'WGAN_generator.pkl')
    # torch.save(WGAN_discriminator.state_dict(), 'WGAN_discriminator.pkl')

    # load trained model and generate sample images
    # print("loading WGAN model...")
    load_model(WGAN_generator, 'WGAN_generator.pkl')
    load_model(WGAN_discriminator, 'WGAN_discriminator.pkl')
    generate_images(WGAN_generator)