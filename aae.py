import cv2
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
N = 1000
n_classes = 62
z_dim = 2
X_dim = 128*128
batch_size = 256


class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            # nn.BatchNorm2d(128),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(128),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(256),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
        )
        self._projection = nn.Linear(16 * 16 * 128, z_dim)

    def forward(self, x):
        features = self._net(x)
        xgauss = self._projection(features.view((-1, 16 * 16 * 128)))
        return xgauss


class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self._net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            # nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            # nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            # nn.Sigmoid()
        )
        self._projection = nn.Linear(z_dim + n_classes, 16 * 16 * 128)
    
    
    def forward(self, z, label):
        z = torch.cat((z, label), dim=1)
        projection = self._projection(z)
        reconstruction = self._net(projection.view((-1, 128, 16, 16)))
        return reconstruction


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))


# read in data
class FontDataset(Dataset):
    def __init__(self, path='../dataset'):
        super(FontDataset, self).__init__()
        self._characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890'
        self._eye = np.eye(len(self._characters), dtype=np.float32)
        self._data = []
        self._trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        print("Loading Data")

        for directory in tqdm(os.listdir(path)):
        # for k, directory in tqdm(enumerate(os.listdir(path))):
        #     if k == 128:
        #         break
            font_path = os.path.join(path, directory)
            images = []

            for i, char in enumerate(self._characters):
                img_path = os.path.join(font_path, char + '.png')
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img = np.min(img, axis=2, keepdims=True)
                    img = self._trsfm(img)
                    images.append((img, self._eye[i]))

            if len(images) > 0:
                self._data.append(images)


    def __getitem__(self, idx):
        font = self._data[idx]
        char1 = random.choice(font)
        char2 = random.choice(font)
        return char1[0], char2[0], char2[1]  # input image, recration image, recreation class


    def __len__(self):
        return len(self._data)


font_dataset = FontDataset()
train_loader = DataLoader(
        font_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=test_batch_size, shuffle=True, **kwargs)


Q, P = Q_net(), P_net()     # Encoder/Decoder
D_gauss = D_net_gauss()                # Discriminator adversarial
if torch.cuda.is_available():
    Q = Q.cuda()
    P = P.cuda()
    D_cat = D_gauss.cuda()
    D_gauss = D_net_gauss().cuda()

# Set learning rates
gen_lr, reg_lr = 0.0006, 0.0008
# Set optimizators
P_decoder = optim.Adam(P.parameters(), lr=gen_lr)
Q_encoder = optim.Adam(Q.parameters(), lr=gen_lr)
Q_generator = optim.Adam(Q.parameters(), lr=reg_lr)
D_gauss_solver = optim.Adam(D_gauss.parameters(), lr=reg_lr)

TINY = 1e-15

for epoch in range(60):
    total_loss = 0
    progress = tqdm(enumerate(train_loader))
    for i, (input_img, target, target_class) in progress: 
        input_img = input_img.cuda()
        target = target.cuda()
        target_class = target_class.cuda()

        #init gradients
        P.zero_grad()
        Q.zero_grad()
        D_gauss.zero_grad()
        
        z_sample = Q(input_img)
        reconstruction = P(z_sample, target_class)
        recon_loss = F.binary_cross_entropy(reconstruction + TINY, input_img + TINY)
        recon_loss.backward()
        print(recon_loss.item())
        P_decoder.step()
        Q_encoder.step()
        
        Q.eval()    
        z_real_gauss = Variable(torch.randn(batch_size, z_dim) * 5)   # Sample from N(0,5)
        if torch.cuda.is_available():
            z_real_gauss = z_real_gauss.cuda()
        z_fake_gauss = Q(input_img)
        
        # Compute discriminator outputs and loss
        D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)
        D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
        D_loss_gauss.backward()       # Backpropagate loss
        D_gauss_solver.step()   # Apply optimization step
        
        total_loss += D_loss_gauss.item()
        # Generator
        Q.train()   # Back to use dropout
        z_fake_gauss = Q(input_img)
        D_fake_gauss = D_gauss(z_fake_gauss)
        
        G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
        G_loss.backward()
        Q_generator.step()

        progress.set_description("E: {} L: {}".format(epoch, total_loss/(i+1)))

    for i in range(1):
        fig, ax = plt.subplots(ncols=13, nrows=4, squeeze=False)
        z_real_gauss = Variable(torch.stack([torch.randn(z_dim) * 5] * 52)).cuda()   # Sample from N(0,5)
        labels = torch.from_numpy(np.eye(62)[:52, :]).float().cuda()
        sampled_x = P(z_real_gauss, labels).cpu().detach()
        for idx_i in range(4):
            for idx_j in range(13):
                ax[idx_i][idx_j].imshow(sampled_x[idx_i * 13 + 4][0], cmap='gray')
                ax[idx_i][idx_j].axis('off')
        plt.savefig("{}reconstruction{}.png".format(epoch, i))
        plt.close()

with open("encoder.pkl", "wb") as of:
    pickle.dump(P, of)
with open("decoder.pkl", "wb") as of:
    pickle.dump(Q, of)
with open("critic.pkl", "wb") as of:
    pickle.dump(D_gauss, of)



