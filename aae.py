import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
N = 1000
n_classes = 10
z_dim = 2
X_dim = 28*28
y_dim = 10
batch_size = 16
test_batch_size=1000


class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3gauss = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss


class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, X_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return torch.sigmoid(x)


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

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)


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

progress = tqdm(enumerate(train_loader))
total_loss = 0
for i, (X, target) in progress: 
    X = X.cuda()
    X = X.view(batch_size, X_dim)

    #init gradients
    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()

    
    z_sample = Q(X)
    X_sample = P(z_sample)
    recon_loss = F.binary_cross_entropy(X_sample + TINY, X + TINY)
    recon_loss.backward()
    P_decoder.step()
    Q_encoder.step()
    
    Q.eval()    
    z_real_gauss = Variable(torch.randn(batch_size, z_dim) * 5)   # Sample from N(0,5)
    if torch.cuda.is_available():
        z_real_gauss = z_real_gauss.cuda()
    z_fake_gauss = Q(X)
    
    # Compute discriminator outputs and loss
    D_real_gauss, D_fake_gauss = D_gauss(z_real_gauss), D_gauss(z_fake_gauss)
    D_loss_gauss = -torch.mean(torch.log(D_real_gauss + TINY) + torch.log(1 - D_fake_gauss + TINY))
    D_loss_gauss.backward()       # Backpropagate loss
    D_gauss_solver.step()   # Apply optimization step
    
    total_loss += D_loss_gauss.item()
    # Generator
    Q.train()   # Back to use dropout
    z_fake_gauss = Q(X)
    D_fake_gauss = D_gauss(z_fake_gauss)
    
    G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
    G_loss.backward()
    Q_generator.step()
    progress.set_description("{}".format(total_loss/(i+1)))





