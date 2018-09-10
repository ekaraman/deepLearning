import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from plyfile import PlyData, PlyElement
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import pandas as pd
import torch
import torch.utils.data
import time



class dataset2D(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.points = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.filelist = [f for f in os.listdir(self.root_dir) if f.endswith("ply")]
        self.data_dict = {}
        for j in range(len(self.points)):
            self.data_dict[self.points.iloc[j, 0]] = self.points.iloc[j, 1:].as_matrix()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.filelist[idx])
        plydata = PlyData.read(img_name)
        vertices = plydata.elements[0].data[:]
        object3d=np.zeros([vertices.shape[0],3])
        for i in range(vertices.shape[0]):
            for j in range(3):
                object3d[i,j]=vertices[i][j]

        #image = io.imread(img_name)
        points2d = self.data_dict[self.filelist[idx]]
        points2d = points2d.astype('float').reshape(-1, 2)
        sketch_order = np.array([3, 2, 4, 6, 4, 2, 8, 10, 8, 2, 1, 7, 5, 7, 1, 11, 9]) - 1
        for i in range(len(sketch_order)-1):
            a = np.array([points2d[sketch_order[i+1]]])
            b = np.array([points2d[sketch_order[i]]])
            if a[0, 0] == b[0, 0]:
                a1 = np.array([np.ones(20) * a[0, 0]]).reshape(-1, 1)
            else:
                a1 = np.arange(a[0, 0], b[0, 0], (b[0, 0] - a[0, 0]) * 0.05).reshape(-1, 1)
            if a[0, 1] == b[0, 1]:
                b1 = np.array([np.ones(20) * a[0, 1]]).reshape(-1, 1)
            else:
                b1 = np.arange(a[0, 1], b[0, 1], (b[0, 1] - a[0, 1]) * 0.05).reshape(-1, 1)
            c = np.concatenate((a1, b1), 1)
            points2d = np.concatenate((points2d, c), 0)

        relative = np.copy(points2d[:len(points2d) - 1])
        for j in range(len(relative)):
            points2d[j + 1] = points2d[j + 1] - relative[j]
        points2d[0] -= points2d[0]

        sample = {'object3d': object3d, 'points2d': points2d}

        if self.transform:
            sample = self.transform(sample)

        return sample

dataset = dataset2D(csv_file='./rotated_files2/mesh000/mesh000_selected_points.csv',
                                    root_dir='./rotated_files2/mesh000')

#print dataset[0]['object3d']
#print dataset[0]['points2d']

class ToTensor2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        object3d, points2d = sample['object3d']/2, sample['points2d']/2
        return {'object3d': torch.from_numpy(object3d).float(),
                'points2d': torch.from_numpy(points2d).float()}

# class ConcatDataset(Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets
#
#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)
#
#     def __len__(self):
#         return min(len(d) for d in self.datasets)

dataset = dataset2D(csv_file='./rotated_files2/mesh000/mesh000_selected_points.csv',
                                    root_dir='./rotated_files2/mesh000',
                                           transform=transforms.Compose([
                                               ToTensor2()
                                           ]))
for i in range(1,51) + range(52, 72):
    if i<10:
        dataset1 = dataset2D(csv_file='./rotated_files2/mesh00'+str(i)+'/mesh00'+str(i)+'_selected_points.csv',
                                    root_dir='./rotated_files2/mesh00'+str(i),
                                           transform=transforms.Compose([
                                               ToTensor2()
                                           ]))

    else:
        dataset1 = dataset2D(csv_file='./rotated_files2/mesh0' + str(i) + '/mesh0' + str(i) + '_selected_points.csv',
                             root_dir='./rotated_files2/mesh0' + str(i),
                             transform=transforms.Compose([
                                 ToTensor2()
                             ]))

    dataset = torch.utils.data.ConcatDataset((dataset, dataset1))


#dataloader2 = DataLoader(dataset0, batch_size=40, shuffle=True)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)



def reparameterize(mu, logvar, training):
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)/100000
        return eps.mul(std).add_(mu)
    else:
        return mu


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(12500*3, 900)
        self.fc2 = nn.Linear(900, 500)
        self.fc31 = nn.Linear(500, 256)
        self.fc32 = nn.Linear(500, 256)
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(256, 500)
        self.fc2 = nn.Linear(500, 900)
        self.fc3 = nn.Linear(900, 12500*3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Encoder2d(nn.Module):
    def __init__(self):
        super(Encoder2d, self).__init__()
        self.fc1 = nn.Linear(331 * 2, 500)
        self.fc2 = nn.Linear(500, 400)
        self.fc31 = nn.Linear(400, 256)
        self.fc32 = nn.Linear(400, 256)

        # nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    BCE = torch.sqrt(torch.mean((recon_x - x) ** 2))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())*0.00001
    #print BCE
    ##print KLD

    return BCE + KLD
def loss_kl(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())*0.00001
    #print BCE
    ##print KLD

    return KLD


encoder = Encoder().cuda()
decoder = Decoder().cuda()
encoder2d = Encoder2d().cuda()

learning_rate = 1e-3
num_epochs1 = 0
num_epochs2 = 0
num_epochs3 = 800

outpath='outputVAE_rall1/'

parameters1 = list(encoder.parameters()) + list(decoder.parameters())
criterion1 = nn.MSELoss()
optimizer1 = torch.optim.Adam(parameters1, lr=learning_rate, weight_decay=1e-5)

#model = autoencoder().cuda()
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(
#    model.parameters(), lr=learning_rate, weight_decay=1e-5)
B=16
k=0
tt=0
###episode1
print time.clock(), "first"
print len(train_loader)
for epoch in range(num_epochs1):
    for data in train_loader:
        img = data['object3d']
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()

        # ===================forward=====================
        mu, logvar = encoder(img)
        z = reparameterize(mu, logvar, 1)
        output = decoder(z)
        loss = loss_function(output, img, mu, logvar)
        # ===================backward====================
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        if epoch == num_epochs1-1 and tt<100:
            for i in range(len(data)):
                pose = output.cpu().data
                sample = pose.chunk(B,0)
                sample = sample[i].view(12500, 3)
                filename = outpath + 'ep1_model_' + str(tt) + '.xyz'
                np.savetxt(filename, sample, delimiter=' ')


                pose = img.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                sampleml=np.copy(sample)
                filename = outpath + 'ep1_inp_model_' + str(tt) + '.xyz'
                np.savetxt(filename, sampleml, delimiter=' ')

                tt=tt+1
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs1, loss.data[0]))

#parameters2 = list(encoder.parameters()) + list(encoder2d.parameters())
parameters2 = list(encoder2d.parameters())
criterion2 = nn.MSELoss()
optimizer2 = torch.optim.Adam(parameters2, lr=learning_rate, weight_decay=1e-5)

###episode2
tt=0
for epoch in range(num_epochs2):
    for data in train_loader:
        img = data['object3d']
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        #print img.requires_grad

        joints = data['points2d']
        joints = joints.view(joints.size(0), -1)
        joints = Variable(joints).cuda()
        # ===================forward=====================
        mu1, logvar1 = encoder(img)
        z1 = reparameterize(mu1, logvar1, 1)

        mu2, logvar2 = encoder2d(joints)
        z2 = reparameterize(mu2, logvar2, 1)

        outputfrom3d = decoder(z1)
        outputfrom2d = decoder(z2)
        loss = loss_kl(mu1, logvar1) + loss_kl(mu2, logvar2)
        loss += criterion1(mu2, Variable(mu1.data, requires_grad=False)) + criterion1(logvar2, Variable(logvar1.data, requires_grad=False))

        #loss = criterion2(output2, output1)
        # ===================backward====================
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
        if epoch == num_epochs2-1 and tt<100:
            for i in range(len(data['object3d'])):
                pose = outputfrom3d.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                filename = outpath + 'ep2_model_' + str(tt) + '.xyz'
                np.savetxt(filename, sample, delimiter=' ')


                pose = outputfrom2d.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                filename = outpath + 'ep2_model_2d_' + str(tt) + '.xyz'
                np.savetxt(filename, sample, delimiter=' ')


                pose = img.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                sampleml=np.copy(sample)
                filename = outpath + 'ep2_model_ml_' + str(tt) + '.xyz'
                np.savetxt(filename, sampleml, delimiter=' ')

                tt=tt+1

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs2, loss.data[0]))

learning_rate = 1e-3


parameters3 = list(encoder.parameters()) + list(encoder2d.parameters()) + list(decoder.parameters())
optimizer3 = torch.optim.Adam(parameters3, lr=learning_rate, weight_decay=1e-5)
tt=0

for epoch in range(num_epochs3):
    for data in train_loader:
        img = data['object3d']
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()

        joints = data['points2d']
        joints = joints.view(joints.size(0), -1)
        joints = Variable(joints).cuda()

        # ===================forward=====================
        mu1, logvar1 = encoder(img)
        z1 = reparameterize(mu1, logvar1, 1)

        mu2, logvar2 = encoder2d(joints)
        z2 = reparameterize(mu2, logvar2, 1)

        output = decoder(z1)
        outputfrom2d = decoder(z2)

        loss = loss_function(output, img, mu1, logvar1) + loss_function(outputfrom2d, img, mu2, logvar2)
        loss += criterion1(mu1, Variable(mu2.data, requires_grad=False)) + criterion1(mu2, Variable(mu1.data, requires_grad=False))
        loss += criterion1(logvar1, Variable(logvar2.data, requires_grad=False))+criterion1(logvar2, Variable(logvar1.data, requires_grad=False))
        #loss = loss_function(img, output, output1, output2)
        #loss = criterion1(output, img)
        # ===================backward====================
        optimizer3.zero_grad()
        loss.backward()
        optimizer3.step()

        if epoch == num_epochs3-1 and tt<50:
            for i in range(len(data['object3d'])):
                pose = output.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                filename = outpath + 'ep3_model_' + str(tt) + '.xyz'
                np.savetxt(filename, sample, delimiter=' ')


                pose = outputfrom2d.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                filename = outpath + 'ep3_model2d_' + str(tt) + '.xyz'
                np.savetxt(filename, sample, delimiter=' ')

                pose = img.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                sampleml = np.copy(sample)
                filename = outpath + 'ep3_model_org_' + str(tt) + '.xyz'
                np.savetxt(filename, sampleml, delimiter=' ')


                pose = img.cpu().data
                sample = pose.chunk(B, 0)
                sample = sample[i].view(12500, 3)
                sampleml=np.copy(sample)
                sampleml[:, 2] = sample[:, 1]
                sampleml[:, 1] = sample[:, 2]
                filename = outpath + 'ep3_orgsketch_' + str(tt) + '.xyz'
                np.savetxt(filename, sampleml, delimiter=' ')

                tt += 1







    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs3, loss.data[0]))
    del loss
    del output
    del outputfrom2d

    # if epoch % 100==0:
    #     pose = img.cpu().data
    #     sample=pose.view(12500, 3)
    #     fig = plt.figure()
    #     #ax = fig.add_subplot(111, projection='3d')
    #     #ax.scatter(sample[:, 2], sample[:, 1], sample[:, 0])
    #     plt.scatter(sample[:, 0], sample[:, 2])
    #     fig.savefig('org3d' + str(k) + '.png')
    #     k=k+1
    #     #plt.show()

torch.save(encoder.state_dict(), './VAE_Rall1_enc.pth')
torch.save(encoder2d.state_dict(), './VAE_Rall1_enc2d.pth')
torch.save(decoder.state_dict(), './VAE_Rall1_dec.pth')

