import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class FC2_20_2Dense(nn.Module):

    def __init__(self, embed_dim, ff_dim1=20, ff_dim2=20, ff_dim3=20, spec_norm=False):
        super(FC2_20_2Dense, self).__init__()

        self.embed_dim = embed_dim
        self.spec_norm = spec_norm
        
        # Define layers and modules
        self.fc1 = self._add_spectral_norm(nn.Linear(embed_dim, ff_dim1))
        self.fc2 = self._add_spectral_norm(nn.Linear(ff_dim1, ff_dim2))

        self.bn1 = nn.BatchNorm1d(ff_dim1)
        self.bn2 = nn.BatchNorm1d(ff_dim2)
        self.bn3 = nn.BatchNorm1d(ff_dim1)
        self.bn4 = nn.BatchNorm1d(ff_dim2)
        self.bn5 = nn.BatchNorm1d(ff_dim3)

        # The same is done for input sequence 2.
        self.fc3 = self._add_spectral_norm(nn.Linear(embed_dim, ff_dim1))
        self.fc4 = self._add_spectral_norm(nn.Linear(ff_dim1, ff_dim2))

        # Both outputs are concatenated and fed to a fully connected layer with 20 neurons. Then, batch normalization is applied.
        self.fc5 = self._add_spectral_norm(nn.Linear(2*ff_dim2, ff_dim3))
        # The output of this layer is fed to a fully connected layer with 1 neuron.
        self.fc6 = self._add_spectral_norm(nn.Linear(ff_dim3, 1))

        # The model has 2 classes, 0 and 1
        self.classes = (0,1)

    def forward(self, x):
        test = x.split(1, dim=1)
        
        x1 = test[0]
        x1 = x1.to(torch.float32)
        x1 = x1.contiguous()
        x1 = x1.squeeze(1)
        x1 = x1.view(x1.size(0), -1)  # Reshape to (a, c*d)

        x2 = test[1]
        x2 = x2.to(torch.float32)
        x2 = x2.contiguous()
        x2 = x2.squeeze(1)
        x2 = x2.view(x2.size(0), -1)

        x1 = F.relu(self.fc1(x1))
        x1 = self.bn1(x1)
        x1 = F.relu(self.fc2(x1))
        x1 = self.bn2(x1)

        x2 = F.relu(self.fc3(x2))
        x2 = self.bn3(x2)
        x2 = F.relu(self.fc4(x2))
        x2 = self.bn4(x2)


        x = torch.cat((x1,x2), 1)
        x = F.relu(self.fc5(x))
        x = self.bn5(x) 
        x = self.fc6(x)
        x = x.view(x.size(0), -1)
        

        # classification is done using a sigmoid function
        x = torch.sigmoid(x)
        return x
    
    def get_classes(self):
        return self.classes
    
    def _add_spectral_norm(self, layer):
        if self.spec_norm:
            return spectral_norm(layer)
        else:
            return layer