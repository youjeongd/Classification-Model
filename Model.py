import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 2D CNN encoder using AlexNet pretrained
class AlexCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=512, in_channels=1):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(AlexCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(256 * 6 * 6, fc_hidden1),
            nn.BatchNorm1d(4096, momentum=0.01),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.BatchNorm1d(4096, momentum=0.01),
            nn.ReLU()
        )

        self.layer8 = nn.Linear(fc_hidden2, CNN_embed_dim)


    def forward(self, x_3d):
        cnn_embed_seq = []

        for t in range(x_3d.size(1)):
            # AlexNet CNN
            with torch.no_grad():
                x = x_3d[:, t, :, :, :]  # AlexNet

            # FC layers
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = x.view(x.size(0), -1)
            x = self.layer6(x)
            x = self.layer7(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.layer8(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=2, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.GRU = nn.GRU(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.GRU.flatten_parameters()
        RNN_out, (h_n, h_c) = self.GRU(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        return RNN_out[:, -1, :]
        '''
        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x
        '''


'''
데이터 라벨 분류 => Classification
'''
class PredictLabelModel(nn.Module):
    def __init__(self, num_classes=8, RNNdim=128, FCdim=256, drop_p=0.3):
        super(PredictLabelModel, self).__init__()
        self.drop_p = drop_p
        self.fc1 = nn.Linear(RNNdim, FCdim)
        self.fc2 = nn.Linear(FCdim, num_classes)

    def forward(self, RNN_out):
        x = self.fc1(RNN_out)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

'''
Shear 값 예측 => Regression
'''
class PredictSheerModel(nn.Module):
    def __init__(self, dim=2, RNNdim=128, FCdim=64, drop_p=0.3):
        super(PredictSheerModel, self).__init__()
        self.drop_p = drop_p
        self.fc1 = nn.Linear(RNNdim, FCdim)
        self.fc2 = nn.Linear(FCdim, dim)

    def forward(self, RNN_out):
        x = RNN_out
        x = self.fc1(x)
        x = self.fc2(x)

        return x

