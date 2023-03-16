import torch
import torch.nn as nn
from torchinfo import summary


class MyEncodeCNN(nn.Module):
    def __init__(self):
        super(MyEncodeCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 90 * 100 * 1
            # Out = 30 * 98 * 16
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=(1, 1), padding=0),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 30 * 98 * 16
            # Out = 26 * 94 * 32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=(1, 1), padding=0),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 26 * 94 * 32
            # Out = 22 * 90 * 64
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=(1, 1), padding=0),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 22 * 90 * 64
            # Out = 18 * 86 * 128
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=(1, 1), padding=0),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 18 * 86 * 128
            # Out = 14 * 82 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(14, 82), stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)

        # x = self.dropout(x)
        # x = self.fclayers(x)
        return x
        # return x.view(-1, 18 * 128, 86)


class MyEncodeLSTM(nn.Module):
    def __init__(self):
        super(MyEncodeLSTM, self).__init__()

        self.hidden_size = 128
        self.num_layers = 2

        self.layer = nn.LSTM(128 * 18 * 2, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)

    def forward(self, x):
        out, (final_hidden_state, final_cell_state) = self.layer(x)
        return out


class MyEncodeFC(nn.Module):
    def __init__(self):
        super(MyEncodeFC, self).__init__()

        self.fclayers = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fclayers(x)
        return x


class MyEncoder(nn.Module):
    def __init__(self):
        super(MyEncoder, self).__init__()
        self.cnn1 = MyEncodeCNN()
        self.cnn2 = MyEncodeCNN()
        # self.lstm = MyEncodeLSTM()
        self.fc = MyEncodeFC()

    def forward(self, x):
        x = torch.chunk(x.view(-1, 2, 90, 100), 2, dim=1)
        x1 = self.cnn1.forward(x[0])
        x2 = self.cnn2.forward(x[1])

        # size_x = batch_size * 18 * 86

        out = torch.cat([x1, x2], dim=1)
        # size_out = batch_size * 128 * 86
        # out = self.lstm.forward(out.transpose(1, 2))
        out = self.fc(out.view(-1, 512))

        return out


class MyDecodeFC(nn.Module):
    def __init__(self):
        super(MyDecodeFC, self).__init__()

        self.fclayers = nn.Sequential(
            nn.Linear(128, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(), )

        self.onehotout = nn.Sequential(
            nn.Linear(4096, 3),
        )

        self.singleout = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fclayers(x)
        x = self.onehotout(x)
        return x


class MyDecodeLSTM(nn.Module):
    def __init__(self):
        super(MyDecodeLSTM, self).__init__()

        self.hidden_size = 128
        self.num_layers = 2
        self.hidden = self.init_hidden()

        self.layer = nn.LSTM(128, self.hidden_size, self.num_layers, batch_first=True, dropout=0.1)

    def init_hidden(self):
        # Only for initialization
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        out, (final_hidden_state, final_cell_state) = self.layer(x)
        return out[:, -1, :]


class MyDecoder(nn.Module):
    def __init__(self):
        super(MyDecoder, self).__init__()

        # self.lstm = MyDecodeLSTM()
        self.fc = MyDecodeFC()

    def forward(self, x):
        # x = self.lstm.forward(x)
        x = self.fc.forward(x)
        return x


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.encoder = MyEncoder()
        self.decoder = MyDecoder()

    def intro(self):
        print("[CNN-GAP]-[FC]")

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


if __name__ == "__main__":
    m1 = MyEncoder()
    summary(m1, input_size=(2, 90, 100))
    m1.intro()
