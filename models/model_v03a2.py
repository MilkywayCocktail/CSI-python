import torch
import torch.nn as nn
from torchinfo import summary


# ------------------------------------- #
# Model v03a1
# VAE version
# Adaptive to normalized depth images
# ImageEncoder: in = 128 * 128, out = 1 * 256
# ImageDecoder: in = 1 * 256, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256

def bn(channels, batchnorm):
    if batchnorm:
        return nn.BatchNorm2d(channels)
    else:
        return nn.Identity(channels)


class ImageEncoder(nn.Module):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8):
        super(ImageEncoder, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
            # In = 128 * 128 * 1
            # Out = 64 * 64 * 16
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
            # In = 64 * 64 * 16
            # Out = 32 * 32 * 32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
            # In = 32 * 32 * 32
            # Out = 16 * 16 * 64
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
            # In = 16 * 16 * 64
            # Out = 8 * 8 * 128
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
            # In = 8 * 8 * 128
            # Out = 4 * 4 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2 * self.latent_dim),
            # nn.Sigmoid()
        )

    def __str__(self):
        return 'Model_v02a1_ImgEn_' + self.bottleneck.capitalize()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.bottleneck == 'fc':
            x = self.fclayers(x.view(-1, 4 * 4 * 256))
        elif self.bottleneck == 'gap':
            x = self.gap(x)
            x = nn.Sigmoid(x)

        mu, logvar = x.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)

        return x, mu, logvar


class ImageDecoder(nn.Module):
    def __init__(self, batchnorm=False, latent_dim=8):
        super(ImageDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 16 * 16 * 1
            # Out = 32 * 32 * 64
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 32 * 32 * 64
            # Out = 64 * 64 * 32
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            nn.Sigmoid(),
            # In = 64 * 64 * 32
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'Model_v03a1_ImgDe_' + self.fc

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(logvar/2)

    def forward(self, x):

        mu, logvar = x.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        z = self.fclayers(z)

        z = self.layer1(z.view(-1, 1, 16, 16))
        z = self.layer2(z)
        z = self.layer3(z)

        return z.view(-1, 1, 128, 128)


class CsiEncoder(nn.Module):
    def __init__(self, bottleneck='last', batchnorm=False, latent_dim=8):
        super(CsiEncoder, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 90 * 100 * 1
            # Out = 30 * 98 * 16
                )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=0),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 30 * 98 * 16
            # Out = 14 * 48 * 32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 14 * 48 * 32
            # Out = 12 * 46 * 64
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 12 * 46 * 64
            # Out = 10 * 44 * 128
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 10 * 44 * 128
            # Out = 8 * 42 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool1d(kernel_size=8*42, stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(256 * 8 * 42, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.lstm = nn.Sequential(
            nn.LSTM(512, 2 * self.latent_dim, 2, batch_first=True, dropout=0.1)
        )

    def __str__(self):
        return 'Model_v03a1_CsiEn_' + self.bottleneck.capitalize()

    def forward(self, x):
        x = torch.chunk(x.view(-1, 2, 90, 100), 2, dim=1)
        x1 = self.layer1(x[0])
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.layer5(x1)

        x2 = self.layer1(x[1])
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.layer5(x2)

        out = torch.cat([x1, x2], dim=1)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(out.view(-1, 512, 8 * 42).transpose(1, 2))

        if self.bottleneck == 'full_fc':
            out = self.fclayers(out.view(-1, 256 * 8 * 42))

        elif self.bottleneck == 'full_gap':
            out = self.gap(out.transpose(1, 2))

        elif self.bottleneck == 'last':
            out = out[:, -1, :]

        mu, logvar = out.view(-1, 2 * self.latent_dim).chunk(2, dim=-1)

        return out, mu, logvar


if __name__ == "__main__":
    m1 = ImageEncoder(batchnorm=False)
    summary(m1, input_size=(1, 128, 128))
    #m2 = ImageDecoder(batchnorm=False)
    #summary(m1, input_size=(1, 16))
    #m3 = CsiEncoder(batchnorm=False)
    #summary(m1, input_size=(2, 90, 100))
