import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from TrainerTS import MyDataset, split_loader, MyArgs, TrainerTeacherStudent, bn, Interpolate


# ------------------------------------- #
# Model v03b1
# Added interpolating decoder
# Adjusted for MNIST

# ImageEncoder: in = 128 * 128, out = 1 * 256
# ImageDecoder: in = 1 * 256, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256


class ImageEncoder(nn.Module):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8):
        super(ImageEncoder, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 128 * 128 * 1
            # Out = 64 * 64 * 16
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 64 * 64 * 16
            # Out = 32 * 32 * 32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 32 * 32 * 32
            # Out = 16 * 16 * 64
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 16 * 16 * 64
            # Out = 8 * 8 * 128
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 8 * 8 * 128
            # Out = 4 * 4 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            nn.Sigmoid()
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

        return x.view(-1, 256)


class ImageDecoder(nn.Module):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 1 * 1 * 256
            # Out = 4 * 4 * 128
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 4 * 4 * 128
            # Out = 8 * 8 * 64
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 8 * 8 * 64
            # Out = 16 * 16 * 32
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 16 * 16 * 32
            # Out = 32 * 32 * 16
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 32 * 32 * 16
            # Out = 64 * 64 * 8
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 64 * 64 * 8
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'Model_v03a1_ImgDe_' + self.fc

    def forward(self, x):

        x = self.fclayers(x.view(-1, self.latent_dim))
        x = self.layer1(x.view(-1, 256, 1, 1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x.view(-1, 1, 128, 128)


class ImageDecoderInterp(ImageDecoder):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderInterp, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(16, 16)),
            # In = 4 * 4 * 32
            # Out = 16 * 16 * 16
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(64, 64)),
            # In = 16 * 16 * 16
            # Out = 64 * 64 * 8
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            bn(4, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(128, 128)),
            # In = 64 * 64 * 8
            # Out = 128 * 128 * 4
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 128 * 128 * 4
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'Model_v03b2_ImgDe_' + self.fc

    def forward(self, x):

        z = self.fclayers(x.view(-1, self.latent_dim))
        z = self.layer1(z.view(-1, 32, 4, 4))
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)

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
            nn.AvgPool1d(kernel_size=8 * 42, stride=1, padding=0)
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
        return 'Model_v02a1_CsiEn_' + self.bottleneck.capitalize()

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

        return out


class TrainerTS(TrainerTeacherStudent):

    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 optimizer=torch.optim.Adam,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.SmoothL1Loss(),
                 temperature=20,
                 alpha=0.3,
                 latent_dim=8,
                 ):
        super(TrainerTS, self).__init__(img_encoder=img_encoder, img_decoder=img_decoder, csi_encoder=csi_encoder,
                                        teacher_args=teacher_args, student_args=student_args,
                                        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                                        optimizer=optimizer,
                                        div_loss=div_loss,
                                        img_loss=img_loss,
                                        temperature=temperature,
                                        alpha=alpha)
        self.latent_dim = latent_dim

    def train_teacher(self, autosave=False, notion=''):
        start = time.time()

        for epoch in range(self.teacher_args.epochs):
            self.img_encoder.train()
            self.img_decoder.train()
            train_epoch_loss = []
            for idx, (data_y, data_x) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.teacher_args.device)
                self.teacher_optimizer.zero_grad()
                latent = self.img_encoder(data_y).data
                output = self.img_decoder(latent)
                loss = self.teacher_args.criterion(output, data_y)
                loss.backward()
                self.teacher_optimizer.step()
                train_epoch_loss.append(loss.item())
                self.train_loss['t_train'].append(loss.item())
                if idx % (len(self.train_loader) // 2) == 0:
                    print("\rTeacher: epoch={}/{},{}/{}of train, loss={}".format(
                        epoch, self.teacher_args.epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_loss['t_train_epochs'].append(np.average(train_epoch_loss))
            self.teacher_epochs += 1

        end = time.time()
        print("\nTotal training time:", end - start, "sec")

        if autosave is True:
            torch.save(self.img_encoder.state_dict(),
                       '../Models/ImgEn_' + str(self.img_encoder) + notion + '_tep' + str(self.teacher_epochs) + '.pth')
            torch.save(self.img_decoder.state_dict(),
                       '../Models/ImgDe_' + str(self.img_decoder) + notion + '_tep' + str(self.teacher_epochs) + '.pth')

        # =====================valid============================
        self.img_encoder.eval()
        self.img_decoder.eval()
        valid_epoch_loss = []

        for idx, (data_y, data_x) in enumerate(self.valid_loader, 0):
            data_y = data_y.to(torch.float32).to(self.teacher_args.device)
            latent = self.img_encoder(data_y).data
            output = self.img_decoder(latent)
            loss = self.teacher_args.criterion(output, data_y)
            valid_epoch_loss.append(loss.item())
            self.train_loss['t_valid'].append(loss.item())
        self.train_loss['t_valid_epochs'].append(np.average(valid_epoch_loss))

    def train_student(self, autosave=False, notion=''):
        start = time.time()

        for epoch in range(self.student_args.epochs):
            self.img_encoder.eval()
            self.img_decoder.eval()
            self.csi_encoder.train()
            train_epoch_loss = []
            straight_epoch_loss = []
            distil_epoch_loss = []
            image_epoch_loss = []

            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.teacher_args.device)
                data_y = data_y.to(torch.float32).to(self.teacher_args.device)

                student_preds = self.csi_encoder(data_x)
                with torch.no_grad():
                    teacher_preds = self.img_encoder(data_y)
                    image_preds = self.img_decoder(student_preds)

                image_loss = self.img_loss(image_preds, data_y)
                student_loss = self.student_args.criterion(student_preds, teacher_preds)

                distil_loss = self.div_loss(nn.functional.softmax(student_preds / self.temperature, -1),
                                            nn.functional.softmax(teacher_preds / self.temperature, -1))

                loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

                self.train_loss['s_train'].append(loss.item())

                self.student_optimizer.zero_grad()
                loss.backward()
                self.student_optimizer.step()

                train_epoch_loss.append(loss.item())
                straight_epoch_loss.append(student_loss.item())
                distil_epoch_loss.append(distil_loss.item())
                image_epoch_loss.append(image_loss.item())

                if idx % (len(self.train_loader) // 2) == 0:
                    print("\rStudent: epoch={}/{},{}/{}of train, student loss={}, distill loss={}".format(
                        epoch, self.student_args.epochs, idx, len(self.train_loader),
                        loss.item(), distil_loss.item()), end='')

            self.train_loss['s_train_epochs'].append(np.average(train_epoch_loss))
            self.train_loss['s_train_straight_epochs'].append(np.average(straight_epoch_loss))
            self.train_loss['s_train_distil_epochs'].append(np.average(distil_epoch_loss))
            self.train_loss['s_train_image_epochs'].append(np.average(image_epoch_loss))
            self.student_epochs += 1

        end = time.time()
        print("\nTotal training time:", end - start, "sec")

        if autosave is True:
            torch.save(self.csi_encoder.state_dict(),
                       '../Models/CsiEn_' + str(self.csi_encoder) + notion + '_tep' + str(self.teacher_epochs) +
                       '_sep' + str(self.student_epochs) + '.pth')

        # =====================valid============================
        self.csi_encoder.eval()
        self.img_encoder.eval()
        self.img_decoder.eval()
        valid_epoch_loss = []
        straight_epoch_loss = []
        distil_epoch_loss = []
        image_epoch_loss = []

        for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
            data_x = data_x.to(torch.float32).to(self.student_args.device)
            data_y = data_y.to(torch.float32).to(self.student_args.device)

            teacher_preds = self.img_encoder(data_y)
            student_preds = self.csi_encoder(data_x)
            image_preds = self.img_decoder(student_preds)
            image_loss = self.img_loss(image_preds, data_y)

            student_loss = self.student_args.criterion(student_preds, teacher_preds)

            distil_loss = self.div_loss(nn.functional.softmax(student_preds / self.temperature, -1),
                                        nn.functional.softmax(teacher_preds / self.temperature, -1))

            loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss
            self.train_loss['s_valid'].append(loss.item())

            valid_epoch_loss.append(loss.item())
            straight_epoch_loss.append(student_loss.item())
            distil_epoch_loss.append(distil_loss.item())
            image_epoch_loss.append(image_loss.item())

        self.train_loss['s_valid_epochs'].append(np.average(valid_epoch_loss))
        self.train_loss['s_valid_straight_epochs'].append(np.average(straight_epoch_loss))
        self.train_loss['s_valid_distil_epochs'].append(np.average(distil_epoch_loss))
        self.train_loss['s_valid_image_epochs'].append(np.average(image_epoch_loss))

    def test_teacher(self, mode='test'):
        self.t_test_loss = self.__gen_teacher_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_y, data_x) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.teacher_args.device)
            if loader.batch_size != 1:
                data_y = data_y[0][np.newaxis, ...]

            latent = self.img_encoder(data_y)
            output = self.img_decoder(latent)
            loss = self.teacher_args.criterion(output, data_y)

            self.t_test_loss['loss'].append(loss.item())
            self.t_test_loss['predicts'].append(output.cpu().detach().numpy().squeeze().tolist())
            self.t_test_loss['groundtruth'].append(data_y.cpu().detach().numpy().squeeze().tolist())

            if idx % (len(self.test_loader)//5) == 0:
                print("\rTeacher: {}/{}of test, loss={}".format(idx, len(loader), loss.item()), end='')

    def test_student(self, mode='test'):
        self.s_test_loss = self.__gen_student_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y) in enumerate(loader, 0):
            data_x = data_x.to(torch.float32).to(self.student_args.device)
            data_y = data_y.to(torch.float32).to(self.teacher_args.device)

            teacher_latent_preds = self.img_encoder(data_y)
            student_latent_preds = self.csi_encoder(data_x)
            student_image_preds = self.img_decoder(student_latent_preds)
            student_loss = self.student_args.criterion(student_latent_preds, teacher_latent_preds)
            image_loss = self.img_loss(student_image_preds, data_y)

            distil_loss = self.div_loss(nn.functional.softmax(student_latent_preds / self.temperature, -1),
                                        nn.functional.softmax(teacher_latent_preds / self.temperature, -1))

            loss = self.alpha * student_loss + (1 - self.alpha) * distil_loss

            self.s_test_loss['loss'].append(image_loss.item())
            self.s_test_loss['latent_straight_loss'].append(student_loss.item())
            self.s_test_loss['latent_distil_loss'].append(loss.item())
            self.s_test_loss['student_image_loss'].append(image_loss.item())
            self.s_test_loss['student_latent_predicts'].append(student_latent_preds.cpu().detach().numpy().squeeze().tolist())
            self.s_test_loss['teacher_latent_predicts'].append(teacher_latent_preds.cpu().detach().numpy().squeeze().tolist())
            self.s_test_loss['student_image_predicts'].append(student_image_preds.cpu().detach().numpy().squeeze().tolist())
            self.s_test_loss['groundtruth'].append(data_y.cpu().detach().numpy().squeeze().tolist())

            if idx % (len(self.test_loader) // 5) == 0:
                print("\rStudent: {}/{}of test, student loss={}, distill loss={}, image loss={}".format(
                    idx, len(self.test_loader), student_loss.item(), distil_loss.item(), image_loss.item()), end='')

    def plot_teacher_test(self, autosave=False, notion=''):
        self.__plot_settings__()

        # Depth Images
        imgs = np.random.choice(list(range(len(self.t_test_loss['groundtruth']))), 8)
        imgs = np.sort(imgs)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Teacher Test Results')
        subfigs = fig.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle('Ground Truth')
        ax = subfigs[0].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ima = ax[a].imshow(self.t_test_loss['groundtruth'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))
        subfigs[0].colorbar(ima, ax=ax, shrink=0.8)

        subfigs[1].suptitle('Estimated')
        ax = subfigs[1].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            imb = ax[a].imshow(self.t_test_loss['predicts'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))
        subfigs[1].colorbar(imb, ax=ax, shrink=0.8)

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_t_predict" + notion + '_' + '.jpg')
        plt.show()

        # Test Loss
        plt.figure(constrained_layout=True)
        plt.title('Teacher Test Loss')
        plt.scatter(list(range(len(self.t_test_loss['groundtruth']))), self.t_test_loss['loss'], alpha=0.6)
        for i in imgs:
            plt.scatter(i, self.t_test_loss['loss'][i], c='magenta', marker=(5, 1), linewidths=4)
        plt.xlabel('#Sample')
        plt.ylabel('Loss')
        plt.grid()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_t_test" + notion + '_' + '.jpg')
        plt.show()

    def save_all_params(self, notion=''):
        save_path = '../Models/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.img_encoder.state_dict(),
                   '../Models/ImgEn_' + str(self.img_encoder) + notion + '_tep' + str(self.teacher_epochs) + '.pth')
        torch.save(self.img_decoder.state_dict(),
                   '../Models/ImgDe_' + str(self.img_decoder) + notion + '_tep' + str(self.teacher_epochs) + '.pth')

        torch.save(self.csi_encoder.state_dict(),
                   '../Models/CSIEn_' + str(self.csi_encoder) + notion + '_tep' + str(self.teacher_epochs) +
                   '_sep' + str(self.student_epochs) + '.pth')


if __name__ == "__main__":
    m1 = ImageDecoder(batchnorm=False, active_func=nn.ReLU(), latent_dim=256)
    summary(m1, input_size=(1, 256))