import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.stats import norm
from TrainerTS import MyDataset, split_loader, MyArgs, TrainerTeacherStudent


class TrainerVariationalTS(TrainerTeacherStudent):
    def __init__(self, latent_dim):
        super(TrainerVariationalTS, self).__init__()
        self.latent_dim = latent_dim

    @staticmethod
    def __gen_train_loss__():
        train_loss = {'t_train': [],
                      't_valid': [],
                      't_train_epochs': [],
                      't_valid_epochs': [],
                      't_train_kl': [],
                      't_valid_kl': [],
                      't_train_kl_epochs': [],
                      't_valid_kl_epochs': [],
                      't_train_recon': [],
                      't_valid_recon': [],
                      't_train_recon_epochs': [],
                      't_valid_recon_epochs': [],

                      's_train': [],
                      's_valid': [],
                      's_train_epochs': [],
                      's_valid_epochs': [],
                      's_train_straight_epochs': [],
                      's_valid_straight_epochs': [],
                      's_train_distil_epochs': [],
                      's_valid_distil_epochs': [],
                      's_train_image_epochs': [],
                      's_valid_image_epochs': []}
        return train_loss

    @staticmethod
    def __gen_teacher_test__():
        test_loss = {'loss': [],
                     'recon': [],
                     'kl': [],
                     'predicts': [],
                     'groundtruth': []}
        return test_loss

    def train_teacher(self, autosave=False, notion=''):
        start = time.time()

        for epoch in range(self.teacher_args.epochs):
            self.img_encoder.train()
            self.img_decoder.train()
            train_epoch_loss = []
            kl_epoch_loss = []
            recon_epoch_loss = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_y = data_y.to(torch.float32).to(self.teacher_args.device)
                self.teacher_optimizer.zero_grad()
                latent, mu, logvar = self.img_encoder(data_y).data
                output = self.img_decoder(latent)

                recon_loss = self.teacher_args.criterion(output, data_y)
                kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss

                loss.backward()
                self.teacher_optimizer.step()
                train_epoch_loss.append(loss.item())
                self.train_loss['t_train'].append(loss.item())
                self.train_loss['t_train_kl'].append(kl_loss)
                self.train_loss['t_train_recon'].append(recon_loss.item())
                if idx % (len(self.train_loader) // 2) == 0:
                    print("\rTeacher: epoch={}/{},{}/{}of train, loss={}".format(
                        epoch, self.teacher_args.epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_loss['t_train_epochs'].append(np.average(train_epoch_loss))
            self.train_loss['t_train_kl_epochs'].append(np.average(kl_epoch_loss))
            self.train_loss['t_train_recon_epochs'].append(np.average(recon_epoch_loss))
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
        valid_kl_epoch_loss = []
        valid_recon_epoch_loss = []

        for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
            data_y = data_y.to(torch.float32).to(self.teacher_args.device)
            latent, mu, logvar = self.img_encoder(data_y).data
            output = self.img_decoder(latent)

            recon_loss = self.teacher_args.criterion(output, data_y)
            kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            valid_epoch_loss.append(loss.item())
            self.train_loss['t_valid'].append(loss.item())
            self.train_loss['t_valid_kl'].append(kl_loss)
            self.train_loss['t_valid_recon'].append(recon_loss.item())
        self.train_loss['t_valid_epochs'].append(np.average(valid_epoch_loss))
        self.train_loss['t_valid_kl_epochs'].append(np.average(valid_kl_epoch_loss))
        self.train_loss['t_valid__recon_epochs'].append(np.average(valid_recon_epoch_loss))

    def test_teacher(self, mode='test'):
        self.t_test_loss = self.__gen_teacher_test__()
        self.img_encoder.eval()
        self.img_decoder.eval()

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, (data_x, data_y) in enumerate(loader, 0):
            data_y = data_y.to(torch.float32).to(self.teacher_args.device)
            if loader.batch_size != 1:
                data_y = data_y[0][np.newaxis, ...]

            latent, mu, logvar = self.img_encoder(data_y).data
            output = self.img_decoder(latent)

            recon_loss = self.teacher_args.criterion(output, data_y)
            kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            self.t_test_loss['loss'].append(loss.item())
            self.t_test_loss['kl'].append(kl_loss)
            self.t_test_loss['recon'].append(recon_loss.item())
            self.t_test_loss['predicts'].append(output.cpu().detach().numpy().squeeze().tolist())
            self.t_test_loss['groundtruth'].append(data_y.cpu().detach().numpy().squeeze().tolist())

            if idx % (len(self.test_loader)//5) == 0:
                print("\rTeacher: {}/{}of test, loss={}".format(idx, len(self.test_loader), loss.item()), end='')

    def plot_teacher_loss(self, autosave=False, notion=''):
        self.__plot_settings__()

        # Training Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Teacher Train Loss')
        axes = fig.subplots(2, 3)
        axes[0].plot(self.train_loss['t_train_epochs'], 'b')
        axes[1].plot(self.train_loss['t_train_kl_epochs'], 'b')
        axes[2].plot(self.train_loss['t_train_recon_epochs'], 'b')

        axes[3].plot(self.train_loss['t_valid_epochs'], 'orange')
        axes[4].plot(self.train_loss['t_valid_kl_epochs'], 'orange')
        axes[5].plot(self.train_loss['t_valid_recon_epochs'], 'orange')

        axes[0].set_title('Train')
        axes[1].set_title('Train KL Loss')
        axes[2].set_title('Train Recon Loss')

        axes[3].set_title('Valid')
        axes[4].set_title('Valid KL Loss')
        axes[5].set_title('Valid Recon Loss')

        for ax in axes:
            ax.set_xlabel('#epoch')
            ax.set_ylabel('loss')
            ax.grid()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_t_train" + notion + '_' + '.jpg')
        plt.show()

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
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Teacher Test Loss')
        subfigs = fig.subfigures(nrows=1, ncols=3)
        subfigs[0].suptitle('Loss')
        subfigs[0].suptitle('KL Loss')
        subfigs[0].suptitle('Recon Loss')
        subfigs[0].scatter(list(range(len(self.t_test_loss['groundtruth']))), self.t_test_loss['loss'], alpha=0.6)
        for i in imgs:
            subfigs[0].scatter(i, self.t_test_loss['loss'][i], c='magenta', marker=(5, 1), linewidths=4)
        subfigs[1].scatter(list(range(len(self.t_test_loss['groundtruth']))), self.t_test_loss['kl'], alpha=0.6)
        for i in imgs:
            subfigs[1].scatter(i, self.t_test_loss['loss'][i], c='magenta', marker=(5, 1), linewidths=4)
        subfigs[2].scatter(list(range(len(self.t_test_loss['groundtruth']))), self.t_test_loss['recon'], alpha=0.6)
        for i in imgs:
            subfigs[2].scatter(i, self.t_test_loss['loss'][i], c='magenta', marker=(5, 1), linewidths=4)
        for subfig in subfigs:
            subfig.xlabel('#Sample')
            subfig.ylabel('Loss')
            subfig.grid()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_t_test" + notion + '_' + '.jpg')
        plt.show()

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

                student_preds, mu, logvar = self.csi_encoder(data_x)
                with torch.no_grad():
                    teacher_preds, t_mu, t_logvar = self.img_encoder(data_y)
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

    def traverse_latent(self, input_img, dim1=0, dim2=1, granularity=11, autosave=False):
        self.img_encoder.eval()
        self.img_decoder.eval()
        trvs = []

        img = input_img.to(torch.float32).to(self.teacher_args.device)
        img = img[0][np.newaxis, ...]

        latent, mu, logvar = self.img_encoder(img).data
        z = self.img_decoder.reparameterize(mu, logvar)
        z = z.squeeze()
        grid_x = norm.ppf(np.linspace(0.05, 0.95, granularity))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, granularity))

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z[dim1], z[dim2] = xi, yi
                output = self.img_decoder(z)
                trvs.append(output.cpu().detach().numpy().squeeze().tolist())

        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Teacher Traverse with dims ' + str(dim1) + str(dim2))
        subfigs, ax = fig.subfigures(nrows=granularity, ncols=granularity)
        for i in range(len(trvs)):
            ax[i].imshow(trvs[i])
            ax[i].axis('off')

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        "_t_traverse_" + str(dim1) + str(dim2) + '_gran' + str(granularity) + '.jpg')
        plt.show()



