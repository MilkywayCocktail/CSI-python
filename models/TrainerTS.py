import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time


class TrainerTeacherStudent:

    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 optimizer=torch.optim.Adam,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.SmoothL1Loss,
                 temperature=20):
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        self.csi_encoder = csi_encoder

        self.teacher_args = teacher_args
        self.student_args = student_args

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.teacher_optimizer = optimizer
        self.student_optimizer = optimizer

        self.train_loss = self.__gen_train_loss__()
        self.t_test_loss = self.__gen_teacher_loss__()
        self.s_test_loss = self.__gen_student_loss__()

        self.teacher_epochs = 0
        self.student_epochs = 0

        self.div_loss = div_loss
        self.temperature = temperature
        self.img_loss = img_loss

    @staticmethod
    def __gen_train_loss__():
        train_loss = {'t_train': [],
                      't_valid': [],
                      't_train_epochs': [],
                      't_valid_epochs': [],
                      's_train': [],
                      's_valid': [],
                      's_train_epochs': [],
                      's_valid_epochs': [],
                      's_train_distil': [],
                      's_valid_distil': [],
                      's_train_epochs_distil': [],
                      's_valid_epochs_distil': []}
        return train_loss

    @staticmethod
    def __gen_teacher_loss__():
        test_loss = {'loss': [],
                     'predicts': [],
                     'groundtruth': []}
        return test_loss

    @staticmethod
    def __gen_student_loss__():
        test_loss = {'loss': [],
                     'latent_loss': [],
                     'latent_distil_loss': [],
                     'teacher_latent_predicts': [],
                     'student_latent_predicts': [],
                     'student_image_predicts': [],
                     'groundtruth': []}
        return test_loss

    def train_teacher(self, autosave=False, notion=''):
        start = time.time()

        for epoch in range(self.teacher_args.epochs):
            self.img_encoder.train()
            self.img_decoder.train()
            train_epoch_loss = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
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
                       '../Models/ImgEn_' + str(self.img_encoder) + notion + '_ep' + str(self.teacher_epochs) + '.pth')
            torch.save(self.img_decoder.state_dict(),
                       '../Models/ImgDe_' + str(self.img_decoder) + notion + '_ep' + str(self.teacher_epochs) + '.pth')

        # =====================valid============================
        self.img_encoder.eval()
        self.img_decoder.eval()
        valid_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
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
            self.csi_encoder.train()
            train_epoch_loss = []
            train_epoch_loss_distil = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.teacher_args.device)
                data_y = data_y.to(torch.float32).to(self.teacher_args.device)

                with torch.no_grad():
                    teacher_preds = self.img_encoder(data_y)

                student_preds = self.csi_encoder(data_x)
                student_loss = self.student_args.criterion(student_preds. teacher_preds)

                distil_loss = self.div_loss(nn.functional.softmax(
                    student_preds / self.temperature), nn.functional.softmax((teacher_preds / self.temperature)))

                self.train_loss['s_train'].append(student_loss.item())
                self.train_loss['s_train_distil'].append(distil_loss.item())

                self.student_optimizer.zero_grad()
                distil_loss.backward()
                self.student_optimizer.step()

                train_epoch_loss.append(student_loss.item())
                train_epoch_loss_distil.append(distil_loss.item())
                if idx % (len(self.train_loader) // 2) == 0:
                    print("\rStudent: epoch={}/{},{}/{}of train, student loss={}, distill loss={}".format(
                        epoch, self.student_args.epochs, idx, len(self.train_loader),
                        student_loss.item(), distil_loss.item()), end='')

            self.train_loss['s_train_epochs'].append(np.average(train_epoch_loss))
            self.train_loss['s_train_epochs_distil'].append(np.average(train_epoch_loss_distil))
            self.student_epochs += 1

        end = time.time()
        print("\nTotal training time:", end - start, "sec")

        if autosave is True:
            torch.save(self.img_encoder.state_dict(),
                       '../Models/CsiEn_' + str(self.csi_encoder) + notion + '_ep' + str(self.student_epochs) + '.pth')

        # =====================valid============================
        self.csi_encoder.eval()
        self.img_encoder.eval()
        valid_epoch_loss = []
        valid_epoch_loss_distil = []
        for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
            data_x = data_x.to(torch.float32).to(self.student_args.device)
            data_y = data_y.to(torch.float32).to(self.student_args.device)

            teacher_preds = self.img_encoder(data_y)
            student_preds = self.csi_encoder(data_x)
            student_loss = self.student_args.criterion(student_preds.teacher_preds)

            distil_loss = self.div_loss(nn.functional.softmax(
                student_preds / self.temperature), nn.functional.softmax((teacher_preds / self.temperature)))

            self.train_loss['s_valid'].append(student_loss.item())
            self.train_loss['s_valid_distil'].append(distil_loss.item())
            valid_epoch_loss.append(student_loss.item())
            valid_epoch_loss_distil.append(distil_loss.item())

        self.train_loss['s_valid_epochs'].append(valid_epoch_loss)
        self.train_loss['s_valid_epochs_distil'].append(valid_epoch_loss_distil)

    def test_teacher(self):
        self.t_test_loss = self.__gen_teacher_loss__()
        self.img_encoder.eval()
        self.img_decoder.eval()
        for idx, (data_x, data_y) in enumerate(self.test_loader, 0):
            data_y = data_y.to(torch.float32).to(self.teacher_args.device)

            latent = self.img_encoder(data_y)
            output = self.img_decoder(latent)
            loss = self.teacher_args.criterion(output, data_y)

            self.t_test_loss['loss'].append(loss.item())
            self.t_test_loss['predicts'].append(output.cpu().detach().numpy().squeeze().tolist())
            self.t_test_loss['groundtruth'].append(data_y.cpu().detach().numpy().squeeze().tolist())

            if idx % (len(self.test_loader)//5) == 0:
                print("\rTeacher: {}/{}of test, loss={}".format(idx, len(self.test_loader), loss.item()), end='')

    def test_student(self):
        self.s_test_loss = self.__gen_student_loss__()
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.csi_encoder.eval()
        for idx, (data_x, data_y) in enumerate(self.test_loader, 0):
            data_x = data_x.to(torch.float32).to(self.student_args.device)
            data_y = data_y.to(torch.float32).to(self.teacher_args.device)

            teacher_latent_preds = self.img_encoder(data_y)
            student_latent_preds = self.csi_encoder(data_x)
            student_image_preds = self.img_decoder(student_latent_preds)
            student_loss = self.student_args.criterion(student_latent_preds, teacher_latent_preds)
            image_loss = self.img_loss(student_image_preds, data_y)

            distil_loss = self.div_loss(nn.functional.softmax(
                student_latent_preds / self.temperature), nn.functional.softmax((teacher_latent_preds / self.temperature)))

            self.s_test_loss['loss'].append(image_loss.item())
            self.s_test_loss['latent_loss'].append(student_loss.item())
            self.s_test_loss['latent_distil_loss'].append(distil_loss.item())
            self.s_test_loss['student_predicts'].append(student_latent_preds.cpu().detach().numpy().squeeze().tolist())
            self.s_test_loss['teacher_predicts'].append(teacher_latent_preds.cpu().detach().numpy().squeeze().tolist())
            self.s_test_loss['student_image_predicts'].append(student_image_preds.cpu().detach().numpy().squeeze().tolist())
            self.s_test_loss['groundtruth'].append(data_y.cpu().detach().numpy().squeeze().tolist())

            if idx % (len(self.test_loader) // 5) == 0:
                print("\rStudent: {}/{}of test, student loss={}, distill loss={}, image loss={}".format(
                    idx, len(self.test_loader), student_loss.item(), distil_loss.item(), image_loss.item()), end='')

    def plot_train_loss(self, autosave=False, notion=''):

        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Train Status')
        subfigs = fig.subfigures(nrows=1, ncols=3)

        subfigs[0].suptitle('Teacher')
        ax = subfigs[0].subplots(nrows=2, ncols=1)
        for a in ax:
            a.set_title('Train')
            a.ylabel('loss')
            a.xlabel('#epoch')
            a.grid(True)

        ax[0].plot(self.train_loss['t_train_epochs'][1:], 'b', label='training_loss')
        ax[1].plot(self.train_loss['t_valid_epochs'][1:], 'orange', label='training_loss')

        subfigs[1].suptitle('Student')
        ax = subfigs[1].subplots(nrows=2, ncols=1)
        for a in ax:
            a.set_title('Train')
            a.ylabel('loss')
            a.xlabel('#epoch')
            a.grid(True)

        ax[0].plot(self.train_loss['s_train_epochs'][1:], 'b', label='training_loss')
        ax[1].plot(self.train_loss['s_valid_epochs'][1:], 'orange', label='training_loss')

        subfigs[2].suptitle('Distillation')
        ax = subfigs[2].subplots(nrows=2, ncols=1)
        for a in ax:
            a.set_title('Train')
            a.ylabel('loss')
            a.xlabel('#epoch')
            a.grid(True)

        ax[0].plot(self.train_loss['s_train_epochs_distil'][1:], 'b', label='training_loss')
        ax[1].plot(self.train_loss['s_valid_epochs_distil'][1:], 'orange', label='training_loss')

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_loss" + notion + '_' + '.jpg')
        plt.show()

    def plot_teacher_test(self):

        imgs = np.random.choice(list(range(len(self.t_test_loss['groundtruth']))), 8)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Teacher Test Results')
        subfigs = fig.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle('Ground Truth')
        ax = subfigs[0].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].imshow(self.t_test_loss['groundtruth'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        subfigs[1].suptitle('Estimated')
        ax = subfigs[1].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].imshow(self.t_test_loss['predicts'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        plt.show()

    def plot_student_test(self):

        imgs = np.random.choice(list(range(len(self.s_test_loss['groundtruth']))), 8)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Student Test Results')
        subfigs = fig.subfigures(nrows=4, ncols=1)

        subfigs[0].suptitle('Ground Truth')
        ax = subfigs[0].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].imshow(self.s_test_loss['groundtruth'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        subfigs[1].suptitle('Estimated')
        ax = subfigs[1].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].imshow(self.t_test_loss['predicts'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        subfigs[2].suptitle('Teacher\'s Latent')
        ax = subfigs[2].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].plot(self.t_test_loss['teacher_latent_predicts'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        subfigs[3].suptitle('Student\'s Latent')
        ax = subfigs[3].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].imshow(self.t_test_loss['student_latent_predicts'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        plt.show()

