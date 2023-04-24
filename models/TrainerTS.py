import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import time


class MyDataset(Data.Dataset):
    def __init__(self, x_path, y_path, number=0):
        self.seeds = None
        self.data = self.load_data(x_path, y_path, number=number)
        print('loaded')

    def __getitem__(self, index):
        return self.data['x'][index], self.data['y'][index]

    def __len__(self):
        return self.data['x'].shape[0]

    def load_data(self, x_path, y_path, number):
        x = np.load(x_path)
        y = np.load(y_path)

        if x.shape[0] == y.shape[0]:
            total_count = x.shape[0]
            if number != 0:
                picked = np.random.choice(list(range(total_count)), size=number, replace=False)
                self.seeds = picked
                x = x[picked]
                y = y[picked]
        else:
            print(x.shape, y.shape, "lengths not equal!")

        return {'x': x, 'y': y}


def split_loader(dataset, train_size, valid_size, test_size, batch_size):
    train_dataset, valid_dataset, test_dataset = Data.random_split(dataset, [train_size, valid_size, test_size])
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    return train_loader, valid_loader, test_loader


class MyArgs:
    def __init__(self, cuda=1, epochs=30, learning_rate=0.001, criterion=nn.CrossEntropyLoss()):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
        self.criterion = criterion


class TrainerTeacherStudent:

    def __init__(self, img_encoder, img_decoder, csi_encoder,
                 teacher_args, student_args,
                 train_loader, valid_loader, test_loader,
                 optimizer=torch.optim.Adam,
                 div_loss=nn.KLDivLoss(reduction='batchmean'),
                 img_loss=nn.SmoothL1Loss(),
                 temperature=20,
                 alpha=0.3):
        self.img_encoder = img_encoder
        self.img_decoder = img_decoder
        self.csi_encoder = csi_encoder

        self.teacher_args = teacher_args
        self.student_args = student_args

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.teacher_optimizer = optimizer([{'params': self.img_encoder.parameters()},
                                           {'params': self.img_decoder.parameters()}],
                                           lr=self.teacher_args.learning_rate)
        self.student_optimizer = optimizer(self.csi_encoder.parameters(), lr=self.student_args.learning_rate)

        self.train_loss = self.__gen_train_loss__()
        self.t_test_loss = self.__gen_teacher_test__()
        self.s_test_loss = self.__gen_student_test__()

        self.teacher_epochs = 0
        self.student_epochs = 0

        self.div_loss = div_loss
        self.temperature = temperature
        self.alpha = alpha
        self.img_loss = img_loss

        # Plot settings
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.rcParams['lines.markersize'] = 10
        plt.rcParams['axes.titlesize'] = 35
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

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
                      's_train_straight_epochs': [],
                      's_valid_straight_epochs': [],
                      's_train_distil_epochs': [],
                      's_valid_distil_epochs': [],
                      's_train_image': [],
                      's_valid_image_epochs': []}
        return train_loss

    @staticmethod
    def __gen_teacher_test__():
        test_loss = {'loss': [],
                     'predicts': [],
                     'groundtruth': []}
        return test_loss

    @staticmethod
    def __gen_student_test__():
        test_loss = {'loss': [],
                     'latent_straight_loss': [],
                     'latent_distil_loss': [],
                     'student_image_loss': [],
                     'teacher_latent_predicts': [],
                     'student_latent_predicts': [],
                     'student_image_predicts': [],
                     'groundtruth': []}
        return test_loss

    def save_status(self, model=None, notion=''):
        if model is None:
            model = ['img_encoder, img_decoder, csi_encoder']
        for mo in model:
            torch.save(eval('self.' + mo + '.state_dict()'),
                       '../Models/' + mo + '_' + str(self.img_encoder) + notion +
                       '_ep' + str(self.teacher_epochs) + '.pth')

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

                image_loss = nn.SmoothL1Loss(image_preds, data_y)
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
            self.train_loss['s_train_image'].append(np.average(image_epoch_loss))
            self.student_epochs += 1

        end = time.time()
        print("\nTotal training time:", end - start, "sec")

        if autosave is True:
            torch.save(self.img_encoder.state_dict(),
                       '../Models/CsiEn_' + str(self.csi_encoder) + notion + '_ep' + str(self.student_epochs) + '.pth')

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
            image_loss = nn.SmoothL1Loss(image_preds, data_y)

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

        for idx, (data_x, data_y) in enumerate(loader, 0):
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
                print("\rTeacher: {}/{}of test, loss={}".format(idx, len(self.test_loader), loss.item()), end='')

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

    def plot_teacher_loss(self, autosave=False, notion=''):

        fig = plt.figure(constrained_layout=True)
        fig.subtitle('Teacher Train Loss')
        plt.subplots(2, 1, 1)
        plt.plot(self.train_loss['t_train_epochs'], 'b')
        plt.title('Train')
        plt.xlabel('#epoch')
        plt.ylabel('loss')
        plt.grid()

        plt.subplots(2, 1, 2)
        plt.plot(self.train_loss['t_valid_epochs'], 'orange')
        plt.title('Validation')
        plt.xlabel('#epoch')
        plt.ylabel('loss')
        plt.grid()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_t_train" + notion + '_' + '.jpg')
        plt.show()

    def plot_student_loss(self, autosave=False, notion=''):

        # Train Loss
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.suptitle('Validation Status')
        axes = axes.flatten()

        loss_items = ('s_train_epochs', 's_train_straight_epochs', 's_train_distil_epochs', 's_train_image_epochs')

        for i, lo in enumerate(loss_items):
            axes[i].plot(self.train_loss[lo], 'b')
            axes[i].set_ylabel('loss')
            axes[i].set_xlabel('#epoch')
            axes[i].grid(True)

        axes[0].set_title('Student Loss')
        axes[1].set_title('Straight Loss')
        axes[2].set_title('Distillation Loss')
        axes[3].set_title('Image Loss')
        plt.tight_layout()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_s_train" + notion + '_' + '.jpg')
        plt.show()

        # Validation Loss
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.suptitle('Validation Status')
        axes = axes.flatten()

        loss_items = ('s_valid_epochs', 's_valid_straight_epochs', 's_valid_distil_epochs', 's_valid_image_epochs')

        for i, lo in enumerate(loss_items):
            axes[i].plot(self.train_loss[lo], 'orange')
            axes[i].set_ylabel('loss')
            axes[i].set_xlabel('#epoch')
            axes[i].grid(True)

        axes[0].set_title('Student Loss')
        axes[1].set_title('Straight Loss')
        axes[2].set_title('Distillation Loss')
        axes[3].set_title('Image Loss')
        plt.tight_layout()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_s_valid" + notion + '_' + '.jpg')
        plt.show()

    def plot_teacher_test(self, autosave=False, notion=''):

        # Depth Images
        imgs = np.random.choice(list(range(len(self.t_test_loss['groundtruth']))), 8)
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

    def plot_student_test(self, autosave=False, notion=''):

        # Depth Images
        imgs = np.random.choice(list(range(len(self.s_test_loss['groundtruth']))), 8)
        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Student Test Results - images')
        subfigs = fig.subfigures(nrows=2, ncols=1)

        subfigs[0].suptitle('Ground Truth')
        ax = subfigs[0].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ima = ax[a].imshow(self.s_test_loss['groundtruth'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))
        subfigs[0].colorbar(ima, ax=ax, shrink=0.8)

        subfigs[1].suptitle('Estimated')
        ax = subfigs[1].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            imb = ax[a].imshow(self.s_test_loss['student_image_predicts'][imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))
        subfigs[1].colorbar(imb, ax=ax, shrink=0.8)

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_s_predict" + notion + '_' + '.jpg')
        plt.show()

        # Latent Vectors
        fig, axes = plt.subplots(nrows=2, ncols=4)
        fig.suptitle('Student Test Results - latent')
        axes = axes.flatten()
        for a in range(len(axes)):
            axes[a].bar(range(256), self.s_test_loss['teacher_latent_predicts'][imgs[a]], width=1, fc='blue',
                        alpha=0.8, label='Teacher')
            axes[a].bar(range(256), self.s_test_loss['student_latent_predicts'][imgs[a]], width=1, fc='orange',
                        alpha=0.8, label='student')
            axes[a].set_title('#' + str(imgs[a]))
            axes[a].grid()

        axes[0].legend()
        plt.tight_layout()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_s_latent" + notion + '_' + '.jpg')
        plt.show()

        # Test Loss
        fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.suptitle('Train Status')
        axes = axes.flatten()
        loss_items = ('loss', 'latent_straight_loss', 'latent_distil_loss', 'student_image_loss')

        for i, lo in enumerate(loss_items):
            axes[i].scatter(list(range(len(self.s_test_loss['groundtruth']))), self.s_test_loss[lo], alpha=0.6)
            for m in imgs:
                axes[i].scatter(m, self.s_test_loss[lo], c='magenta', marker=(5, 1), linewidths=4)
            axes[i].set_ylabel('loss')
            axes[i].set_xlabel('#epoch')
            axes[i].grid(True)

        axes[0].set_title('Student Loss')
        axes[1].set_title('Straight Loss')
        axes[2].set_title('Distillation Loss')
        axes[3].set_title('Image Loss')
        plt.tight_layout()

        if autosave is True:
            plt.savefig('t_ep' + str(self.teacher_epochs) +
                        '_s_ep' + str(self.student_epochs) +
                        "_s_latent" + notion + '_' + '.jpg')
        plt.show()
