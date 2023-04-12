import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import time

import seaborn as sns
from sklearn.metrics import confusion_matrix


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


class Trainer:
    def __init__(self, model, args, train_loader, valid_loader, test_loader, optimizer):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=self.args.learning_rate)
        self.train_loss = None
        self.valid_loss = None
        self.train_epochs_loss = None
        self.valid_epochs_loss = None
        self.test_loss = None
        self.estimates = None
        self.predicts = None
        self.groundtruth = None
        self.total_epochs = 0
        self.y_type = None
        self.__gen_train_loss__()
        self.__gen_test_loss__()
        self.__gen_y_type__()

    def __gen_train_loss__(self):
        self.train_loss = []
        self.valid_loss = []
        self.train_epochs_loss = []
        self.valid_epochs_loss = []

    def __gen_test_loss__(self):
        self.test_loss = []
        self.estimates = []
        self.predicts = []
        self.groundtruth = []

    def __gen_y_type__(self):
        if isinstance(self.args.criterion, nn.CrossEntropyLoss):
            self.y_type = torch.long
        else:
            self.y_type = torch.float32

    def train_and_eval(self, autosave=False, notion=''):
        start = time.time()

        for epoch in range(self.args.epochs):
            self.model.train()
            train_epoch_loss = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args.device)
                data_y = data_y.to(self.y_type).to(self.args.device)
                self.optimizer.zero_grad()
                outputs = self.model(data_x)
                loss = self.args.criterion(outputs, data_y)
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss.item())
                self.train_loss.append(loss.item())
                if idx % (len(self.train_loader) // 2) == 0:
                    print("\repoch={}/{},{}/{}of train, loss={}".format(
                        epoch, self.args.epochs, idx, len(self.train_loader), loss.item()), end='')
            self.train_epochs_loss.append(np.average(train_epoch_loss))
            self.total_epochs += 1

        end = time.time()
        print("\nTotal training time:", end - start, "sec")

        if autosave is True:
            torch.save(self.model.state_dict(),
                       '../Models/' + str(self.model) + notion + '_ep' + str(self.total_epochs) + '.pth')

        # =====================valid============================
        self.model.eval()
        valid_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(self.valid_loader, 0):
            data_x = data_x.to(torch.float32).to(self.args.device)
            data_y = data_y.to(torch.long).to(self.args.device)
            outputs = self.model(data_x)
            loss = self.args.criterion(outputs, data_y)
            valid_epoch_loss.append(loss.item())
            self.valid_loss.append(loss.item())
        self.valid_epochs_loss.append(np.average(valid_epoch_loss))

        # ==================early stopping======================
        # early_stopping(valid_epochs_loss[-1],model=MyModel,path=r'')
        # if early_stopping.early_stop:
        #    print("Early stopping")
        #    break
        # ====================adjust lr========================
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        # if epoch in lr_adjust.keys():
        #    lr = lr_adjust[epoch]
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = lr
        #    print('Updating learning rate to {}'.format(lr))

    def plot_loss(self, autosave=False, notion=''):
        plt.figure()
        plt.suptitle("Training loss and Validation loss")
        plt.subplot(2, 1, 1)
        plt.plot(self.train_epochs_loss[1:], 'b', label='training_loss')
        plt.ylabel('loss')
        plt.xlabel('#epoch')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.valid_loss, 'b', label='validation_loss')
        plt.ylabel('loss')
        plt.xlabel('#iter')
        plt.legend()
        if autosave is True:
            plt.savefig(str(self.model) + "_loss" + notion + '_ep' + str(self.total_epochs) + '.jpg')
        plt.show()

    def test(self):
        self.__gen_test_loss__()
        self.model.eval()
        for idx, (data_x, data_y) in enumerate(self.test_loader, 0):
            data_x = data_x.to(torch.float32).to(self.args.device)
            data_y = data_y.to(self.y_type).to(self.args.device)
            outputs = self.model(data_x)
            loss = self.args.criterion(outputs, data_y)
            self.estimates.append(outputs.cpu().detach().numpy().squeeze().tolist())
            self.groundtruth.append(data_y.cpu().detach().numpy().squeeze().tolist())
            self.test_loss.append(np.mean(loss.item()))
            if idx % (len(self.test_loader)//5) == 0:
                print("\r{}/{}of test, loss={}".format(idx, len(self.test_loader), loss.item()), end='')

    def plot_test_results(self):
        pass


class TrainerClassifier(Trainer):

    def plot_test_results(self):
        self.predicts = [np.argmax(row) for row in self.estimates]
        plt.clf()
        sns.set()
        f, ax = plt.subplots()
        cf = confusion_matrix(self.groundtruth, self.predicts)

        sns.heatmap(cf, annot=True, ax=ax)

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.show()


class TrainerGenImage(Trainer):

    def plot_test_results(self):

        imgs = np.random.choice(list(range(len(self.groundtruth))), 8)

        fig = plt.figure(constrained_layout=True)
        fig.suptitle('Estimation Results')

        subfigs = fig.subfigures(nrows=2, ncols=1)
        subfigs[0].suptitle('Ground Truth')
        ax = subfigs[0].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].imshow(self.groundtruth[imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        subfigs[1].suptitle('Estimated')
        ax = subfigs[1].subplots(nrows=1, ncols=8)
        for a in range(len(ax)):
            ax[a].imshow(self.estimates[imgs[a]])
            ax[a].axis('off')
            ax[a].set_title('#' + str(imgs[a]))
            ax[a].set_xlabel(str(imgs[a]))

        plt.show()
