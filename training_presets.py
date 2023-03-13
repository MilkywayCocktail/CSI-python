# My Dataset
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
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
        else:
            print(x.shape, y.shape, "lengths not equal!")

        if number != 0:
            picked = np.random.choice(list(range(total_count)), size=number, replace=False)
            self.seeds = picked
            x = x[picked]
            y = y[picked]

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
    def __init__(self, model, args, train_loader, valid_loader, test_loader, optimizer=torch.optim.Adam):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(model.parameters(), lr=args.learning_rate)
        self.train_loss = None
        self.valid_loss = None
        self.train_epochs_loss = None
        self.valid_epochs_loss = None
        self.total_epochs = 0
        self.__gen_loss_logs__()

    def __gen_loss_logs__(self):
        self.train_loss = []
        self.valid_loss = []
        self.train_epochs_loss = []
        self.valid_epochs_loss = []

    def train(self):
        y_type = torch.long if isinstance(self.args.criterion, nn.CrossEntropyLoss) else torch.float32
        start = time.time()

        for epoch in range(self.args.epochs):
            self.model.train()
            train_epoch_loss = []
            for idx, (data_x, data_y) in enumerate(self.train_loader, 0):
                data_x = data_x.to(torch.float32).to(self.args.device)
                data_y = data_y.to(y_type).to(self.args.device)
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

