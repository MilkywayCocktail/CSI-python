import torch
import torch.utils.data as Data
import numpy as np
from PIL import Image


class MyDataset(Data.Dataset):
    """
    DATASET READER
    """
    def __init__(self, name, x_path, y_path, img_size=(128, 128), transform=None, img='y', int_image=False, number=0):
        """
        Wraps a dataset.\n
        :param name: the name of the dataset
        :param x_path: path of x file (npy)
        :param y_path: path of y file (npy)
        :param img_size: original image size (height * width)
        :param transform: apply torchvision.transforms
        :param img: whether 'y' or 'x'. Default is 'y'
        :param int_image: whether convert images to np.uint8. Default is False
        :param number: select a number of samples. Default is 0 (all)
        """
        self.name = name
        self.seeds = None
        self.img_size = img_size
        self.transform = transform
        self.img = img
        self.int_img = int_image
        self.data = self.__load_data__(x_path, y_path, number=number)
        print('loaded')

    def __convert__(self, sample):
        """
        Optionally convert a sample to np.uint8.
        :param sample: image
        :return: converted image
        """
        if self.int_img:
            return np.uint8(np.array(sample * 255))
        else:
            return np.array(sample)

    def __transform__(self, sample):
        """
        Optionally apply transforms on images.\n
        :param sample: image
        :return: transformed image
        """
        if self.transform:
            return self.transform(Image.fromarray((self.__convert__(sample)).squeeze(), mode='L'))
        else:
            return self.__convert__(sample)

    def __getitem__(self, index):
        """
        Retrieving samples.\n
        :param index: index of sample
        :return: x, y, index
        """
        if self.img == 'y':
            return self.data['x'][index], self.__transform__(self.data['y'][index]), index
        elif self.img == 'x':
            return self.__transform__(self.data['x'][index]), self.data['y'][index], index

    def __len__(self):
        return self.data['x'].shape[0]

    def __load_data__(self, x_path, y_path, number):
        """
        Load data.\n
        :param x_path: path of x file (npy)
        :param y_path: path of y file (npy)
        :param number: select a number of samples. Default is 0 (all)
        :return: loaded dataset
        """
        x = np.load(x_path)
        y = np.load(y_path)
        print(f"{self.name}: loaded x {x.shape}, y {y.shape}")
        if self.img == 'x':
            x = x.reshape((-1, 1, self.img_size[0], self.img_size[1]))
        elif self.img == 'y':
            y = y.reshape((-1, 1, self.img_size[0], self.img_size[1]))

        if number != 0:
            if x.shape[0] == y.shape[0]:
                total_count = x.shape[0]
                picked = np.random.choice(list(range(total_count)), size=number, replace=False)
                self.seeds = picked
                x = x[picked]
                y = y[picked]
            else:
                print("Lengths not equal!")

        return {'x': x, 'y': y}


class MnistDataset(MyDataset):
    """
    DATASET READER FOR MNIST
    """
    def __init__(self, name, mnist, img_size=(28, 28), transform=None, swap_xy=False, number=0):
        """
        Load MNIST data.
        :param mnist: path of mnist file (npy)
        :param img_size: original image size (height * width)
        :param transform: apply torchvision.transforms
        :param swap_xy: whether swap the x and y in dataset. Default is False
        :param number: select a number of samples. Default is 0 (all)
        """
        MyDataset.__init__(name=name, x_path=None, y_path=None, img_size=img_size)
        self.swap_xy = swap_xy
        self.data = self.__load_data__(mnist, number=number)
        print('loaded')

    def __load_data__(self, mnist, number):
        """
        Load data.
        :param mnist: path of mnist file (npy)
        :param number: select a number of samples. Default is 0 (all)
        :return: loaded dataset
        """
        x = mnist[:, 0].reshape((-1, 1, self.img_size[0], self.img_size[1]))
        y = mnist[:, 1]
        print(f"Loaded x {x.shape}, y {y.shape}")

        if number != 0:
            if x.shape[0] == y.shape[0]:
                total_count = x.shape[0]
                picked = np.random.choice(list(range(total_count)), size=number, replace=False)
                self.seeds = picked
                x = x[picked]
                y = y[picked]
        else:
            print("Lengths not equal!")

        if self.swap_xy:
            return {'x': y, 'y': x}
        else:
            return {'x': x, 'y': y}


class DataSplitter:
    """
    DATASET SPLITTER (LOADER)
    """
    def __init__(self, data, train_ratio=0.8, valid_ratio=0.1, batch_size=64,
                 random=True, shuffle=True, generator=True):
        self.data = data
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.random = random
        self.shuffle = shuffle
        self.generator = generator
        self.train_size, self.valid_size, self.test_size = self.__sizing__()

    def __sizing__(self):
        """
        Calculate train, valid, test sizes.
        :return: int sizes
        """
        train_size = int(len(self.data) * self.train_ratio)
        valid_size = int(len(self.data) * self.valid_ratio)
        test_size = int(len(self.data)) - train_size - valid_size
        return train_size, valid_size, test_size

    def unsplit_loader(self, batch_size=None, shuffle=None):
        """
        Export a loader without splitting.
        :param batch_size: default is 64
        :param shuffle: whether to shuffle samples. Default is True
        :return: data loader
        """
        print(f"Exporting loader of len {len(self.data)}...")
        if not batch_size:
            batch_size = self.batch_size
        if not shuffle:
            shuffle = self.shuffle
        loader = Data.DataLoader(self.data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        return loader

    def split_loader(self, batch_size=None, random=None, shuffle=None, generator=None):
        """
        Split the dataset into train, validation and test.
        :param batch_size: default is 64
        :param random: whether to split the dataset randomly. Default is True
        :param shuffle: whether to shuffle samples. Default is True
        :param generator: random seed generator for random split. Default is None
        :return: train/valid/test dataloaders
        """
        if not batch_size:
            batch_size = self.batch_size
        if not random:
            random = self.random
        if not shuffle:
            shuffle = self.shuffle

        if random:
            train_dataset, valid_dataset, test_dataset = Data.random_split(
                self.data, [self.train_size, self.valid_size, self.test_size], generator=generator)
        else:
            r1 = self.train_size
            r2 = r1 + self.valid_size
            r3 = r2 + self.test_size
            train_dataset = torch.utils.data.Subset(self.data, range(r1))
            valid_dataset = torch.utils.data.Subset(self.data, range(r1, r2))
            test_dataset = torch.utils.data.Subset(self.data, range(r2, r3))

        print(f"Exporting loader len: train {len(train_dataset)}, valid {len(valid_dataset)}, test {len(test_dataset)}")
        train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        valid_loader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=shuffle)

        return train_loader, valid_loader, test_loader
