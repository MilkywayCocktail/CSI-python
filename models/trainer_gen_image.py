import training_presets
import torch
import matplotlib as plt
import numpy as np


class TrainerGenImage(training_presets.Trainer):

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
            self.test_loss.append(loss.item())
            if idx % (len(self.test_loader)//5) == 0:
                print("\r{}/{}of test, loss={}".format(idx, len(self.test_loader), loss.item()), end='')

    def plot_test_results(self):

        imgs = np.random.randint(len(self.groundtruth), 8)
        plt.figure()
        f, ax = plt.subplots(1, 8)

        for a in range(len(ax)):

            ax[a].plot(self.groundtruth[imgs[a]])
            ax[a].set_xlabel(str(imgs[a]))

        plt.suptitle('Ground Truth')
        plt.show()
        
        plt.figure()
        f, ax = plt.subplots(1, 8)

        for a in range(len(ax)):

            ax[a].plot(self.estimates[imgs[a]])
            ax[a].set_xlabel(str(imgs[a]))

        plt.suptitle('Estimated')
        plt.show()