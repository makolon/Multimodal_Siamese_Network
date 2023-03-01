from matplotlib import markers
from numpy.lib.function_base import sinc
import torch
from sklearn import svm
from dataset import OmniglotPair, Omniglot, get_loaders, get_loaders_for_multimodal
from model import SiameseSVMNet
from svmloss import SVMLoss, compute_accuracy
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import bhtsne
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from train import get_args
import scipy as sp


class TestSiameseNetwork():
    def __init__(self):
        self.args = get_args()
        self.model = SiameseSVMNet()
        if self.args.cuda:
            self.model = self.model.cuda()

    def test(self, cropped_images, labels):
        model_path = 'model.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        path = 'datasets'
        cropped_image_path = []
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        class_num = 0
        choosen_classes=None
        transform = None
        for subdirectory in folders:
            if (choosen_classes is None) or class_num in choosen_classes:
                folders2 = [f for f in os.listdir(os.path.join(
                    path, subdirectory)) if not f[0] == '.']
                for file in folders2:
                    cropped_image_path.append(
                        (os.path.join(path, subdirectory, file), class_num))

        # temporary: load target image from dataset

        i = np.random.randint(0, len(cropped_image_path))
        print('cropped_image_path: ', cropped_image_path[i])
        img = Image.open(cropped_image_path[i][0])
        img = img.convert('RGB')
        if transform is not None:
            img = transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 3, img_array.shape[0], img_array.shape[1]))
        x1 = img_array
        
        # predict
        output_results = {}
        x1 = torch.from_numpy(x1.astype(np.float32)).clone()
        embed_features = np.empty((1, 4096))
        for idx, x0 in enumerate(cropped_images):
            if self.args.cuda:
                x0, x1 = x0.cuda(), x1.cuda()
            x0, x1 = Variable(x0), Variable(x1)
            output, output1, output2 = self.model(x0, x1)
            output1 = output1.to('cpu').detach().numpy()
            embed_features = np.concatenate([embed_features, output1], axis=0)
            output_results[idx] = output
        
        embed_features = embed_features[1:]
        print('embed_feature.shape: ', embed_features.shape)
        self.visualize(embed_features, labels)

    def visualize(self, embed_feature, labels):
        data_tsne = bhtsne.tsne(embed_feature.astype(sp.float64),
            dimensions=2, 
            perplexity=30,
            theta=0.1,
            rand_seed=1,
        )

        xmin = data_tsne[:,0].min() -10
        xmax = data_tsne[:,0].max() + 10
        ymin = data_tsne[:,1].min() - 10
        ymax = data_tsne[:,1].max() + 10

        plt.figure(figsize=(16,12))
        import matplotlib
        color_list = matplotlib.colors.cnames
        colors = {}
        for idx, v in enumerate(color_list):
            colors[idx] = v
        for data, label in zip(data_tsne, labels):
            # plt.plot(data[0], data[1], color=colors[int(label)], 
            #     marker='o', markersize=15, alpha=0.3)
            plt.annotate(int(label), (data[0], data[1]))
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel("component 0")
        plt.ylabel("component 1")
        plt.title("t-SNE visualization")
        plt.savefig("tsne.png")

if __name__ == '__main__':
    siamese = TestSiameseNetwork()

    path = 'datasets'
    cropped_image_path = []
    folders = [f for f in os.listdir(path) if not f[0] == '.']
    class_num = 0
    choosen_classes=None
    transform = None
    for subdirectory in folders:
        if (choosen_classes is None) or class_num in choosen_classes:
            folders2 = [f for f in os.listdir(os.path.join(
                path, subdirectory)) if not f[0] == '.']
            for file in folders2:
                cropped_image_path.append(
                    (os.path.join(path, subdirectory, file), class_num))
    
    cropped_images = []
    labels = []
    for i in range(300):
        j = np.random.randint(0, len(cropped_image_path))
        label = cropped_image_path[j][0][-9:-7]
        if label[1] == '_':
            label = '0' + label[0]
        labels.append(label)
        img = Image.open(cropped_image_path[j][0])
        img = img.convert('RGB')
        if transform is not None:
            img = transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 3, img_array.shape[0], img_array.shape[1]))
        img_array = torch.from_numpy(img_array.astype(np.float32)).clone()
        cropped_images.append(img_array)
    siamese.test(cropped_images, labels)

