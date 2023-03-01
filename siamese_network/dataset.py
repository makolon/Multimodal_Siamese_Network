from torchvision import datasets
import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from torch.utils.data import DataLoader


def get_random_partition(seed=42):
    digit_indices = [x for x in range(1623)]
    random.seed(seed)
    random.shuffle(digit_indices)
    train = digit_indices[:1150]
    validate = digit_indices[1150:1200]
    test = digit_indices[1200:]
    return train, validate, test


def get_loaders(args):
    train, validate, test = get_random_partition(args.seed)
    dset_train = OmniglotPair("data", train)
    if args.cuda:
        train_loader = DataLoader(dset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(dset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=False)

    dset_validate = OmniglotPair("data", validate)
    if args.cuda:
        validate_loader = DataLoader(dset_validate,
                                     batch_size=args.test_batch_size,
                                     shuffle=True,
                                     num_workers=1,
                                     pin_memory=True)
    else:
        validate_loader = DataLoader(dset_validate,
                                     batch_size=args.test_batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=False)

    dset_test = Omniglot("data", test)

    return train_loader, validate_loader, dset_test


class Omniglot(object):
    def __init__(self, path, choosen_classes=None, transform=None):
        self.path = path
        self.data = []
        self.transform = transform
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for directory in folders:
            folders2 = [f for f in os.listdir(
                os.path.join(path, directory)) if not f[0] == '.']
            for subdirectory in folders2:
                if (choosen_classes is None) or class_num in choosen_classes:
                    folders3 = [f for f in os.listdir(os.path.join(
                        path, directory, subdirectory)) if not f[0] == '.']
                    for file in folders3:
                        self.data.append(
                            (os.path.join(path, directory, subdirectory, file), class_num))
                class_num += 1

    def __getitem__(self, index):
        label = self.data[index][0]
        img = Image.open(self.data[index][0])
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 1, img_array.shape[0], img_array.shape[1]))

        return torch.from_numpy(img_array).float(), label

    def __len__(self):
        return len(self.data)


class OmniglotPair(object):
    def __init__(self, path, choosen_classes=None, transform=None):
        random.seed(42)
        self.path = path
        self.data = []
        self.transform = transform
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for directory in folders:
            folders2 = [f for f in os.listdir(
                os.path.join(path, directory)) if not f[0] == '.']
            for subdirectory in folders2:
                # character is in the choosen classes
                if (choosen_classes is None) or class_num in choosen_classes:
                    folders3 = [f for f in os.listdir(os.path.join(
                        path, directory, subdirectory)) if not f[0] == '.']
                    temp_array = []
                    for file in folders3:
                        temp_array.append(os.path.join(
                            path, directory, subdirectory, file))
                    self.data.append(temp_array)
                class_num += 1

        self.sub_index = []
        for i in range(19):
            for j in range(i, 20):
                self.sub_index.append((i, j))

    def __getitem__(self, index):
        class_num = index // 380
        img_number = index % 380
        if (img_number % 2 == 0):
            img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
            img2 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][1]])
            return img1, img2, torch.from_numpy(np.array([1])).float()
        else:
            img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
            inc = random.randrange(1, len(self.data))
            dn = (class_num + inc) % len(self.data)
            di = random.randrange(0, len(self.data[dn]))
            img2 = self.load_img(self.data[dn][di])
            return img1, img2, torch.from_numpy(np.array([-1])).float()

    def load_img(self, path):
        img = Image.open(path)
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, img_array.shape[0], img_array.shape[1]))
        return torch.from_numpy(img_array).float()

    def __len__(self):
        return 20 * 19 * len(self.data)


def get_random_partition_for_multi(seed=42):
    object_indices = [x for x in range(33)]
    random.seed(seed)
    random.shuffle(object_indices)
    # train = object_indices[:25]
    # validate = object_indices[25:28]
    # test = object_indices[28:]
    train = object_indices[:33]
    validate = object_indices[:33]
    test = object_indices[33:]
    return train, validate, test


def get_loaders_for_multimodal(args):
    train, validate, test = get_random_partition_for_multi(args.seed)
    print('train: ', train)
    dset_train = ImageDatasetPair("/home/multimodal_object_identification/siamese_network/datasets", train)
    if args.cuda:
        train_loader = DataLoader(dset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True)
    else:
        train_loader = DataLoader(dset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=False)

    dset_validate = ImageDatasetPair("/home/multimodal_object_identification/siamese_network/datasets", validate)
    if args.cuda:
        validate_loader = DataLoader(dset_validate,
                                     batch_size=args.test_batch_size,
                                     shuffle=True,
                                     num_workers=1,
                                     pin_memory=True)
    else:
        validate_loader = DataLoader(dset_validate,
                                     batch_size=args.test_batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=False)

    dset_test = ImageDataset("/home/multimodal_object_identification/siamese_network/datasets", test)

    return train_loader, validate_loader, dset_test


class ImageDataset(object):
    def __init__(self, path, choosen_classes=None, transform=None):
        self.path = path
        self.data = []
        self.transform = transform
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for subdirectory in folders:
            if (choosen_classes is None) or class_num in choosen_classes:
                folders2 = [f for f in os.listdir(os.path.join(
                    path, subdirectory)) if not f[0] == '.']
                for file in folders2:
                    self.data.append(
                        (os.path.join(path, subdirectory, file), class_num))
                class_num += 1

    def __getitem__(self, index):
        label = self.data[index][0] # label is object name
        img = Image.open(self.data[index][0])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 3, img_array.shape[0], img_array.shape[1]))

        return torch.from_numpy(img_array).float(), label

    def __len__(self):
        return len(self.data)


class ImageDatasetPair(object):
    def __init__(self, path, choosen_classes=None, transform=None):
        random.seed(42)
        self.path = path
        self.data = []
        self.transform = transform
        self.choosen_classes = choosen_classes
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for subdirectory in folders:
            # character is in the choosen classes
            if (choosen_classes is None) or class_num in choosen_classes:
                folders2 = [f for f in os.listdir(os.path.join(
                    path, subdirectory)) if not f[0] == '.']
                temp_array = []
                for file in folders2:
                    temp_array.append(os.path.join(
                        path, subdirectory, file))
                self.data.append(temp_array)
            class_num += 1

        self.sub_index = []
        for i in range(19): # TODO: check the number of image
            for j in range(i, 20): # the number of pictures
                self.sub_index.append((i, j))

    def __getitem__(self, index):
        class_num = np.random.randint(0, len(self.choosen_classes))
        img_number = index % (2 * len(self.choosen_classes))
        if (img_number % 2 == 0):
            img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
            img2 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][1]])
            return img1, img2, torch.from_numpy(np.array([1])).float()
        else:
            img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
            inc = random.randrange(1, len(self.data))
            dn = (img_number + inc) % len(self.data)
            di = random.randrange(0, len(self.data[dn]))
            img2 = self.load_img(self.data[dn][di])
            # import matplotlib.pyplot as plt
            # plt.imshow(img1[0], cmap='gray')
            # plt.show()
            # plt.imshow(img2[0], cmap='gray')
            # plt.show()
            return img1, img2, torch.from_numpy(np.array([-1])).float()

    def load_img(self, path):
        img = Image.open(path)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (3, img_array.shape[0], img_array.shape[1]))
        return torch.from_numpy(img_array).float()

    def __len__(self):
        return 20 * 19 * len(self.data)


class VibImageDataset(object):
    def __init__(self, path, choosen_classes=None, transform=None):
        self.path = path
        self.data = []
        self.transform = transform
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for subdirectory in folders:
            if (choosen_classes is None) or class_num in choosen_classes:
                folders2 = [f for f in os.listdir(os.path.join(
                    path, subdirectory)) if not f[0] == '.']
                for file in folders2:
                    self.data.append(
                        (os.path.join(path, subdirectory, file), class_num))
                class_num += 1

    def __getitem__(self, index):
        label = self.data[index][0] # label is object name
        img = Image.open(self.data[index][0])
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 1, img_array.shape[0], img_array.shape[1]))

        return torch.from_numpy(img_array).float(), label

    def __len__(self):
        return len(self.data)


class VibImageDatasetPair(object):
    def __init__(self, path, choosen_classes=None, transform=None):
        random.seed(42)
        self.path = path
        self.data = []
        self.transform = transform
        self.choosen_classes = choosen_classes
        class_num = 0
        folders = [f for f in os.listdir(path) if not f[0] == '.']
        for subdirectory in folders:
            # character is in the choosen classes
            if (choosen_classes is None) or class_num in choosen_classes:
                folders2 = [f for f in os.listdir(os.path.join(
                    path, subdirectory)) if not f[0] == '.']
                temp_array = []
                for file in folders2:
                    temp_array.append(os.path.join(
                        path, subdirectory, file))
                self.data.append(temp_array)
            class_num += 1

        self.sub_index = []
        for i in range(19): # TODO: check the number of image
            for j in range(i, 20): # the number of pictures
                self.sub_index.append((i, j))

    def __getitem__(self, index):
        class_num = np.random.randint(0, len(self.choosen_classes))
        img_number = index % (2 * len(self.choosen_classes))
        if (img_number % 2 == 0):
            img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
            img2 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][1]])
            return img1, img2, torch.from_numpy(np.array([1])).float()
        else:
            img1 = self.load_img(
                self.data[class_num][self.sub_index[img_number // 2][0]])
            inc = random.randrange(1, len(self.data))
            dn = (img_number + inc) % len(self.data)
            di = random.randrange(0, len(self.data[dn]))
            img2 = self.load_img(self.data[dn][di])
            # import matplotlib.pyplot as plt
            # plt.imshow(img1[0], cmap='gray')
            # plt.show()
            # plt.imshow(img2[0], cmap='gray')
            # plt.show()
            return img1, img2, torch.from_numpy(np.array([-1])).float()

    def load_img(self, path):
        img = Image.open(path)
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, img_array.shape[0], img_array.shape[1]))
        return torch.from_numpy(img_array).float()