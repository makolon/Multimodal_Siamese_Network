import torch
import torchvision
from sklearn import svm
from dataset import OmniglotPair, Omniglot, get_loaders, get_loaders_for_multimodal
from model import SiameseSVMNetImg, SiameseSVMNetVib
from svmloss import SVMLoss, compute_accuracy
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
import os

# constants
dim = 105


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese kernel SVM')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--C', type=float, default=0.2, metavar='C',
                        help='C regulation coefficients (default: 0.2)')
    parser.add_argument('--test-number', type=int, default=10, metavar='N',
                        help='number of different subset of test (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main():
    args = get_args()
    model = SiameseSVMNet()
    if args.cuda:
        model = model.cuda()
    criterion = SVMLoss(args.C)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, validate_loader, test_data = get_loaders(args)

    def training(epoch):
        print('Epoch', epoch + 1)
        model.train()
        for batch_idx, (x0, x1, label) in enumerate(train_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            optimizer.zero_grad()
            output = model(x0, x1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("\n Batch:  ", batch_idx, " / ",
                      len(train_loader), " --- Loss: ", loss.data)

    def validate():
        model.eval()
        acc = 0
        for batch_idx, (x0, x1, label) in enumerate(validate_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            output = model(x0, x1)
            acc += compute_accuracy(output, label).numpy()

        acc = 100.0 * acc / len(validate_loader.dataset)
        print('\nValidation set: Accuracy: {}%\n'.format(acc))
        return acc

    def test(n, k):
        model.eval()
        clf = svm.SVC(C=args.C, kernel='linear')
        featuremodel = model.get_FeatureNet()
        if args.cuda:
            featuremodel = featuremodel.cuda()
        # choose classes
        acc = 0
        for i in range(args.test_number):
            random.seed(i)
            temp_ = []
            for i in range(1623 - 1200):
                temp_.append(i)
            random.shuffle(temp_)
            choosen_classes = temp_[:n]

            X_train = []
            y_train = []
            y_test = []
            X_test = []
            for cl in choosen_classes:
                for i in range(k):
                    X_train.append(test_data[cl * 20 + i][0])
                    if args.cuda:
                        X_train[-1] = X_train[-1].cuda()
                    y_train.append(cl)
                for i in range(k, 20):
                    X_test.append(test_data[cl * 20 + i][0])
                    if args.cuda:
                        X_test[-1] = X_test[-1].cuda()
                    y_test.append(cl)

            print('X_train: ', X_train)
            print('y_train: ', y_train)
            print('X_test: ', X_test)
            print('y_test: ', y_test)

            # calculate features
            train_features = []
            test_features = []
            for train_point in X_train:
                train_features.append(featuremodel(
                    Variable(train_point)).cpu().data.numpy())
            for test_point in X_test:
                test_features.append(featuremodel(
                    Variable(test_point)).cpu().data.numpy())

            # create features
            train_features = np.array(train_features)
            train_features = np.reshape(
                train_features, (train_features.shape[0], 4096))
            test_features = np.array(test_features)
            test_features = np.reshape(
                test_features, (test_features.shape[0], 4096))

            # predict with SVM
            clf.fit(train_features, y_train)
            pred = clf.predict(test_features)
            print('prediction: ', pred)
            acc += accuracy_score(y_test, pred)

        acc = 100.0 * acc / args.test_number
        print('\nTest set: {} way {} shot Accuracy: {:.4f}%'.format(n, k, acc))
        return acc

    best_val = 0.0
    test_results = []
    for ep in range(args.epochs):
        training(ep)
        val = validate()
        if val > best_val:
            test_results = []
            test_results.append(test(5, 1))
            test_results.append(test(5, 5))
            test_results.append(test(20, 1))
            test_results.append(test(20, 5))

    # Print best results
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
        test_results[0]))
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
        test_results[1]))
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%'.format(
        test_results[2]))
    print('\nResult: 5 way 1 shot Accuracy: {:.4f}%\n'.format(
        test_results[3]))


def main_for_multi():
    args = get_args()
    model = SiameseSVMNet()
    if args.cuda:
        model = model.cuda()
    criterion = SVMLoss(args.C)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, validate_loader, test_data = get_loaders_for_multimodal(args)

    # target_idx
    target_idx = 0

    def training(epoch):
        print('Epoch', epoch + 1)
        model.train()
        model_path = 'model.pth'
        train_loss = 0
        train_acc = 0
        for batch_idx, (x0, x1, label) in enumerate(train_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            # for debug
            # print('x0.shape: ', x0.shape)
            # print('x1.shape: ', x1.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(x0[0].to('cpu').numpy().reshape(105, 105, 3))
            # plt.show()
            # plt.imshow(x1[0].to('cpu').numpy().reshape(105, 105, 3))
            # plt.show()
            # print('label: ', label[0])
            optimizer.zero_grad()
            output = model(x0, x1)
            loss = criterion(output, label)
            train_loss += loss.item()
            train_acc = (output.max(1)[1] == label).sum()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print("\n Batch:  ", batch_idx, " / ",
                      len(train_loader), " --- Loss: ", loss.data)
                torch.save(model.state_dict(), model_path)
                if epoch % 50 == 0:
                    torch.save(model.state_dict(), str(epoch).zfill(3) + '-' + model_path)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        return avg_train_loss, avg_train_acc

    def validate(cropped_images):
        model_path = 'model.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # choose target
        x1_idx = input('choose target: ')

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
        from PIL import Image
        import matplotlib.pyplot as plt
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
        print(x1.shape)
        plt.imshow(x1[0].reshape(105, 105, 3))
        plt.show()
        print('target!!!')

        # predict
        output_results = {}
        x1 = torch.from_numpy(x1.astype(np.float32)).clone()
        for idx, x0 in enumerate(cropped_images):
            plt.imshow(x0[0].reshape(105, 105, 3))
            plt.show()
            print('index: ', idx)
            if args.cuda:
                x0, x1 = x0.cuda(), x1.cuda()
            x0, x1 = Variable(x0), Variable(x1)
            output = model(x0, x1)
            output_results[idx] = output[0]
        # output = model(x1, x1)
        # output_results[idx+1] = output[0]

        target = max(output_results.values())
        print('target: ', target)
        print('output_resuts: ', output_results)
        target_idx = [k for k, v in output_results.items() if v == target[0][0]]
        print('target_idx: ', target_idx)
        target_img = cropped_images[target_idx[0]]

        return target_img

    # training
    train_loss_list = []
    train_acc_list = []

    """
    for ep in range(args.epochs):
        avg_loss, avg_acc = training(ep)
        train_loss_list.append(avg_loss)
        train_acc_list.append(avg_acc)
    
    # plot learning result
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.plot(train_loss_list,label='adam', lw=3, c='b')
    plt.title('Learning Curve')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.show()
    """

    # temporary: to compare, load cropped images
    from PIL import Image
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
    for i in range(20):
        j = np.random.randint(0, len(cropped_image_path))
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

    # show target cropped image!
    target = validate(cropped_images=cropped_images)
    import matplotlib.pyplot as plt
    plt.imshow(target[0].reshape(105, 105, 3))
    plt.show()
    correct = input('is cropped image correct?: ')
    if correct:
        print('correct')
    else:
        print('not correct')
    """
    def validate():
        model.eval()
        acc = 0
        for batch_idx, (x0, x1, label) in enumerate(validate_loader):
            if args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            output = model(x0, x1)
            acc += compute_accuracy(output, label).numpy()

        acc = 100.0 * acc / len(validate_loader.dataset)
        print('\nValidation set: Accuracy: {}%\n'.format(acc))
        return acc
    """
    """
    def test(cropped_images):
        model_path = 'model.pth'
        model.load_state_dict(torch.load(model_path)) # TODO: fix batch_idx
        model.eval()
        clf = svm.SVC(C=args.C, kernel='linear')
        featuremodel = model.get_FeatureNet()
        if args.cuda:
            featuremodel = featuremodel.cuda()

        # choose classes
        choosen_c = input('target object: ')

        X_train = []
        y_train = []
        X_test = []
        for cl in choosen_classes:
            for i in range(k):
                # X_train.append(test_data[cl * 20 + i][0])
                X_train.append(test_data[cl + i][0])
                if args.cuda:
                    X_train[-1] = X_train[-1].cuda()
                y_train.append(cl)
            for i in range(k, 20):
                # X_test.append(test_data[cl * 20 + i][0])
                X_test.append(test_data[cl + i][0])
                if args.cuda:
                    X_test[-1] = X_test[-1].cuda()

        # calculate features
        train_features = []
        test_features = []
        for train_point in X_train:
            train_features.append(featuremodel(
                Variable(train_point)).cpu().data.numpy())
        for test_point in X_test:
            test_features.append(featuremodel(
                Variable(test_point)).cpu().data.numpy())

        # create features
        train_features = np.array(train_features)
        train_features = np.reshape(
            train_features, (train_features.shape[0], 4096))
        test_features = np.array(test_features)
        test_features = np.reshape(
            test_features, (test_features.shape[0], 4096))

        # predict with SVM
        # clf.fit(train_features, y_train)
        pred = clf.predict(test_features)
        print('prediction: ', pred)
    """

class SiameseNetworkImg():
    def __init__(self, args):
        self.args = args
        self.model = SiameseSVMNetImg()
        if self.args.cuda:
            self.model = self.model.cuda()
        self.criterion = SVMLoss(self.args.C)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_loader, self.validate_loader, self.test_data = get_loaders_for_multimodal(self.args)

    def training(self, epoch):
        print('Epoch', epoch + 1)
        self.model.train()
        model_path = '/home/siamese_network/params_img/model.pth'
        train_loss = 0
        train_acc = 0
        for batch_idx, (x0, x1, label) in enumerate(self.train_loader):
            if self.args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            # for debug
            # print('x0.shape: ', x0.shape)
            # print('x1.shape: ', x1.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(x0[0].to('cpu').numpy().reshape(105, 105, 3))
            # plt.show()
            # plt.imshow(x1[0].to('cpu').numpy().reshape(105, 105, 3))
            # plt.show()
            # print('label: ', label[0])
            self.optimizer.zero_grad()
            output = self.model(x0, x1)
            loss = self.criterion(output, label)
            train_loss += loss.item()
            train_acc = (output.max(1)[1] == label).sum()
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print("\n Batch:  ", batch_idx, " / ",
                      len(self.train_loader), " --- Loss: ", loss.data)
                torch.save(self.model.state_dict(), model_path)
                if epoch % 50 == 0:
                    torch.save(self.model.state_dict(), str(epoch).zfill(3) + '-' + model_path)

        avg_train_loss = train_loss / len(self.train_loader.dataset)
        avg_train_acc = train_acc / len(self.train_loader.dataset)

        return avg_train_loss, avg_train_acc

    def validate(self, cropped_images):
        model_path = '/home/siamese_network/params_img/model.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # choose target
        x1_object_name = input('choose target: ')

        path = '/home/siamese_network/datasets_img'
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

        # load target image
        from PIL import Image
        import matplotlib.pyplot as plt
        object_keys = object_keys = {"aluminum_foil": 1, "kitchen_paper": 2, "cooking_paper": 3,
                "saran_wrap": 4, "noodle_red": 5, "noodle_blue": 6, "wooden_cutting_board": 7,
                "plastic_cutting_board": 8, "cloth": 9,
                "white_bowl": 10, "blue_bowl": 11, "black_bowl": 12, "paper_bowl": 13,
                "white_dish": 14, "black_dish": 15, "paper_dish": 16, "square_dish": 17,
                "frypan": 18, "square_frypan": 19, "pot": 20, "white_sponge": 21,
                "black_sponge": 22, "strainer": 23, "drainer": 24, "bowl": 25,
                "tea": 26, "calpis": 27, "coke": 28, "small_coffee": 29, "large_coffee": 30,
                "mitten": 31, "brown_mug": 32, "red_mug": 33, "white_mug": 34,
                "paper_cup": 35, "metal_cup": 36, "straw_pot_mat": 37, "pot_mat": 38,
                "cork_pot_mat": 39, "brown_cup": 40, "white_cup": 41, "brown_mitten": 42, 
                "gray_dish": 43
        }
        target_idx = object_keys[x1_object_name]
        # print(len(cropped_image_path))
        # print('cropped_image_path: ', cropped_image_path[target_idx])
        target_image_path = os.path.join("/home/siamese_network/datasets_img", x1_object_name)
        target_path = os.path.join(target_image_path, str(target_idx).zfill(4) + "_" + "100.png")
        # img = Image.open(cropped_image_path[target_idx][0])
        # target image index: 100
        img = Image.open(target_path)
        img = img.convert('RGB')
        if transform is not None:
            img = transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 3, img_array.shape[0], img_array.shape[1]))
        
        x1 = img_array
        plt.imshow(x1[0].reshape(105, 105, 3))
        plt.show()
        print('target image!!!')

        # predict
        output_results = {}
        x1 = torch.from_numpy(x1.astype(np.float32)).clone()
        for idx, x0 in enumerate(cropped_images):
            # x0 = torchvision.transforms.functional.to_tensor(x0)
            x0 = torch.from_numpy(x0.astype(np.float32)).clone()
            if self.args.cuda:
                x0, x1 = x0.cuda(), x1.cuda()
            x0, x1 = Variable(x0), Variable(x1)
            output = self.model(x0, x1)
            output_results[idx] = output[0]
            print("idx: ", idx)

        target = max(output_results.values())
        print('target: ', target)
        print('output_resuts: ', output_results)
        target_idx = [k for k, v in output_results.items() if v == target[0][0]]
        print('target_idx: ', target_idx)
        target_img = cropped_images[target_idx[0]]
        plt.imshow(target_img[0].reshape(105, 105, 3))
        plt.show()

        return output_results

class SiameseNetworkVib():
    def __init__(self, args):
        # self.args = get_args()
        self.args = args
        self.model = SiameseSVMNetVib()
        if self.args.cuda:
            self.model = self.model.cuda()
        self.criterion = SVMLoss(self.args.C)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_loader, self.validate_loader, self.test_data = get_loaders_for_multimodal(self.args)

    def training(self, epoch):
        print('Epoch', epoch + 1)
        self.model.train()
        model_path = '/home/siamese_network/params_vib/model.pth'
        train_loss = 0
        train_acc = 0
        for batch_idx, (x0, x1, label) in enumerate(self.train_loader):
            if self.args.cuda:
                x0, x1, label = x0.cuda(), x1.cuda(), label.cuda()
            x0, x1, label = Variable(x0), Variable(x1), Variable(label)
            # for debug
            # print('x0.shape: ', x0.shape)
            # print('x1.shape: ', x1.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(x0[0].to('cpu').numpy().reshape(105, 105, 3))
            # plt.show()
            # plt.imshow(x1[0].to('cpu').numpy().reshape(105, 105, 3))
            # plt.show()
            # print('label: ', label[0])
            self.optimizer.zero_grad()
            output = self.model(x0, x1)
            loss = self.criterion(output, label)
            train_loss += loss.item()
            train_acc = (output.max(1)[1] == label).sum()
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print("\n Batch:  ", batch_idx, " / ",
                      len(self.train_loader), " --- Loss: ", loss.data)
                torch.save(self.model.state_dict(), model_path)
                if epoch % 50 == 0:
                    torch.save(self.model.state_dict(), str(epoch).zfill(3) + '-' + model_path)

        avg_train_loss = train_loss / len(self.train_loader.dataset)
        avg_train_acc = train_acc / len(self.train_loader.dataset)

        return avg_train_loss, avg_train_acc

    def validate(self, cropped_images):
        model_path = '/home/siamese_network/params_vib/model.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # choose target
        x1_object_name = input('choose target: ')

        path = '/home/siamese_network/datasets_vib'
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

        # load target image
        from PIL import Image
        import matplotlib.pyplot as plt
        object_keys = object_keys = {"aluminum_foil": 1, "kitchen_paper": 2, "cooking_paper": 3,
                "saran_wrap": 4, "noodle_red": 5, "noodle_blue": 6, "wooden_cutting_board": 7,
                "plastic_cutting_board": 8, "cloth": 9,
                "white_bowl": 10, "blue_bowl": 11, "black_bowl": 12, "paper_bowl": 13,
                "white_dish": 14, "black_dish": 15, "paper_dish": 16, "square_dish": 17,
                "frypan": 18, "square_frypan": 19, "pot": 20, "white_sponge": 21,
                "black_sponge": 22, "strainer": 23, "drainer": 24, "bowl": 25,
                "tea": 26, "calpis": 27, "coke": 28, "small_coffee": 29, "large_coffee": 30,
                "mitten": 31, "brown_mug": 32, "red_mug": 33, "white_mug": 34,
                "paper_cup": 35, "metal_cup": 36, "straw_pot_mat": 37, "pot_mat": 38,
                "cork_pot_mat": 39, "brown_cup": 40, "white_cup": 41, "brown_mitten": 42, 
                "gray_dish": 43
        }
        target_idx = object_keys[x1_object_name]
        # print(len(cropped_image_path))
        # print('cropped_image_path: ', cropped_image_path[target_idx])
        target_image_path = os.path.join("/home/siamese_network/datasets_vib", x1_object_name)
        target_path = os.path.join(target_image_path, str(target_idx).zfill(4) + "_" + "100.png")
        img = Image.open(target_path)
        img = img.convert('L')
        if transform is not None:
            img = transform(img)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(
            (1, 1, img_array.shape[0], img_array.shape[1]))
        
        x1 = img_array
        plt.imshow(x1[0].reshape(105, 105, 1))
        plt.show()
        print('target image!!!')

        # predict
        output_results = {}
        x1 = torch.from_numpy(x1.astype(np.float32)).clone()
        for idx, x0 in enumerate(cropped_images):
            # x0 = torchvision.transforms.functional.to_tensor(x0)
            x0 = torch.from_numpy(x0.astype(np.float32)).clone()
            if self.args.cuda:
                x0, x1 = x0.cuda(), x1.cuda()
            x0, x1 = Variable(x0), Variable(x1)
            output = self.model(x0, x1)
            output_results[idx] = output[0]
            print("idx: ", idx)

        target = max(output_results.values())
        print('target: ', target)
        print('output_resuts: ', output_results)
        target_idx = [k for k, v in output_results.items() if v == target[0][0]]
        print('target_idx: ', target_idx)
        target_img = cropped_images[target_idx[0]]
        plt.imshow(target_img[0].reshape(105, 105, 1))
        plt.show()

        return output_results

if __name__ == '__main__':
    main_for_multi()
    # main()
