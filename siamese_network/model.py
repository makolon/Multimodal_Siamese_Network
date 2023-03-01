import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNetImg(nn.Module):
    def __init__(self):
        super(FeatureNetImg, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 10)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size = 7)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(128, 128, kernel_size = 4)
        self.dropout3 = nn.Dropout(p=0.1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size = 4)
        self.dropout4 = nn.Dropout(p=0.1)

        self.fc = nn.Linear(9216, 4096)
    

    def forward(self, x):
        block1 = self.dropout1(F.max_pool2d(F.relu(self.conv1(x)),2))
        block2 = self.dropout2(F.max_pool2d(F.relu(self.conv2(block1)),2))
        block3 = self.dropout3(F.max_pool2d(F.relu(self.conv3(block2)),2))
        block4 = self.dropout3(F.relu(self.conv4(block3)))
        flatten = block4.view(-1,9216)
        output = self.fc(flatten)
        return output


class SiameseSVMNetImg(nn.Module):
    def __init__(self):
        super(SiameseSVMNetImg, self).__init__()
        self.featureNet = FeatureNetImg()
        self.fc = nn.Linear(4096, 1)
        self.visualize = True

    def forward(self, x1, x2):
        output1 = self.featureNet(x1)
        output2 = self.featureNet(x2)
        difference = torch.abs(output1 - output2)
        output = self.fc(difference)
        if self.visualize:
            return output, output1, output2
        else:
            return output

    def get_FeatureNet(self):
        return self.featureNet


class FeatureNetVib(nn.Module):
    def __init__(self):
        super(FeatureNetVib, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 10)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size = 7)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv2d(128, 128, kernel_size = 4)
        self.dropout3 = nn.Dropout(p=0.1)

        self.conv4 = nn.Conv2d(128, 256, kernel_size = 4)
        self.dropout4 = nn.Dropout(p=0.1)

        self.fc = nn.Linear(9216, 4096)
    

    def forward(self, x):
        block1 = self.dropout1(F.max_pool2d(F.relu(self.conv1(x)),2))
        block2 = self.dropout2(F.max_pool2d(F.relu(self.conv2(block1)),2))
        block3 = self.dropout3(F.max_pool2d(F.relu(self.conv3(block2)),2))
        block4 = self.dropout3(F.relu(self.conv4(block3)))
        flatten = block4.view(-1,9216)
        output = self.fc(flatten)
        return output


class SiameseSVMNetVib(nn.Module):
    def __init__(self):
        super(SiameseSVMNetVib, self).__init__()
        self.featureNet = FeatureNetVib()
        self.fc = nn.Linear(4096, 1)
        self.visualize = True

    def forward(self, x1, x2):
        output1 = self.featureNet(x1)
        output2 = self.featureNet(x2)
        difference = torch.abs(output1 - output2)
        output = self.fc(difference)
        if self.visualize:
            return output, output1, output2
        else:
            return output

    def get_FeatureNet(self):
        return self.featureNet
        
