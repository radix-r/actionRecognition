import torch

__author__ = "Ross Wagner"

import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------

        x = self.layer1(x)

        x = torch.sigmoid(self.output_layer(x))

        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, 40, 5, 1)
        self.conv2 = nn.Conv2d(40, 40, 5, 1)
        self.layer1 = nn.Linear(4, hidden_size)

        # self.conv2 =
        self.output_layer = nn.Linear(40*40*10, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------



        # first convolution
        x = self.conv1(x)
        sig = F.sigmoid(x)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(sig, (2, 2))

        # second convolution
        x = self.conv2(x)
        sig = F.sigmoid(x)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(sig, (2, 2))

        # fully connected layer
        x.view(x.size(0), -1)
        # x = torch.reshape(x, (-1, self.input_dim, 1, 1))
        x = self.layer1(x)

        # finalize output
        x = x.view(-1, self.num_flat_features(x))

        x = torch.sigmoid(self.output_layer(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------


        self.filter = torch.ones(1,5,5)
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, 40, 5, 1)
        self.conv2 = nn.Conv2d(40, 40, 5, 1)
        self.layer1 = nn.Linear(40*4*4, hidden_size)
        self.rel = nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv2 =
        self.output_layer = nn.Linear(hidden_size, 10)



    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # first convolution
        x = self.conv1(x)
        x = F.relu(x)
        # Max pooling over a (2, 2) window
        x = self.pool(x)

        # second convolution
        x = self.conv2(x)
        x = F.relu(x)
        # Max pooling over a (2, 2) window
        x = self.pool(x)

        # fully connected layer
        x = torch.reshape(x, (-1, 40*4*4))
        x = self.layer1(x)
        x = F.relu(x)

        # finalize output
        x = torch.sigmoid(self.output_layer(x))

        return x

class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------

        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, 40, 5, 1)
        self.conv2 = nn.Conv2d(40, 40, 5, 1)
        self.layer1 = nn.Linear(40 * 4 * 4, hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------

        # first convolution
        x = self.conv1(x)
        x = F.relu(x)
        # Max pooling over a (2, 2) window
        x = self.pool(x)

        # second convolution
        x = self.conv2(x)
        x = F.relu(x)
        # Max pooling over a (2, 2) window
        x = self.pool(x)

        # fully connected layer
        x = torch.reshape(x, (-1, 40 * 4 * 4))
        x = self.layer1(x)
        x = F.relu(x)

        # Second connected layer
        x = self.layer2(x)
        x = F.relu(x)

        # finalize output
        x = torch.sigmoid(self.output_layer(x))

        return x

class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(1, 40, 5, 1)
        self.conv2 = nn.Conv2d(40, 40, 5, 1)
        self.layer1 = nn.Linear(40 * 4 * 4, hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.output_layer = nn.Linear(hidden_size, 10)
        self.dropOut1 = nn.Dropout(p=0.5)
        self.dropOut2 = nn.Dropout(p=0.1)
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features   #
        #
        # ----------------- YOUR CODE HERE ----------------------

        # first convolution
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropOut2(x)
        # Max pooling over a (2, 2) window
        x = self.pool(x)

        # second convolution
        x = self.conv2(x)
        x = F.relu(x)
        x=self.dropOut2(x)
        # Max pooling over a (2, 2) window
        x = self.pool(x)

        # fully connected layer
        x = torch.reshape(x, (-1, 40 * 4 * 4))
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropOut1(x)

        # Second connected layer
        x = self.layer2(x)
        x = F.relu(x)

        # finalize output
        x = torch.sigmoid(self.output_layer(x))

        return x



class Net(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.hidden_size= args.hidden_size
        # model 1: base line
        if mode == 1:
            in_dim = 28*28 # input image size is 28x28
            self.model = Model_1(in_dim, self.hidden_size)

        # model 2: use two convolutional layer
        if mode == 2:
            # in_dim = 28 * 28  # input image size is 28x28
            self.model = Model_2(self.hidden_size)

        # model 3: replace sigmoid with relu
        if mode == 3:
            self.model = Model_3(self.hidden_size)

        # model 4: add one extra fully connected layer
        if mode == 4:
            self.model = Model_4(self.hidden_size)

        # model 5: utilize dropout
        if mode == 5:
            self.model = Model_5(self.hidden_size)


    def forward(self, x):
        if self.mode == 1:
            x = x.view(-1, 28* 28)
            x = self.model(x)
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)
        # ======================================================================
        # Define softmax layer, use the features.
        # ----------------- YOUR CODE HERE ----------------------

        softL = nn.LogSoftmax()
        # softF = ()

        logits = F.softmax(x)

        # Remove NotImplementedError and assign calculated value to logits after code implementation.
        #logits = softL(x)
        return logits

