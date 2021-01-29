import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NeuralNetworkClassification():
    def __init__(
        self,
        g,
        embedding,
        epochs
    ):
        self._g = g
        self._embedding = embedding
        self._epochs = epochs


    def classify(self):

        nodes = self._g.nodes()

        labels = [n[1]['label'] for n in nodes]

        embedding_file = open(self._embedding, 'r')
        lines = embedding_file.readlines()

        node_vectors = []
        for line in lines:
            line = line.split(":")[1]
            line = line.split(",")
            
            line = np.array(line, dtype='float64')
            node_vectors.append(line)


        train_data, test_data, train_labels, test_labels = train_test_split(node_vectors, labels, test_size=.33)

        train_data = torch.FloatTensor(train_data)
        test_data = torch.FloatTensor(test_data)
        train_labels = torch.FloatTensor(train_labels)
        test_labels = torch.FloatTensor(test_labels)

        net = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


        for _ in range(self._epochs):

            running_loss = 0.0

            optimizer.zero_grad()

            outputs = net(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        print('Finished Training')

        correct = 0
        total = 0
        with torch.no_grad():
            for t_d, t_l in test_data, test_labels:
                images, labels = t_d, t_l
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        acc = correct / total

        print("Accuracy", acc)
        print("Precisions: ", precision_score(test_labels, correct, average=None))
        print("Recalls: ", recall_score(test_labels, correct, average=None))