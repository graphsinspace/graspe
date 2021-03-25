import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from evaluation.classifications.base.classifier import Classifier


class Net(nn.Module):
    def __init__(self, attributes_cnt, labels_cnt, layers=[200, 100, 50], p=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(attributes_cnt)
        input_size = attributes_cnt
        all_layers = []
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(input_size, labels_cnt))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.layers(x)
        return x


class NeuralNetworkClassification(Classifier):
    def __init__(self, embedding, epochs):
        super().__init__(embedding)
        self._epochs = epochs
        self._labels = [int(l) for l in self._labels]

    def classify(self):

        train_data, test_data, train_labels, test_labels = train_test_split(
            self._data, self._labels, test_size=0.33
        )

        train_data = torch.FloatTensor(train_data)
        test_data = torch.FloatTensor(test_data)
        train_labels = torch.tensor(train_labels, dtype=torch.int64)
        test_labels = torch.tensor(test_labels, dtype=torch.int64)

        # attributes_cnt, labels_cnt, layers
        labels_cnt = len(set(self._labels))
        atts_cnt = train_data.shape[1]

        net = Net(atts_cnt, labels_cnt)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        for i in range(self._epochs):
            train_pred = net(train_data)
            single_loss = loss_function(train_pred, train_labels)

            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

            print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

        print("Finished Training")

        with torch.no_grad():
            predicted = np.argmax(net(test_data), axis=1)

        correct = (predicted == test_labels).sum().item()
        total = len(predicted)
        acc = correct / total

        print("Accuracy", acc)
        print("Precisions: ", precision_score(test_labels, predicted, average=None))
        print("Recalls: ", recall_score(test_labels, predicted, average=None))
