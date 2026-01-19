'''
На вход подаётся десять точек цен закрытия последовательных периодов
некоторого финансового инструмента
(сайт finam.ru). Создайте три разные полносвязные нейроннные сети,
определяющий направление тренда по трём точкам цен закрытия
последующих периодов.

первые 10 точек - параметры, следующие 3 точки - создают переменную
target (растет или падает стоимость).
вот надо определить растет или падает

в качестве выходной функции можно попробовать
сигмоидную функции активации.
функция потерь - кросс-энтропия

можно сделать разное кол-во слоев - 1, 2 и 3
и в слоях разные функции активации
'''

import torch
import pandas as pd
import numpy as np


class SimpleNetwork(torch.nn.Module):
    '''
    простое линейное преобразование как в предыдущих работах
    1 слой с линейным преобразованием переведенным в отрезок от 0 до 1 (это сделает функция потерь)
    '''
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class TwoLayersNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class ThreeLayersNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def fit(X_train, y_train, net, loss=torch.nn.BCEWithLogitsLoss(), optimizer=None, epoches=1000):
    for epoch in range(epoches):
        optimizer.zero_grad()
        logits = net(torch.tensor(X_train)).squeeze(-1)
        loss_value = loss(logits, torch.tensor(y_train))
        loss_value.backward()
        optimizer.step()


def predict_and_bench(X_test, y_test, net):
    errors = 0
    for i in range(len(X_test)):
        logit = net(torch.tensor(X_test[i])).squeeze(-1).item()
        target = y_test[i]
        pred = 1 if logit > 0 else 0
        if pred != target:
            errors += 1
    return 1 - errors / len(X_test)


def main():
    df = pd.read_csv('data (1).csv', sep=';')
    df = df.iloc[:3000].copy()
    df['CLOSE'] = df['CLOSE']
    df['INDEX'] = range(3000)

    X, y = [], []

    for i in range(len(df) - 13):
        x_frame = df.iloc[i:i+10]
        y_frame = df.iloc[i+10:i+13]
        close = y_frame['CLOSE'].values
        date = y_frame['INDEX'].values
        k = (np.mean(date * close) - np.mean(date) * np.mean(close)) / (np.mean(date ** 2) - np.mean(date) ** 2)
        k = 1 if k > 0 else 0
        X.append(list(x_frame['CLOSE']))
        y.append(k)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std

    split = int(len(df) * 0.8)
    X_train, x_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    net = ThreeLayersNetwork()
    optimizer = torch.optim.SGD(net.parameters(), 0.001)

    fit(X_train, y_train, net, optimizer=optimizer, epoches=100)
    accuracy = predict_and_bench(x_test, y_test, net)
    print(accuracy)


if __name__ == '__main__':
    main()
