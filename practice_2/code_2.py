import pandas as pd
import numpy as np
from random import random

"""
Используя файл vk_perc.csv, сформулируйте задачу для обучения
однослойного персептрона,
выбрав данные первого столбца в качестве решающего правила,
а данные остальных столбцов подайте на вход.
Вычислите метрики оценки качества классификации.

Я сформулировала задачу: определить пользователей,
у которых друзей больше, чем подписчиков, на основе кол-ва фотографий
"""


class Perceptron:
    def __init__(self, n):
        self.col_length = n
        self.w = [(random() - 0.5) * 0.001 for _ in range(n)]  # n признаков
        self.learning_rate = 0.0001

    def _cy(self, y):
        result = 0
        for i in range(len(y)):
            result += self.w[i] * y.iloc[i]
        return 1 if result > 0 else -1

    def fit(self, X, Y, max_epochs=1000):
        epochs = 0
        row_length = len(X)  # кол-во строк
        while epochs < max_epochs:
            epochs += 1
            errors = 0
            for i in range(row_length):
                y = Y.iloc[i]  # объект Series с признаками
                x = X.iloc[i]  # правильный ответ
                y_pred = self._cy(y)
                if y_pred != x:
                    errors += 1
                    for j in range(self.col_length):
                        self.w[j] += self.learning_rate * x * y.iloc[j]
            if epochs % 10 == 0:
                print(f"Эпоха №{epochs}: точность = {1 - (errors / row_length)}")
            if errors == 0:
                print(f"Обучение завершено на эпохе {epochs}")
                print(self.w)
                break


def main():
    df = pd.read_csv("vk_perc.csv", sep=";")
    df["target"] = np.where(df["friends"] > df["followers"], 1, -1)
    X, Y = df["target"], df.drop("target", axis=1)
    p = Perceptron(5)
    p.fit(X, Y)


if __name__ == "__main__":
    main()
