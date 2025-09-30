import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from random import random


class Perceptron:
    def __init__(self):
        self.w = [(random() - 0.5) * 0.001 for _ in range(2)]  # три веса - для x, у и bias
        self.learning_rate = 0.000000001

    def _cy(self, x, y):
        return 1 if (x * self.w[0] + y * self.w[1]) > 0 else -1

    def fit(self, X_train, y_train, max_epochs=1000):
        epoch = 0
        while epoch < max_epochs:
            epoch += 1
            errors = 0
            for i in range(len(X_train)):
                y_pred = self._cy(X_train.iloc[i]["INDEX"], X_train.iloc[i]["CLOSE"])
                if y_pred * y_train.iloc[i] < 0:
                    errors += 1
                    # error = y_train.iloc[i] - y_pred
                    self.w[0] += self.learning_rate * y_train.iloc[i] * X_train.iloc[i]["INDEX"]
                    self.w[1] += self.learning_rate * y_train.iloc[i] * X_train.iloc[i]["CLOSE"]
            if errors / len(X_train) < 0.2:
                print(f"Для обучения модели понадобилось {epoch} циклов")
                print(f"Веса:\n{self.w}")
                break
            if epoch % 10 == 0:  # выводим каждые 10 эпох
                print(f"Эпоха {epoch}: ошибок = {errors}, точность = {1 - errors/len(X_train):.2%}")

    def predict(self, X_test):
        return [
            self._cy(X_test.iloc[i]["INDEX"], X_test.iloc[i]["CLOSE"]) for i in range(len(X_test))
        ]


df = pd.read_csv("data (1).csv", sep=";")
df["INDEX"] = [i for i in range(1, len(df) + 1)]

# строю линию тренда
x = df["INDEX"].values
y = df["CLOSE"].values
mean_x, mean_y = np.mean(x), np.mean(y)
b = (np.mean(x * y) - mean_x * mean_y) / (np.mean(x**2) - np.mean(x) ** 2)
a = mean_y - b * mean_x

# plt.scatter(df["INDEX"], df["CLOSE"], alpha=0.5)
# plt.plot([1, len(df)], [a + b, a + b * len(df)], color="red", linestyle="--")
# plt.xlabel("Порядковый номер закрытия")
# plt.ylabel("Цена закрытия")
# plt.xlim(1, len(df))
# plt.show()

# размечаю данные
df["EXPVAL"] = a + b * df["INDEX"]
df["TARGET"] = np.where(df["CLOSE"] > df["EXPVAL"], 1, -1)

X = df[["INDEX", "CLOSE"]]
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

p = Perceptron()
p.fit(X_train, y_train)
result = p.predict(X_test)

same = 0
for i in range(len(result)):
    if result[i] == y_test[i]:
        same += 1
print(f"Совпало {same / len(X_test) * 100:.2f}% ответов")
