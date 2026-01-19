"""
Из файла med.xlsx рассмотрите в качестве целевых переменных показатели,
указанные в вашем варианте (Вариант 2.4.BMI и MDRD).

1.Удалите некорректные записи. Предусмотрите следующие методы:
а) удаление пустых ячеек из заданного множества массивов данных
с сохранением первоначального соответствия;
б) удаление некорректных значений из заданного множества массивов данных
с сохранением первоначального соответствия;

Коридоры значений
Систолическое АД:      90 – 230 мм рт.ст.
Диастолическое АД:   40 – 120 мм рт.ст.
SCORE:                             0 – 50 %
ИМТ (BMI):                                15 – 45
Глюкоза:                          3 – 25 ммоль/л
Холестерин:                    2 – 15 ммоль/л
СКФ (MDRD):                                  140 – 15

в) Отображение нечисловых значений в фиктивные переменные
с числовыми значениями.

2.Вычислите выборочные характеристики: среднюю дисперсию,
среднеквадратическое отклонение (std), моду (top), медиану, парный коэффициент корреляции.

3.Используя персептроны, разбейте данные на два класса.
Вычислите, какой процент городского и
сельского населения находится в каждом классе.

4.Для прогнозирования значения цен таргетных переменнх
предложите модели множественной регрессии. Вычислите коэффициенты детерминации.

5.Продемонстрируйте умение применять тесты на проверку
выполнения предпосылок МНК.

6.Постройте гистограммы распределения значений таргетных переменных.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.metrics import accuracy_score, r2_score


main_list = [
    "Systolic Blood Pressure",
    "Diastolic Blood Pressure",
    "Risk Score CVRM",
    "BMI",
    "Glucose Fasting",
    "Total Cholesterol",
    "MDRD",
    "Age",
    "Smoking Status_numeric",
    "Patient Gender_numeric",
    "Hypertension_numeric",
]


def typing(df):
    df["Smoking Status_numeric"] = np.where(df["Smoking Status"] == "Never", 0, 1)
    df["Patient Gender_numeric"] = np.where(df["Patient Gender"] == "F", 0, 1)
    df["Hypertension_numeric"] = np.where(df["Hypertension"] == "Yes", 1, 0)


def correcting(df):
    correct_vals = {
        "Systolic Blood Pressure": (90, 230),
        "Diastolic Blood Pressure": (40, 120),
        "Risk Score CVRM": (0, 50),
        "BMI": (15, 45),
        "Glucose Fasting": (3, 25),
        "Total Cholesterol": (2, 15),
        "MDRD": (15, 140),
    }

    for key, (low, high) in correct_vals.items():
        df = df[(df[key] >= low) & (df[key] <= high)]

    return df


def profiling(df, file, main_list=None):
    describe = df.describe(include="all")
    with open(file, "a", encoding="UTF-8") as file:
        if main_list:
            print(describe[main_list], file=file)
        else:
            print(describe, file=file)
        df.info(buf=file)
        print("==============================================\n\n")


def median(df, file):
    cols = df.columns
    with open(file, "a", encoding="UTF-8") as file:
        for key in cols:
            print(f"Медиана {key}: {df[key].median()}", file=file)


def couple_corr(df, file):
    cols = df.columns
    with open(file, "a", encoding="UTF-8") as file:
        print(df[cols].corr(method="pearson"), file=file)


def perc_class(df, file):
    cols = list(df.columns)
    with open(file, "a", encoding="UTF-8") as file:
        y = pd.Series(data=np.where(df["BMI"] >= 30, 1, 0), name="BMI")
        print(f"реальное число толстых {y.sum()}")
        cols.remove("BMI")
        X = df[cols]
        perceptron = Perceptron(max_iter=10000, random_state=42, eta0=0.1, tol=None)
        perceptron.fit(X, y)
        predicted = perceptron.predict(X)
        print(f"Точность разделения данных: {accuracy_score(y, predicted)}", file=file)
        return predicted


def dropna(df):
    df = df[
        ~(
            df["Risk Score CVRM"].isna()
            | df["MDRD"].isna()
            | df["Glucose Fasting"].isna()
            | df["Smoking Status"].isna()
        )
    ]
    return df


def mult_reg(df, file):
    cols = list(df.columns)
    with open(file, "a", encoding="UTF-8") as file:
        y = pd.Series(data=np.where(df["BMI"] >= 30, 1, 0), name="BMI")
        cols.remove("BMI")
        X = df[cols]
        lr = LinearRegression()
        lr.fit(X, y)
        predicted = lr.predict(X)
        r2 = r2_score(y, predicted)
        print(f"Коэффициент детерминации для множественной регрессии: {r2}", file=file)


def main():
    pd.set_option("display.max_columns", None)  # показывать все столбцы
    pd.set_option("display.width", None)  # не ограничивать ширину строки
    pd.set_option("display.max_colwidth", None)  # не обрезать содержимое ячеек

    file = "profile.txt"
    with open(file, "w"):
        pass

    df = pd.read_csv("med.csv", sep=";", low_memory=False)
    profiling(df, file)

    df = dropna(df)
    typing(df)
    df = correcting(df)

    df_full = df.copy()
    df = df[main_list]

    profiling(df, file)

    median(df, file)
    couple_corr(df, file)

    df["Predicted"] = perc_class(df, file)

    # село или город в предсказанных классах
    fat = df["Predicted"].sum()
    print(fat)
    fat_city = np.where(
        (df["Predicted"] == 1) & (df_full["Organisation Name (CVRM Treatment)"] == 1),
        1,
        0,
    ).sum()
    print(f"Среди людей с ИМТ >= 30 гордских жителей {fat_city / fat * 100}%")

    row_count = len(df)
    slim = row_count - fat
    slim_city = np.where(
        (df["Predicted"] == 0) & (df_full["Organisation Name (CVRM Treatment)"] == 1),
        1,
        0,
    ).sum()
    print(f"Среди людей с ИМТ < 30 городских жителей {slim_city / slim * 100}%")

    mult_reg(df, file)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(df["MDRD"], bins=30)
    ax1.set_title("MDRD")
    ax2.hist(df["BMI"], bins=30)
    ax2.set_title("BMI")
    plt.show()


if __name__ == "__main__":
    main()
