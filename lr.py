import numpy as np
import pandas as pd
from scipy import stats
import sys


def data_prep():
    df = pd.read_csv('/Users/samya/PycharmProjects/perceptron/data2.csv', header=None)
    # age = df.iloc[:, 0]
    # weight = df.iloc[:, 1]
    # height= df.iloc[:, 2]

    # standardize data
    for i in range(2):
        mean = df.iloc[:, i].mean()
        std = df.iloc[:, i].std()
        df.iloc[:, i] = df.iloc[:, i].apply(lambda x: (x - mean) / std)
    # print(df)

    # turn to np array and add intercepts to first column
    data = pd.DataFrame(df).to_numpy()
    # print(data)
    ones = np.ones((79, 1))
    data = np.hstack((ones, data))
    # print(data)
    return data


# def lr(data, ):
#     alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
#     for alpha in alphas:
#         for i in range(100):
#             for row in data:
#                 columns = data[:3]
#                 #find f(x)
#                 total = 0
#                 for j in range(i):
#                     j += 1
#                     total += columns[0] + columns[1] * columns - row[3]
#                     beta = beta - ((1/j) * alpha) * total * columns
#         print(beta)

def lr(data):
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    for alpha in alphas:
        betas = [0, 0, 0]
        for i in range(100):
            x_1, x_2, x_3 = 0, 0, 0
            for row in data:
                columns = row[:3]
                # f_x = (betas @ columns) - data[3]
                f_x = np.dot(betas, columns) - row[3]
                # print(f_x)
                # print(betas)
                x_1 += f_x * columns[0]
                x_2 += f_x * columns[1]
                x_3 += f_x * columns[2]
            betas[0] = betas[0] - alpha * (1 / 79) * x_1  # TODO: fix here
            # print(betas[0])
            betas[1] = betas[1] - alpha * (1 / 79) * x_2
            betas[2] = betas[2] - alpha * (1 / 79) * x_3

        print(betas)

    # x_i = element in row
    # beta + dot product


def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    data = data_prep()
    lr(data)


if __name__ == "__main__":
    main()
