import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D


def data_prep(input):
    df = pd.read_csv(input, header=None)
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

def lr(data, file):
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.2]
    output = []
    for alpha in alphas:
        betas = [0, 0, 0]
        for i in range(50):
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


        # print(betas)
        #format(betas)
        output.append(betas)

        #visualize_3d(data, betas, feat1=1, feat2=2, labels=3, title='LinReg Height with Alpha' + alpha)
        # visualize_3d(data, betas, feat1=1, feat2=2, labels=3, xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3), alpha=alpha,
        #              xlabel='age', ylabel='weight', zlabel='height', title='')
    format(output, file)


    # x_i = element in row
    # beta + dot product

def format(output, file):
    alphas = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10])
    betas = np.array(output)
    line = []
    outline = np.array([])
    write_list = np.array([])

    iterations = []
    for i in range(len(alphas)):
        iterations.append(100)
    iterations = np.array(iterations)

    f = open(file, 'w')

    for i in range(len(alphas)):
        beta1 = betas[i][0]
        beta2 = betas[i][1]
        beta3 = betas[i][2]
        line = [alphas[i], iterations[i], beta1, beta2, beta3]
        print(line)
        line = ', '.join(str(e) for e in line)
        f.write(line)
        f.write('\n')



        #line = np.array(line)
        #print(line)

    f.close()
    # visualize_3d(output, betas, feat1=1, feat2=2, labels=3)
   # print(write_list)

    #print(write_list)
    # df = pd.DataFrame(write_list)

    # write_output(df, file)

    # # print(alphas)
    # # print(iterations)
    #
    # print(np.hstack((alphas, iterations)))

def write_output(df, file):
    df.to_csv(file, header=False, index=False)


def visualize_3d(df, lin_reg_weights, feat1=1, feat2=2, labels=3,
                 xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 3),
                 alpha=0., xlabel='age', ylabel='weight', zlabel='height',
                 title=''):
    """
    3D surface plot.
    Main args:
      - df: dataframe with feat1, feat2, and labels
      - feat1: int/string column name of first feature
      - feat2: int/string column name of second feature
      - labels: int/string column name of labels
      - lin_reg_weights: [b_0, b_1 , b_2] list of float weights in order
    Optional args:
      - x,y,zlim: axes boundaries. Default to -1 to 1 normalized feature values.
      - alpha: step size of this model, for title only
      - x,y,z labels: for display only
      - title: title of plot
    """

    # Setup 3D figure
    ax = plt.figure().gca(projection='3d')
    #plt.hold(True)

    # Add scatter plot
    ax.scatter(df[feat1], df[feat2], df[labels])
    print(lin_reg_weights[0])
    print("\n")
    print(lin_reg_weights[1])

    # Set axes spacings for age, weight, height
    axes1 = np.arange(xlim[0], xlim[1], step=.05)  # age
    axes2 = np.arange(xlim[0], ylim[1], step=.05)  # weight
    axes1, axes2 = np.meshgrid(axes1, axes2)
    axes3 = np.array( [lin_reg_weights[0] +
                       lin_reg_weights[1]*f1 +
                       lin_reg_weights[2]*f2  # height
                       for f1, f2 in zip(axes1, axes2)] )
    plane = ax.plot_surface(axes1, axes2, axes3, cmap=cm.Spectral,
                            antialiased=False, rstride=1, cstride=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    if title == '':
        title = 'LinReg Height with Alpha %f' % alpha
    ax.set_title(title)

    plt.show()

def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    input = sys.argv[1]
    output = sys.argv[2]
    data = data_prep(input)
    lr(data, output)


if __name__ == "__main__":
    main()
