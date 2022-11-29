import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import matplotlib.lines as mlines

from matplotlib import pyplot as plt


def train(input_data, output_file):
    # learning_rate = 1 #alpha
    data = pd.DataFrame(input_data).to_numpy()
    zeros = np.ones((17,1))
    data = np.hstack((zeros,data))
    #print(data)

    # w_old = np.zeros(3)
    # w_current = np.zeros(3)
    # for row in data:
    #     if np.subtract(w_current, w_old) == 0:
    #         break
    #     data_point = row[:3] #ignore label
    #     # y_i * f(x_i) = label * sign(weight vector @ data vector)
    #     sign = row[3] * np.sign(w_current@data_point)
    #     if sign <= 0:
    #         #update weights
    #         w_old = w_current
    #         w_current = w_current + sign

    w_old = np.zeros(3)
    w_current = np.zeros(3)
    convergence = False
    #print(data)
    while not convergence:
        w_old = w_current #change bc first run modifies w_old and w_current
        for row in data:
            data_point = row[:3]  # ignore label
            # y_i * f(x_i) = label * sign(weight vector @ data vector)
            #signed = row[3] * np.sign(w_current @ data_point)
            signed = row[3] * np.sign(np.dot(w_current, data_point))
            #print(np.dot(w_current, data_point))

            if signed <= 0:
                # update weights
                w_old = np.copy(w_current)
                w_current += row[3] * data_point
                #print(w_old, w_current)
                #print(w_current)


        # check_zeros = np.subtract(w_current, w_old)
        # check_zeros = np.array_equal(w_current, w_old)

        if np.array_equal(w_current, w_old): #convergence
            print(w_current, w_old)
            convergence = True

    df = pd.DataFrame(data)
    #print(df)

    #print(w_current)
    w_current = [w_current[1], w_current[2], w_current[0]]
    visualize_scatter(df, feat1=1, feat2=2, labels=0, weights=w_current, title='')



def visualize_scatter(df, feat1=0, feat2=1, labels=2, weights=[-1, -1, 1],
                      title=''):
    """
        Scatter plot feat1 vs feat2.
        Assumes +/- binary labels.
        Plots first and second columns by default.
        Args:
          - df: dataframe with feat1, feat2, and labels
          - feat1: column name of first feature
          - feat2: column name of second feature
          - labels: column name of labels
          - weights: [w1, w2, b]
    """

    # Draw color-coded scatter plot
    #colors = pd.Series(['r' if label > 0 else 'b' for label in df[labels]])
    colors = pd.Series(['r' if label > 0 else 'b' for label in df[3]])
    ax = df.plot(x=feat1, y=feat2, kind='scatter', c=colors)

    # Get scatter plot boundaries to define line boundaries
    xmin, xmax = ax.get_xlim()

    # Compute and draw line. ax + by + c = 0  =>  y = -a/b*x - c/b
    a = weights[0]
    b = weights[1]
    c = weights[2]

    def y(x):
        return (-a/b)*x - c/b

    line_start = (xmin, xmax)
    line_end = (y(xmin), y(xmax))
    line = mlines.Line2D(line_start, line_end, color='red')
    ax.add_line(line)


    if title == '':
        title = 'Scatter of feature %s vs %s' %(str(feat1), str(feat2))
    ax.set_title(title)

    plt.show()



def read_data():
    # read csv file into dataframe
    df = pd.read_csv('/Users/samya/PycharmProjects/perceptron/data1.csv')
    # if second column is 1, plot it blue, otherwise red
    colors = np.where(df.iloc[:, 2] == 1, 'blue', 'red')
    # plot
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=colors)
    plt.show()
    return df


def main():
    '''YOUR CODE GOES HERE'''
    data = read_data()
    train(data, 'hello.txt')


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()
