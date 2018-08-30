#
# INF 552 Homework 2
# Part 2: K-Mean Clustering
# Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
# Date: 2/27/2018
#

#
# INF 552 Homework 2
# Part 2: GMM Clustering
# Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
# Date: 2/27/2018
#

from random import *
import settings
import numpy as np
import matplotlib.pyplot as plt

class Table:
    'Common base class for all employees'

    # cood = []

    def __init__(self, *args):
        self.table = np.zeros(shape=(settings.DATA_SIZE, settings.DATA_SIZE));
        self.cood = np.zeros(shape=(settings.DATA_SIZE, settings.DIMENSION));
        # self.table = [0] * settings.DATA_SIZE
        # self.cood = [0] * settings.DATA_SIZE
        self.pivot = [];
        # for i in range(settings.DATA_SIZE):
            # self.table[i] = [0] * settings.DATA_SIZE
            # self.cood[i] = [0] * settings.DIMENSION
        if(args):
            self.readFile(args[0])

    def add(self, line_array):
        print(line_array)
        # self.table[int(line_array[0])][int(line_array[1])] = float(line_array[2])
        # self.table.append(int(line_array))

    def readFile(self, filename):
        with open(filename, "r") as file:
            for line in file:
                line_array = line.split()
                print (line_array)
                self.table[int(line_array[0]) - 1][int(line_array[1]) - 1] = \
                self.table[int(line_array[1]) - 1][int(line_array[0]) - 1] = float(line_array[2])


    # def pickLongestPair(self):
    #     prev = curr = randint(1, settings.DATA_SIZE) - 1
    #     keep_going = 1;
    #     while(keep_going):
    #         next = self.table[curr].index(max(self.table[curr]))
    #         if next == prev :
    #             keep_going = 0;
    #             print [curr, next]
    #             # return [curr, next]
    #         else:
    #             prev = curr;
    #             curr = next;

    def pickLongestPair(self):
        max = np.amax(self.table)
        indices = list(zip(*np.where(self.table == max)))
        print(indices[0])
        self.pivot.append(indices[0])

    def calculateCoordinate(self, dimen):
        a = self.pivot[dimen][0]
        b = self.pivot[dimen][1]
        for i in range(len(self.table)):
            self.cood[i][dimen] = (np.power(self.table[a][i],2) + np.power(self.table[a][b],2) - np.power(self.table[i][b],2))/ (2 * self.table[a][b])
            print (i, self.cood[i][0], self.cood[i][1])

    def updateTable(self, dimen):
        for i in range(0, settings.DATA_SIZE):
            for j in range(0, settings.DATA_SIZE):
                self.table[i][j] = np.sqrt(np.power(self.table[i][j],2) - np.power((self.cood[i][dimen] - self.cood[j][dimen]),2))

    def plotResult(self):
        x = self.cood[:,0]
        y = self.cood[:,1]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        plt.scatter(x, y, label="stars", color="green",
                    marker="*", s=30)
        plt.title("Fast Map Result")

        for i, txt in enumerate(settings.WORDS):
            ax.annotate(txt, (x[i], y[i]))
        plt.show()


    def printTable(self):
        for row in self.table:
            print(row)
