#
# INF 552 Homework 3
# Part 2: Fast Map
# Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
# Date: 2/27/2018
# Programming Language: Python 3.6
#

import numpy as np
import matplotlib.pyplot as plt

DIMENSION = 2
DATA_SIZE = 10

# WORDS = ["acting", "activist", "compute", "coward","forward","interaction","activity","odor","order","international"]
WORDS = []
data_file_name = "fastmap-data.txt"
words_file_name = 'fastmap-wordlist.txt'
table = np.zeros(shape=(DATA_SIZE, DATA_SIZE))
cood = np.zeros(shape=(DATA_SIZE, DIMENSION))
pivot = []

def main():
    readFile(data_file_name)
    print("\nOriginal table:")
    readWords(words_file_name)
    print(WORDS)
    printTable()
    for i in range(DIMENSION):
        print("\n\nThe {i}st cood: ".format(i=i+1))
        pickLongestPair()
        calculateCoordinate(i)
        print("\nUpdate table: ")
        updateTable(i)
        printTable()
    plotResult()

def readFile(filename):
    with open(filename, "r") as file:
        print("Original input:")
        for line in file:
            line_array = line.split()
            print(line_array)
            table[int(line_array[0]) - 1][int(line_array[1]) - 1] = \
                table[int(line_array[1]) - 1][int(line_array[0]) - 1] = float(line_array[2])

def readWords(filename):
    global WORDS
    with open(filename) as file:
        WORDS = file.read().splitlines()

def printTable():
    for row in table:
        print(row)

def pickLongestPair():
    max = np.amax(table)
    indices = list(zip(*np.where(table == max)))
    print("The longest distance pair is {pair}".format(pair = indices[0]))
    print("Pivot is piont {piv}".format(piv = indices[0][0]))
    pivot.append(indices[0])

def calculateCoordinate(dimen):
    a = pivot[dimen][0]
    b = pivot[dimen][1]
    print("The coordinate table")
    for i in range(len(table)):
        cood[i][dimen] = (np.power(table[a][i],2) + np.power(table[a][b],2) - np.power(table[i][b],2))/ (2 * table[a][b])
        print ("{i}\t({x}, {y})".format(i=i, x= round(cood[i][0], 3),y=round(cood[i][1], 3)))

def updateTable(dimen):
    for i in range(0, DATA_SIZE):
        for j in range(0, DATA_SIZE):
            table[i][j] = np.sqrt(np.power(table[i][j],2) - np.power((cood[i][dimen] - cood[j][dimen]),2))

def plotResult():
    x = cood[:, 0]
    y = cood[:, 1]
    fig, ax = plt.subplots()
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.scatter(x, y)
    plt.scatter(x, y, color="red", s=30)
    plt.title("Fast Map Result")

    for i, txt in enumerate(WORDS):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()


main()
