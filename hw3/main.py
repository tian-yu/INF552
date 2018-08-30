from Table import Table
# importing the required module
import matplotlib.pyplot as plt
import numpy as np
import settings

test = [[1, 2, 3],
        [2, 3, 4],
        [3, 2, 2]]


input = Table("fastmap-data.txt")
input.printTable()

print ("\n\nThe 1st cood: ")
input.pickLongestPair()
input.calculateCoordinate(0)
print ("\n\nUpdate table: ")
input.updateTable(0)
input.printTable()

print ("\n\nThe 2st cood: ")
input.pickLongestPair()
input.calculateCoordinate(1)
print ("\n\nUpdate table: ")
input.updateTable(1)
input.printTable()

input.plotResult()