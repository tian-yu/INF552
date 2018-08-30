import numpy as np
from numpy import linalg as LA

INPUT_FILE_NAME = "hmm-data.txt"
NUM_LINES_BEFORE_GRID = 2
NUM_LINES_BEFORE_TOWER = 4
NUM_LINES_BEFORE_STEPS = 4
GRID_WORLD_SIZE = 10
TOWER_NUM = 4
NUM_OF_STEPS = 11
TOWER_UPPER_BOUND = 1.3
TOWER_LOWER_BOUND = 0.7
NUM_OF_STATES = 10**2
a = 0.1


def readFile(file_name):
    grid_world = []
    tower_location = []
    tower_observation = []

    with open(file_name) as file:
        # Read grid world
        for _ in range(NUM_LINES_BEFORE_GRID):
            file.readline()
        for _ in range(GRID_WORLD_SIZE):
            grid_world.append(file.readline().split())

        # Read tower location
        for _ in range(NUM_LINES_BEFORE_TOWER):
            file.readline()
        for _ in range(TOWER_NUM):
            tower_location.append(file.readline().split(":")[1].split())

        # Read tower observation
        for _ in range(NUM_LINES_BEFORE_STEPS):
            file.readline()
        for _ in range(NUM_OF_STEPS):
            tower_observation.append(file.readline().split())
    return np.array(grid_world).astype(int), np.array(tower_location).astype(int), np.array(tower_observation).astype(float)


def createLabels(num):
    labels = []
    for i in range(num):
        for j in range(num):
            # labels.append({"X": i, "Y": j})
            labels.append([i, j])
    # return np.array(labels).reshape((num, num))
    labels_R = np.arange(num ** 2).reshape((num, num))
    return np.array(labels), labels_R


def createTransitionA(num, labels, labels_R,  grid_world):
    A = np.zeros((num ** 2, num ** 2))
    for i in range(num ** 2):
        X_old = labels[i][0]
        Y_old = labels[i][1]
        possible_next_steps = [[X_old-1, Y_old], [X_old+1, Y_old], [X_old, Y_old-1], [X_old, Y_old+1]]
        if(grid_world[X_old, Y_old] != 0):
            valid_coord_set = []
            for X_new, Y_new in possible_next_steps:
                if(isValidNextStep(X_new, Y_new, grid_world)):
                    valid_coord_set.append({"X": X_new, "Y": Y_new})
            if(len(valid_coord_set) > 0):
                for valid_coord in valid_coord_set:
                    A[i, labels_R[valid_coord["X"], valid_coord["Y"]]] = 1.0  / len(valid_coord_set)
                    # A[i, labels_R[valid_coord["X"], valid_coord["Y"]]] = (1.0 + a)/ (len(valid_coord_set) + a * NUM_OF_STATES)
                    # A[i, labels_R[valid_coord["X"], valid_coord["Y"]]] = np.log(1.0 / len(valid_coord_set))
        # A[A == 0] = a / (len(valid_coord_set) + a * NUM_OF_STATES)
    with open('out.txt', 'w') as f:
        print(A, file=f)
    # A = np.log(A)
    # print(A)
    return A


def isValidNextStep(X_new, Y_new, grid_world):
    if(X_new >= 0 and X_new < GRID_WORLD_SIZE and Y_new >=0 and Y_new < GRID_WORLD_SIZE and grid_world[X_new, Y_new] == 1):
        return True
    return False


def createEmissionB(num, labels, labels_R, tower_location, tower_observation):
    # B = []
    B = np.ones((NUM_OF_STATES, NUM_OF_STEPS))
    # B = np.log(B)
    tower_list = tower_location.tolist()
    for tower_index, tower in enumerate(tower_list):
        observation_list = tower_observation[:, tower_index].tolist()
        b = np.zeros((num ** 2, NUM_OF_STEPS))
        for ob_index, observation in enumerate(observation_list):
            for i in range(num ** 2):
                distance = np.linalg.norm(np.array(tower) - [labels[i]])
                # if(distance == 0):
                #     if(observation == 0):
                #         b[i, ob_index] = 1
                #         continue
                #     else:
                #         b[i, ob_index] = 0
                #         continue
                if (distance == 0):
                    if (observation == 0):
                        b[i, ob_index] = (1 + a)/(1 + NUM_OF_STATES * a)
                        continue
                    else:
                        b[i, ob_index] = a/(1 + NUM_OF_STATES * a)
                        continue
                else:
                    max_dis = np.floor(distance * TOWER_UPPER_BOUND * 10)
                    min_dis = np.ceil(distance * TOWER_LOWER_BOUND * 10)
                    valid_range_length = max_dis - min_dis + 1
                    probability = 1.0 / valid_range_length
                    # probability = (1.0 + a) / (valid_range_length + NUM_OF_STATES * a)
                    if(observation * 10 >= min_dis and observation * 10 <= max_dis):
                        b[i, ob_index] = probability
                        # b[i, ob_index] = 1
                    # b[b == 0] = a / (valid_range_length + a * NUM_OF_STATES)

        # B += np.log(b)
        B *= b
    # B = np.log(B)
    return B


def createInitialPi(grid_world):
    free_count = np.count_nonzero(grid_world == 1)
    pi = np.copy(grid_world).astype(float)
    pi[pi > 0.0] = 1.0 / free_count
    # pi[pi > 0.0] = (1.0 + a)/ (free_count + NUM_OF_STATES * a)
    # pi[pi <= 0.0] = a / (free_count + NUM_OF_STATES * a)
    # pi = np.log(pi)
    print(pi)
    return pi


def viterbi(pi, A, B, labels):
    T_1 = np.zeros((NUM_OF_STATES, NUM_OF_STEPS))
    T_2 = np.zeros((NUM_OF_STATES, NUM_OF_STEPS))
    for i in range(NUM_OF_STATES):
        # T_1[:, 0] = pi.ravel() + B[:, 0]
        T_1[:, 0] = pi.ravel() * B[:, 0]

    for j in range(1, NUM_OF_STEPS):
        for i in range(NUM_OF_STATES):
            # column = T_1[:, j-1] + A[:, i] + B[:, j]
            column = T_1[:, j - 1] * A[:, i] * B[:, j]
            # print(column, file=f)
            # T_1[i, j] = np.max(column[column < 0])
            # T_2[i, j] = np.argmax(column[column < 0])
            T_1[i, j] = np.max(column)
            T_2[i, j] = np.argmax(column)
    # with open('out.txt', 'w') as f:
    #     print(T_1, file=f)
    z = np.zeros((NUM_OF_STEPS,), dtype=int)
    x = np.zeros((NUM_OF_STEPS, 2), dtype=int)
    z[NUM_OF_STEPS - 1] = np.argmax(T_1[:, NUM_OF_STEPS - 1])
    print(np.max(T_1[:, NUM_OF_STEPS - 1]))

    x[NUM_OF_STEPS - 1] = labels[z[NUM_OF_STEPS - 1]]
    for i in range(NUM_OF_STEPS - 1, 0, -1):
        z[i - 1] = T_2[z[i], i]
        x[i - 1] = labels[z[i - 1]]
    # print(z)
    return x

    # return T_1


if __name__ == '__main__':
    grid_world, tower_location, tower_observation = readFile(INPUT_FILE_NAME)
    print(grid_world)

    # labels: 100 items, each being [x, y]
    # labels_R: From 0 - 99
    labels, labels_R = createLabels(GRID_WORLD_SIZE)
    np.set_printoptions(threshold=np.inf)
    A = createTransitionA(GRID_WORLD_SIZE, labels, labels_R, grid_world)
    np.set_printoptions(threshold=np.inf)
    # with open('out.txt', 'w') as f:
    #     print(A, file=f)

    B = createEmissionB(GRID_WORLD_SIZE, labels, labels_R, tower_location, tower_observation)

    # with open('out.txt', 'w') as f:
    #     print("A=\n", file=f)
        # print(A, file=f)
        # print("\n\n\n", file=f)
        # print("B=\n", file=f)
        # print(B, file=f)

    pi = createInitialPi(grid_world)
    # print(pi)

    x = viterbi(pi, A, B, labels)
    print()
    print(x.tolist())
