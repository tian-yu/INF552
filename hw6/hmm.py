"""
INF 552 Homework 6
Hidden Markov Model(HMM) and Viterbi Algorithm
Group Members: Tianyu Zhang (zhan198),  Minyi Huang (minyihua),  Jeffy Merin Jacob (jeffyjac)
Date: 4/15/2018
Python 3.6
"""

import math
import numpy as np
grid_world= []
tower_location = []
noisy_distances = []
states =[]
transition_matrix = []
initial_probability =[]
states_coordinates = []
distance_matrix =[]
observation = []
#read grid world
def readfile(file_name):
    grid_w= []
    tower_l = []
    noisy_d = []
    with open(file_name) as f:
        f.readline()
        f.readline()
        
        for i in range(10):
            grid_w.append([int(j) for j in f.readline().split()])
    #read tower location
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        
        for i in range(4):
           tower_l.append([int(j) for j in  f.readline().split()[2:]]) 
    #read noisy distances
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        
        for i in range(11):
            noisy_d.append([float(j) for j in f.readline().split()])
    return grid_w,tower_l,noisy_d  

#calculate conditional probability table
def conditional_prob(gw):
    s = []
    transition_m =[]
    initial_p = []
    total_cells = 0
    valid_cells = 0
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    states_coord = {}
    for i in gw:
        for j in i:
            if j==1:
                states_coord[(valid_cells+1)] = [x1,y1]
                s.extend([valid_cells+1])
                valid_cells+=1
                total_cells+=1
                y1+=1
            else:
                total_cells+=1
                y1+=1
        y1 = 0
        x1+=1
    init_p =(1/float(valid_cells))
    initial_p = np.ones((1,valid_cells))*float(init_p)
    transition_m = np.zeros((valid_cells, valid_cells))

    
    for i in range(valid_cells):
        for j in range(valid_cells):
            x2,y2 = states_coord[i+1]
            x1,y1 = states_coord[j+1]
            transition_m[i][j]= probability_calc(x1,y1,x2,y2,gw)
    return s,initial_p,transition_m,states_coord

#calculate probability for transition matrix
def probability_calc(x1,y1,x2,y2,gw):
    if((x1-1 ==x2) and (y1 ==y2)):#bottom of
        return next_step_prob(x2,y2,gw)
    elif((x1+1 ==x2) and (y1 ==y2)):#top of
        return next_step_prob(x2,y2,gw)
    elif((x1 == x2) and (y1-1 ==y2)):#right of
        return next_step_prob(x2,y2,gw)
    elif((x1 ==x2) and (y1+1 ==y2)):#left off
        return next_step_prob(x2,y2,gw)
    else:
        return 0
    
    
 #calculate next step probobility for transition matrix   
def next_step_prob(x,y,gw):
    valid=0
    if((y-1>=0) and (gw[x][y-1]==1)):
        valid+=1
    if((y+1<10) and (gw[x][y+1]==1)):
        valid+=1
    if((x-1>=0) and (gw[x-1][y]==1)):
        valid+=1
    if((x+1<10) and (gw[x+1][y]==1)):
        valid+=1
    if(valid>0):
        return 1/float(valid)
    else:
        return 0

#distance of each cell to tower
def get_distance(tower_l, grid_world): 
    prob_m = []
    dis_matrix = [] 
    obser = []  
    max_distance = 0
    min_d = 0 
    max_d = 0
    p = 0.0
    dec_unit = 0.1**1
    tx, ty = 0, 0 
    
    max_distance = math.sqrt((9)**2 + (9)**2)
    for i in range(0, int(max_distance * 1.7/dec_unit)+1): 
        obser.extend([round(i*dec_unit,1)])
    for tower in tower_l:
        prob_m = []
        for cell in states:
            tx = tower[0]
            ty = tower[1]
            distance = math.sqrt((states_coordinates[cell][0]-tx)**2+(states_coordinates[cell][1]-ty)**2)
            min_d = round(0.7 * distance, 1) 
            max_d = round(1.3* distance, 1)
            p = 1/((max_d - min_d + dec_unit)/dec_unit) 
            prob_m.append([])
            for i in obser:
                if (min_d <= i and i <= max_d):
                    prob_m[cell-1].extend([p])
                else:
                    prob_m[cell-1].extend([0])                    
        dis_matrix.append(prob_m)
    return dis_matrix, obser

        
def viterbi(noisy_dist):
        trace_back_states = []
        states_max_prob = [] 
        previous_states = [] 
        states_seq = [] 
        em_prob = [] 
        for step in range(11):
            em_prob = remove_evidence(noisy_dist, step) 
            if step == 0: 
                states_max_prob = initial_probability * em_prob             
            else:
                states_max_prob, previous_states = max_prob_state(states_max_prob * transition_matrix) 
                states_max_prob = states_max_prob * em_prob
                trace_back_states.append(previous_states)       
        states_seq.extend([max_prob(states_max_prob)]) 
        trace_back_states.reverse()
        for _previous_state_list in trace_back_states:
            states_seq.extend([_previous_state_list[states_seq[-1]]])
        states_seq.reverse()    
        return states_seq
    
def remove_evidence(noisy_dist, step):
        state_prob = np.ones((1, 87))
        tmp_state_prob = []

        for i, _ev in enumerate(noisy_dist[step]):
            tmp_state_prob.append([])
            for j, _st in enumerate(states):
                for k, _obs in enumerate(observation): 
                    if(_ev == _obs):
                        tmp_state_prob[i].extend([distance_matrix[i][j][k]]) 
                        break
                    else:
                        pass
        
        for tmp_prob in tmp_state_prob: 
            state_prob *= tmp_prob 
        return state_prob
        
def max_prob_state(states_prob):
        sts_prob = states_prob
        prob = 0
        max_prob = 0
        max_pre_state = 0
        max_probs_list = []
        max_pre_states_list = []
        
        for current_state in range(87): 
            max_prob = 0
            previous_state = 0
            for previous_state in range(87):
                prob = sts_prob[current_state][previous_state]
                if (prob > max_prob):
                    max_prob = prob
                    max_pre_state = previous_state
                else:
                    pass 
            max_probs_list.extend([max_prob])  
            max_pre_states_list.extend([max_pre_state]) 
        return max_probs_list, max_pre_states_list

def max_prob(states_prob):
        sts_prob = states_prob
        state = 0
        max_prob = 0
        
        for st, prob in enumerate(sts_prob[0]): 
            if (prob > max_prob):
                max_prob = prob
                state = st
            else:
                pass    
        return state
        
def convert_state_to_coordinate(sequence_list, labels_dictionary):
        label_seq = []
        
        for seq in sequence_list:
            label_seq.extend([labels_dictionary[seq+1]]) 
        
        return label_seq 

if __name__=='__main__':
    grid_world,tower_location,noisy_distances = readfile("hmm-data.txt") 
    print ('Grid World')
    print (np.matrix(grid_world))
    print ("\n")
    print ('Tower Locations')
    print (tower_location)
    print ("\n")
    print ('Time Step Distance')
    print (np.matrix(noisy_distances))
    print ("\n")
    states,initial_probability,transition_matrix,states_coordinates=conditional_prob(grid_world) 
    distance_matrix,observation = get_distance(tower_location,grid_world)
    state_sequence =viterbi(noisy_distances)
#    print ("TRANSITION_MATRIX")
#    print (transition_matrix)
#    print ("\n")
#    print ('STATE COORDINATES')
#    print (states_coordinates)
#    print ("\n")
#    print ('STATE SEQUENCE')
#    print ([x+1 for x in state_sequence])
#    print ("\n")
    print ('Path:')
    print (convert_state_to_coordinate(state_sequence, states_coordinates))
    print ("\n")
