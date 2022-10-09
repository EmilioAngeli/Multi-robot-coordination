#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import math
import time
import operator
import copy
import sys
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import numpy as np
import random
import math
import time
import operator
import copy
import sys
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import os
import pygame
import traceback
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, GridSearchCV

# import NN layers and other componenets.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt # for plotting data and creating different charts.
import numpy as np # for math and arrays
import pandas as pd # data from for the data.
import seaborn as sns # for plotting.


# # The Node class.

# In[2]:


class Node():
    #PA, robots, grid, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, t

    
    def __init__(self, PA, robots, C, grid, max_steps = None, env = None, n_to_pa=None, free_panels=None, current_insp_robot=None, current_rech_robot=None, urgencies=None, t=None, initializing = False):

        if initializing is True:
            self.C = C
            self.max_steps = max_steps
            self.t = 0
            self.n = len(grid)
            self.robots = robots
            self.PA = {}
            for p in PA:  #initializing the priority of each panel as 0
                p = tuple(p)
                self.PA[p] = 0
            self.grid = grid
            self.n = len(grid)
            
            #DICTIONARYIES (for time simulation, keys are time instants)
            self.free_panels = {}
            self.current_insp_robot = {}
            self.current_rech_robot = {}
            self.env = {}
            #set for 3700 iterations
            for i in range(2*max_steps):
                self.env[i] = [{},{}]
                self.current_insp_robot[i] = []
                #the recharging robots are initialized to be available in any time instant:
                self.current_rech_robot[i] = []
                for r in self.robots:
                    if self.robots[r][3] == 'r':
                        self.current_rech_robot[i].append(r) 
                        self.env[i][0][r] = robots[r]
                self.free_panels[i] = []
                for p in PA:
                    self.free_panels[i].append(tuple(p))    
        else:
            self.C = C
            self.max_steps = max_steps
            self.robots = robots
            self.PA = PA
            self.env = env
            self.free_panels = free_panels#
            self.current_insp_robot = current_insp_robot
            self.current_rech_robot = current_rech_robot
            self.t = t  #quando parte l'azione dopo (l'azione prima finisce a t-1)
            self.grid = grid 
            self.n_to_pa = n_to_pa        
            self.urgencies = urgencies
            self.n = len(grid)
           
    
    def initialize_env(self): 
        # This method returns the dictionary x: the keys are time instants while the values are lists 
        # [robots, panels], in this way is stored the situation of robots and panels at any time instant.
        
        C = self.C
        
        #OBSERVATIONS: 
        # - this method is implemented supposing that n. of panels is >= n. of robots,
        # - initially every inspecting robot goes to the nearest solar panel 
        #   not chosen by some other inspecting robot yet.
        panel_to_insp = {} #this is a dictionary: key = robot, value = panel to inspect.
        d = {}
        z = 0
        for r in self.robots:
            if self.robots[r][3] == 'i':
                z += 1 # z is the number of inspecting robots
                for p in self.PA:
                    d[(r,p)] = np.sqrt((self.robots[r][0]-p[0])**2 + (self.robots[r][1]-p[1])**2)
        chosen_r = []
        chosen_p = []
        while d != {} and len(chosen_r) < min(z+1,len(self.PA)):
            couple = min(d.items(), key = operator.itemgetter(1))[0]
            d.pop(couple)
            if couple[0] not in chosen_r and couple[1] not in chosen_p:
                panel_to_insp[couple[0]] = couple[1]
                chosen_r.append(couple[0])
                chosen_p.append(couple[1])
                
        #self.free_panels = [pa for pa in self.PA if pa not in chosen_p]
        
        #compute the path of each inspecting robot:
        paths = {}
        for r in panel_to_insp:
            paths[r] = self.path_i(self.robots[r][0:2], panel_to_insp[r], 0)
            for i in range(len(paths[r])):
                self.free_panels[i].remove(panel_to_insp[r])
            
        max_path_length = max([len(paths[r]) for r in paths])
        
        #adjourn the environment for the next steps:
        for i in range(max_path_length):
            robots = {}
            panels = {}
            
            #adjourning robots
            for r in paths:
                if i < len(paths[r]):
                    robots[r] = [*paths[r][i], C-i, 'i']
            
            #adjourning panels
            for pa in self.PA:
                #the priority of the panel is augmented by 1
                if  i == 0:
                    panels = self.PA
                    #robot = self.robots
                for r in paths:
                    if self.be_neighbours(paths[r][-1], pa):
                        if 0 < i < len(paths[r]):
                            panels[pa] = self.env[i-1][1][pa] + 1
                        if i == len(paths[r]) - 1:
                            panels[pa] = 0
                if i > 0 and pa in self.free_panels[i]:
                    panels[pa] = self.env[i-1][1][pa] + 1
            self.env[i] = [robots, panels]
            
            #the recharging robots stay steel at least until the longest path is completed
            for r in self.robots:
                if self.robots[r][3] == 'r':
                    self.env[i][0][r] = self.robots[r]
            
        for r in paths:
            self.current_insp_robot[len(paths[r])].append(r)
            
            
        
        next_t = min([i for i in self.current_insp_robot.keys() if self.current_insp_robot[i] != []]) 


        next_env = copy.deepcopy(self.env)
        next_current_insp_robot = copy.deepcopy(self.current_insp_robot)
        next_current_rech_robot = copy.deepcopy(self.current_rech_robot)
        next_free_panels = copy.deepcopy(self.free_panels)
        
        next_urgencies = 0
        for i in range(next_t):
            for panel in self.env[i][1]:
                next_urgencies -= (self.env[i][1][panel])**2

        for i in range(0, next_t-1): 
            next_env.pop(i, None)
            next_current_insp_robot.pop(i, None)
            next_current_rech_robot.pop(i, None)
            next_free_panels.pop(i, None)                
        
        
        return next_t, next_env, next_current_insp_robot, next_current_rech_robot, next_free_panels, next_urgencies
        
    
    def bring_to_start(self, total_t):
        t = copy.deepcopy(self.t)
        self.t = 1
        self.urgencies = 0
        new_env, new_free_panels, new_current_insp_robot, new_current_rech_robot = {}, {}, {}, {}
        
        for i in self.env:
            new_env[i-t+1] =  self.env[i]
            
        key_list = list(self.env.keys())
        key_list.sort()
        last_t = copy.copy(key_list[-1])
        
        for i in self.free_panels:
            new_free_panels[i-t+1] =  self.free_panels[i]
        for i in self.current_insp_robot:
            new_current_insp_robot[i-t+1] =  self.current_insp_robot[i]
        for i in self.current_rech_robot:
            new_current_rech_robot[i-t+1] =  self.current_rech_robot[i]
            
        for i in range(total_t):
            if i not in new_env:
                new_env[i] = [{},{}]
            if i not in new_current_insp_robot:
                new_current_insp_robot[i] = []
            if i not in new_current_rech_robot:
                new_current_rech_robot[i] = []
                for r in self.robots:
                    if self.robots[r][3] == 'r':
                        new_current_rech_robot[i].append(r) 
                        new_env[i][0][r] = self.env[last_t][0][r]
            if i not in new_free_panels:          
                self.free_panels[i] = []
                for p in self.PA:
                    self.free_panels[i].append(tuple(p))     
        
        
        self.env, self.free_panels, self.current_insp_robot, self.current_rech_robot = new_env, new_free_panels, new_current_insp_robot, new_current_rech_robot
        
     
    
    
    
    
    def PA_(self):
        return self.PA

    def robots_(self):
        return self.robots
    
    def C_(self):
        return self.C

    def grid_(self):
        return self.grid
    
    def max_steps_(self):
        return self.max_steps
    
    def n_to_pa_(self):
        return self.n_to_pa
    
    def pa_to_n_(self):
        pa_to_n = {}
        h = 1
        for p in self.PA:
            p = tuple(p)
            pa_to_n[p] = h
            h += 1
        return pa_to_n
    

    
    def initial_panel(self):
        #print('Starting panel situation:')
        #print(self.env[1][1])
        return self.env[1][1]
    
    
    
    
        
    def visualize(self, i):
        printa = copy.deepcopy(self.grid)
        env = self.env[i]
        for r in env[0]:
            printa[env[0][r][0]][env[0][r][1]] = r
        for v in printa: 
            print(v)
            
            
    def path_i(self,start, goal, t):  
        #FASTER AND OPTIMAL
        #this method returns the shortest path (while a* does not always do) the starting tile is included,
        #the last tile is a tile next to the target panel (goal)
        start, goal = tuple(start), tuple(goal)
        visited = set()
        parent = {}
        parent[start] = None
        current_tiles = [start]
        k = t
        start_time = time.time()
        while goal not in current_tiles:
            if time.time() - start_time > 0.1:
                return 'Too_long'
            frontier = [] 
            for current_tile in current_tiles:
                i, j = current_tile[0], current_tile[1]
                for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
                    # Get node position
                    neighb = (i + new_position[0], j + new_position[1])
                    # Make sure within range
                    if neighb[0] > (self.n - 1) or neighb[0] < 0 or neighb[1] > (self.n -1) or neighb[1] < 0:
                        continue
                    # Make sure walkable terrain
                    if neighb in visited or (self.grid[neighb[0]][neighb[1]] != '__' and neighb != goal):
                        continue
                    # Make sure not crushing with other robots:
                    crush = False
                    for robot in self.env[k][0]:
                        if self.env[k][0][robot][0] == neighb[0] and self.env[k][0][robot][1] == neighb[1]:
                            crush = True
                            continue
                    if neighb != start:
                        parent[neighb] = current_tile
                    if not crush:
                        frontier.append(neighb)
                        visited.add(neighb)
            #reset the current_tiles set as the new frontier:
            current_tiles = frontier
            k += 1
        path = []
        current_tile = goal
        while parent[current_tile] is not None:
            path.append(parent[current_tile]) 
            current_tile = parent[current_tile]
        path.reverse()
        return path
    
    
    
    def path_r(self, insp, rech, t):   #same idea as before but this time with 2 robots
        #insp and rech are the starting positions of the inspecting and the recharging robot, respectively.
        #this method returns 2 paths: the first is for the insp rob and the second for the recharging rob.
        #the starting tiles are included
        insp, rech = tuple(insp), tuple(rech)
        visited_insp = set()
        visited_rech = set()
        parent = {}
        parent[insp] = None
        parent[rech] = None
        current_insp, current_rech = [insp], [rech]
        
        for x in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: 
                    if insp == (rech[0] + x[0], rech[1] + x[1]): #then the robots are already next to each other
                        return [insp], [rech]
        
        k = t
        start_time = time.time()
        while True:
            frontier_insp, frontier_rech = [], []
            if time.time() - start_time > 0.1:
                return 'Too_long'
            #THE INSPECTING FRONTIER
            for current_tile in current_insp:
                i, j = current_tile[0], current_tile[1]
                for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent square
                    neighb = (i + new_position[0], j + new_position[1])
                    # Make sure within range
                    if neighb[0] > (self.n - 1) or neighb[0] < 0 or neighb[1] > (self.n -1) or neighb[1] < 0:
                        continue
                    # Make sure walkable terrain
                    if neighb in visited_insp or self.grid[neighb[0]][neighb[1]] != '__':
                        continue
                    # Make sure not crushing with other robots:
                    crush = False
                    for robot in self.env[k][0]:
                        if self.env[k][0][robot][0] == neighb[0] and self.env[k][0][robot][1] == neighb[1]:
                            crush = True
                            continue
                        
                        
                    #ARRESTING CRITERIUM:
                    if neighb in visited_rech:
                        insp_path, rech_path = [current_tile], [neighb]
                        #constructing the path of the inspecting robot:
                        while parent[current_tile] is not None:
                            insp_path.append(parent[current_tile]) 
                            current_tile = parent[current_tile]
                        insp_path.reverse()
                        #constructing the path of the recharging robot:
                        while parent[neighb] is not None:
                            rech_path.append(parent[neighb]) 
                            neighb = parent[neighb]
                        rech_path.reverse()
                        return insp_path, rech_path
                        
                    
                    if neighb != insp:
                        parent[neighb] = current_tile
                    
                    if not crush:
                        frontier_insp.append(neighb)
                        visited_insp.add(neighb)
                        
            k += 1           
            #reset the current_tiles set as the new frontier for the inspecting robot:
            current_insp = frontier_insp
            
            #THE RECHARGING FRONTIER
            for current_tile in current_rech:
                i, j = current_tile[0], current_tile[1]
                for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
                    # Get node position
                    neighb = (i + new_position[0], j + new_position[1])
                    # Make sure within range
                    if neighb[0] > (self.n - 1) or neighb[0] < 0 or neighb[1] > (self.n -1) or neighb[1] < 0:
                        continue
                    # Make sure walkable terrain
                    if neighb in visited_rech or self.grid[neighb[0]][neighb[1]] != '__':
                        continue
                    # Make sure not crushing with other robots:
                    crush = False
                    for robot in self.env[k][0]:
                        if self.env[k][0][robot][0] == neighb[0] and self.env[k][0][robot][1] == neighb[1]:
                            crush = True
                            continue   
                        
                    #ARRESTING CRITERIUM:
                    if neighb in visited_insp:
                        rech_path, insp_path = [current_tile], [neighb]
                        #constructing the path of the inspecting robot:
                        while parent[current_tile] is not None:
                            rech_path.append(parent[current_tile]) 
                            current_tile = parent[current_tile]
                        rech_path.reverse()
                        #constructing the path of the recharging robot:
                        while parent[neighb] is not None:
                            insp_path.append(parent[neighb]) 
                            neighb = parent[neighb]
                        insp_path.reverse()
                        return insp_path, [*rech_path, rech_path[-1]] #in this way the paths have the same length
                         
                    if neighb != rech:
                        parent[neighb] = current_tile
                    
                    if not crush:
                        frontier_rech.append(neighb)
                        visited_rech.add(neighb)
                      
            k += 1   
            #reset the current_tiles set as the new frontier for the recharging robot:
            current_rech = frontier_rech
        

    def be_neighbours(self, a, b): #2 positions are expected, it returns True if a and b are neighbours
        a, b = tuple(a), tuple(b)
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: 
            neighb = (a[0] + new_position[0], a[1] + new_position[1])
            if neighb == b:
                return True
        return False
    
    def current_insp(self, i = None):
        if i is None:
            return self.current_insp_robot[self.t] 
        return self.current_insp_robot[i]
        
    def current_rech(self): 
        return self.current_rech_robot[self.t]

    def current_env(self, t): 
        return self.env[t] 

    def current_t(self): 
        return self.t 

    
    
    def compute_hl(self, rob, action, wait_one = False):  
        # It returns the environment after the moove, its output needs to be usable by itself in order
        # to compute the successive action.
        
        t = self.t 
        C = self.C
        total_t =  2 * self.max_steps 
        
        #INITIAL CHECKS:
        
        
        if wait_one: # i.e. if no one of the current insp rob can do something
            

            
            ############### Errore
            vediamo = [i for i in self.current_insp_robot.keys() if (self.current_insp_robot[i] != [] and i > t)]
            if vediamo == []:
                next_t = t+1
            else:
                next_t = min(vediamo)
                
                        
            next_current_insp_robot = copy.deepcopy(self.current_insp_robot)
            next_current_insp_robot[next_t] = [* self.current_insp_robot[t], *next_current_insp_robot[next_t]]
            next_current_rech_robot = copy.deepcopy(self.current_rech_robot)
            next_free_panels = copy.deepcopy(self.free_panels)
            next_urgencies = copy.deepcopy(self.urgencies)
            
            
            for i in range(t, next_t):
                for r in next_current_insp_robot[t]:
                    self.env[i][0][r] = self.env[t-1][0][r]
                for panel in self.env[i][1]:
                    next_urgencies -= (self.env[i][1][panel])**2
            
            next_env = copy.deepcopy(self.env)
            for i in range(t-1, next_t-1): 
                next_env.pop(i, None)
                next_current_insp_robot.pop(i, None)
                next_current_rech_robot.pop(i, None)
                next_free_panels.pop(i, None)            

            return next_t, next_env, next_current_insp_robot, next_current_rech_robot, next_free_panels, next_urgencies

        

        # If the robot available at time t are more than 1 (rare eventuality) we compute the choice 
        # only for one (the first) of them and posticipate the choice for the others to the next time step.
        if len(self.current_insp_robot[t]) > 1:
            to_inser = []
            for rob_ in self.current_insp_robot[t]: 
                if rob_ != rob and rob_ not in self.current_insp_robot[t+1]:
                    to_inser.append(rob_)
                    self.env[t][0][rob_] = self.env[t-1][0][rob_]
                    self.env[t][0][rob_][2] = max(self.env[t-1][0][rob_][2] -1, 0)
            self.current_insp_robot[t+1] = [*to_inser, *self.current_insp_robot[t+1]] 
                    
        # If all the inspecting robots become available in same t, then we need to adjourn the solar panels.
        if len(self.current_insp_robot[t]) == len([r for r in self.robots if self.robots[r][3] == 'i']):
            for pa in self.env[t-1][1]:
                self.env[t][1][pa] = self.env[t-1][1][pa] + 1
        
        while True:
            
            if action != 0:
                PA_to_inspect = self.n_to_pa[action]
                path = self.path_i(self.env[t-1][0][rob][0:2], PA_to_inspect, t)[1:] 

                # adjourning self.free_panels:
                for i in range(t+1, t+len(path)):
                    if PA_to_inspect in self.free_panels[i]: 
                        self.free_panels[i].remove(PA_to_inspect)

                # SPECIAL CASE: the inspecting robot is already next to the panel to inspect
                if self.be_neighbours(self.env[t-1][0][rob][0:2],PA_to_inspect):
                    self.env[t][0][rob] = self.env[t-1][0][rob]
                    self.env[t][0][rob][2] = max(self.env[t][0][rob][2]-1, 0)
                    self.current_insp_robot[t+1].append(rob)
                    self.env[t][1][PA_to_inspect] = 0
                    for pa in self.PA:
                        if pa not in self.env[t+1][1]:
                            self.env[t+1][1][pa] = self.env[t][1][pa] + 1   
                    j = copy.copy(t+1)
                    while PA_to_inspect in self.env[j][1]:
                        self.env[j][1][PA_to_inspect] = j - t 
                        j += 1
                    break

                # arrived here the panel and the recharger are not next to eachother
                last_C = self.env[t-1][0][rob][2]
                for i in range(t, t+len(path)): 

                    #adjourning panels:
                    for pa in self.PA:
                        if pa not in self.env[i][1]:
                            self.env[i][1][pa] = self.env[i-1][1][pa] + 1           

                    #adjourning current robot:
                    self.env[i][0][rob] = [*path[i-t], max(last_C-(i-t)-1, 0 ), 'i']

                    #if the robot is discharged before of the finish of the action:
                    if self.env[i][0][rob][2] <= 0 and i < t+len(path)-1:
                        self.current_insp_robot[i+1].append(rob)
                        for j in range(i,t+len(path)):
                            self.free_panels[j].append(PA_to_inspect)                    
                        break

                    #instructions to finish the action:
                    if i == t+len(path)-1:
                        self.env[i][1][PA_to_inspect] = 0
                        j = i+1
                        while PA_to_inspect in self.env[j][1]:
                            self.env[j][1][PA_to_inspect] = j - i
                            j += 1
                        self.current_insp_robot[i+1].append(rob)
    
                break
            ###############################################################################################
            # ACTION 0: go recharge.
            
            
            #SPECIAL CASE: the insp rob is already next to a recharger.
            if action == 0:
                interrupt = False
                for rech in self.env[t-1][0]:
                    if self.env[t-1][0][rech][3] == 'r' and self.be_neighbours(self.env[t-1][0][rob][0:2],self.env[t-1][0][rech][0:2]) and rech in self.current_rech_robot[t]:
                        self.env[t][0][rob] = self.env[t-1][0][rob]
                        self.env[t][0][rob][2] = C
                        self.current_rech_robot[t].remove(rech)
                        self.current_insp_robot[t+1].append(rob)
                        for pa in self.PA:
                            if pa not in self.env[t+1][1]:
                                self.env[t+1][1][pa] = self.env[t][1][pa] + 1    
                        interrupt = True 
                        break      
                if interrupt:
                    break
                    
            #FIRST CASE: the inspecting robot is not discharged yet, both the insp and the rech moove toward each others.
            if action == 0 and self.env[t-1][0][rob][2] > 0:

                # The distance is evaluated in terms of length of the path needed: 
                # we consider the possible presence of obs
                rech = self.current_rech_robot[t][0]
                path = self.path_r(self.env[t-1][0][rob][0:2],self.env[t-1][0][rech][0:2], t)
                rob_path = path[0][1:]                
                rech_path = path[1][1:]
                
                if len(self.current_rech_robot[t]) > 1:
                    for i in range(1,len(self.current_rech_robot[t])):
                        rech_ = self.current_rech_robot[t][i]
                        path_ = self.path_r(self.env[t-1][0][rech][0:2], self.env[t-1][0][rob][0:2], t)[1:]
                        rob_path_ = path_[0][0][1:]                
                        rech_path_ = path_[0][1][1:]
                        if len(path_) < len(path):
                            rech, rob_path, rech_path = rech_, rob_path_, rech_path_
                                        
                
                last_C = self.env[t-1][0][rob][2]
                for i in range(t, t+len(rob_path)):

                    #adjourning involved robots:
                    self.env[i][0][rob] = [*rob_path[i-t], max(last_C-(i-t)-1, 0),'i']
                    self.env[i][0][rech] = [*rech_path[i-t], -1, 'r']

                    #the current recharging robot is occupied now:
                    self.current_rech_robot[i].remove(rech)

                    #adjourning panels:
                    for pa in self.PA:
                        if pa not in self.env[i][1]:
                            self.env[i][1][pa] = self.env[i-1][1][pa] + 1    

                    #if the robot is discharged before of the finish of the action:
                    if self.env[i][0][rob][2] <= 0 and i < t+len(rob_path)-1:
                        self.current_insp_robot[i+1].append(rob)
                        for j in range(i, total_t):
                            self.env[j][0][rech] = self.env[j][0][rech]
                        break

                    #instructions to finish the action:
                    if i == t+len(rob_path)-1:
                        self.current_insp_robot[i+1].append(rob)
                        self.env[i][0][rob][2] = C

                for i in range(t+len(rob_path)-1, total_t): 
                    self.env[i][0][rech] = self.env[t+len(rob_path)-1][0][rech]
                
                break 
                
            #SECOND CASE: the inspecting robot is discharged, the recharged robot needs to reach it.
            if action == 0 and self.env[t-1][0][rob][2] <= 0:
                # The distance is evaluated in terms of length of the path needed: 
                
                # qui 
                rech = self.current_rech_robot[t][0]
                
                
                
                path = self.path_i(self.env[t-1][0][rech][0:2], self.env[t-1][0][rob][0:2], t)[1:]
                if len(self.current_rech_robot[t]) > 1:
                    for i in range(1,len(self.current_rech_robot[t])):
                        rech_ = self.current_rech_robot[t][i]
                        path_ = self.path_i(self.env[t-1][0][rech][0:2], self.env[t-1][0][rob][0:2], t)[1:]
                        if len(path_) < len(path):
                            rech, path = rech_, path_
                            
                        
                last_C = self.env[t-1][0][rob][2]
                for i in range(t, t+len(path)):

                    #adjourning involved robots:
                    
                    self.env[i][0][rob] = [*self.env[t-1][0][rob][0:2], max(last_C-(i-t)-1,0),'i']
                    self.env[i][0][rech] = [*path[i-t], -1, 'r']

                    #the current recharging robot is occupied now:
                    self.current_rech_robot[i].remove(rech)

                    #adjourning panels:
                    for pa in self.PA:
                        if pa not in self.env[i][1]:
                            self.env[i][1][pa] = self.env[i-1][1][pa] + 1    

                    #instructions to finish the action:
                    if i == t+len(path)-1:
                        self.current_insp_robot[i+1].append(rob)
                        self.env[i][0][rob][2] = C

                for i in range(t+len(path)-1, total_t): 
                    self.env[i][0][rech] = self.env[t+len(path)-1][0][rech]

                break
                
                
        next_t = min([i for i in self.current_insp_robot.keys() if (self.current_insp_robot[i] != [] and i > t)])
        next_env = copy.deepcopy(self.env)
        next_current_insp_robot = copy.deepcopy(self.current_insp_robot)
        next_current_rech_robot = copy.deepcopy(self.current_rech_robot)
        next_free_panels = copy.deepcopy(self.free_panels)
        next_urgencies = copy.deepcopy(self.urgencies)
        
        for i in range(t, min(self.max_steps,next_t)):  #the rewards are untill 200 time steps
            for panel in self.env[i][1]:
                next_urgencies -= (self.env[i][1][panel])**2

        for i in range(t-1, next_t-1): 
            next_env.pop(i, None)
            next_current_insp_robot.pop(i, None)
            next_current_rech_robot.pop(i, None)
            next_free_panels.pop(i, None)                
        
        
        return next_t, next_env, next_current_insp_robot, next_current_rech_robot, next_free_panels, next_urgencies
    
   
        
        
            
    def is_allowed(self, rob, action):  # Return False when the action is not possible.
        t = self.t
        
        # (1) 'go recharge' and no recharger is available
        if action == 0 and self.current_rech_robot[t] == []: 
            return False
        # (2) 'go inspect' and self.current_insp is discharged
        if action != 0 and self.env[t-1][0][rob][2] <= 0:
            return False
        if action != 0 and self.n_to_pa[action] not in self.free_panels[t]:
            return False
        
        
        return True    
        # OBS: no action is possible if the robot is discharged and no recharger is available.
        
    def make_copy(self):
        robots = self.robots
        PA = self.PA
        grid = self.grid
        n_to_pa = self.n_to_pa 
        C =self.C
        max_steps = self.max_steps
        
        env = copy.deepcopy(self.env)
        free_panels = copy.deepcopy(self.free_panels)
        current_insp_robot = copy.deepcopy(self.current_insp_robot) 
        current_rech_robot = copy.deepcopy(self.current_rech_robot)
        t = copy.deepcopy(self.t)    
        urgencies = copy.deepcopy(self.urgencies)
        
        x = Node(PA, robots, C, grid, max_steps, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, t, initializing = False)

        return x
        
        
    
    def find_children(self, n_of_children): ##DA RICONTROLLARE
        
        current_available_robots = copy.copy(self.current_insp())
        children = []#set()
        while current_available_robots:
            rob = copy.deepcopy(current_available_robots[0])
            must_wait = True
            for action in range(0, len(self.PA)+1):
                if self.is_allowed(rob, action):
                    try:   
                        copia = self.make_copy()
                        t, env, current_insp_robot, current_rech_robot, free_panels, urgencies = copia.compute_hl(rob, action)
                    except (IndexError, ValueError, KeyError, TypeError):
                        continue
                    must_wait = False

                    n_of_children += 1
                    children.append((n_of_children,rob,action))   
            if must_wait:
                copia = self.make_copy()
                t, env, current_insp_robot, current_rech_robot, free_panels, urgencies = copia.compute_hl(rob, action, wait_one = True)

                n_of_children += 1
                #children.add((n_of_children,rob,'wait_one'))
                children.append((n_of_children,rob,'wait_one'))
            current_available_robots.remove(rob)   
        return children
    
    
    def find_random_child(self, with_action = False):
        current_available_robots = copy.copy(self.current_insp())
        available_actions = list(range(len(self.PA) + 1))
        index = random.randint(0,len(self.PA))
        action = available_actions[index]
        rob = current_available_robots[0]
        waiting = False
        stop = False
        errors = 0
        while not stop:
            while not self.is_allowed(rob, action):
                if not available_actions :
                    if len(current_available_robots) == 1: 
                        t, env, current_insp_robot, current_rech_robot, free_panels, urgencies = self.compute_hl(rob, action, wait_one = True)
                        current_available_robots.remove(rob)
                        waiting = True
                        stop = True
                        break
                    available_actions = list(range(len(self.PA) + 1))
                    index = random.randint(0,len(self.PA))
                    action = available_actions[index]
                    current_available_robots.remove(rob)
                    rob = current_available_robots[0]
                available_actions.remove(action)
                if available_actions:
                    index = random.randint(0,len(available_actions)-1)
                    action = available_actions[index]
            if not waiting:
                try:
                    t, env, current_insp_robot, current_rech_robot, free_panels, urgencies = self.compute_hl(rob, action)
                    stop = True
                    break
                except (IndexError, ValueError, KeyError, TypeError):
                    errors += 1
                    if errors > 100:
                        return 'error'
                    continue
        
        
        
        node = Node(self.PA, self.robots, self.C, self.grid, self.max_steps, env, self.n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, t)
        #(self, PA, robots, C, grid, max_steps = None, env = None, n_to_pa=None, free_panels=None, current_insp_robot=None, current_rech_robot=None, urgencies=None, t=None, initializing = False):

        if with_action is True:
            return node, (rob,action)
        return node
        
        
    
        
    def is_terminal(self):
        "Returns True if the node has no children"
        if self.t >= self.max_steps:
            return True
        return False
    
    def reward(self):
        return self.urgencies

    


# ## The MCTS class:

# In[3]:


from collections import defaultdict
import math

class MCTS:

    def __init__(self, Q=None, N=None, best_path=None, s=None, best_reward=None, children=None, best_action_seq = None):
 #self, Q=defaultdict(int), N=defaultdict(int), best_path=[], s=[], best_reward=-math.inf, children={}):
        
        if Q is None:
            Q = defaultdict(int)
        if N is None:
            N = defaultdict(int)
        if best_path is None:
            best_path = []
        if s is None:
            s=[]
        if best_reward is None:    
            best_reward=-math.inf
        if children is None:
            children = {}
        if best_action_seq is None:
            best_action_seq = []
    
        self.Q = Q  # total reward of each node
        self.N = N  # total visit count for each node
        self.best_path = best_path
        self.s = s
        self.best_reward = best_reward
        self.children = children  # children of each node
        self.best_action_seq = best_action_seq
        self.n_prunings = 0
        self.n_of_children = 0 
        
        
    def current_situation(self):
        return [self.Q, self.N, self.best_path, self.s, self.best_reward, self.children]
    
    def best_r(self):
        return self.best_reward
        
    def best_p(self):
        return self.best_path
   
    def scores(self):
        return self.s
    
    def best_actions(self):
        return self.best_action_seq
    
    def print_n_prunings(self):
        print('EARLY PRUNING OCCURRED ',self.n_prunings,' TIMES')
        
    def print_children(self):
        print(self.children)
    
    def do_rollout(self, node, e_weight, NN, other_actions, model, NN_game_rate, NN_mooves_rate):
        "Make the tree one layer better. (Train for one iteration.)"

        
        while True:
            action_path = self._select(node, e_weight)
            
            #reconstruct path and other_actions:
            path = [node]
            l = node
            other_actions1 = copy.deepcopy(other_actions)
            
            for k in range(1,len(action_path)):
                
                PA = l.PA_()
                robots = l.robots_()
                C = l.C_()
                grid = l.grid_()
                max_steps = l.max_steps_()
                n_to_pa = l.n_to_pa_()
                
                
                rob, action = action_path[k][1], action_path[k][2]
                copia = copy.deepcopy(l)
                if action == 'wait_one':
                    t, env, current_insp_robot, current_rech_robot, free_panels, urgencies = copia.compute_hl(rob, action, wait_one = True)
                else:
                    t, env, current_insp_robot, current_rech_robot, free_panels, urgencies = copia.compute_hl(rob, action)
                path.append(copia)        
                l = Node(PA, robots, C, grid, max_steps, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, t)

                
                if NN:
                    #adjourning other_actions
                    t = copia.current_t()
                    previs = copia.current_env(t)
                    durata = 0
                    while rob in previs[0]:
                        durata += 1
                        previs = copia.current_env(t + durata)
                    other_actions1[rob] = [action , t + durata]
                
                    
                    
            if len(action_path) == 1:
                break
            
            if urgencies <= self.best_reward:  #  EARLY PRUNING.
                #print('Early pruned: at heigh', len(action_path), 'the ugency =',-urgencies,'>', -self.best_reward,'= best score achived')
                self.children[action_path[-2]].remove(action_path[-1])
                self.n_prunings += 1
            else:
                break
        
        
        self._expand(l, action_path[-1])
        reward = self._simulate(l, path, action_path, NN, other_actions1, model, NN_game_rate, NN_mooves_rate)
        if reward != 'no_reward':
            self._backpropagate(action_path, reward)
    
    
    
    def _select(self, node, e_weight):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node, e_weight)  # descend a layer deeper
            
            
            
    def _expand(self, leaf, action_leaf):
        "Update the `children` dict with the children of `node`"
        if leaf in self.children:# or action_leaf in self.children:
            return  # already expanded
        self.children[action_leaf] = leaf.find_children(self.n_of_children)
        self.n_of_children += len(self.children[action_leaf])
        
        
        
    def _simulate(self, l, path, action_path, NN, other_actions1, model, NN_game_rate, NN_mooves_rate):
        "Returns the reward for a random simulation (to completion) of `node`"
        
        side_path = copy.deepcopy(path)
        side_action_path = []
        node = copy.deepcopy(l)
        
        if NN and random.random() < NN_game_rate:

            nodes_sequence, side_action_path, reward = NN_plays(model, node, other_actions1, NN_mooves_rate)
            side_path += nodes_sequence
        
            self.s.append(reward)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_path = copy.deepcopy(side_path)
                self.best_action_seq = [a[1:] for a in action_path[1:]] + side_action_path #DA AGGIUNGERE IL RESTO DEL PATH

            return reward
        
        
        #it = 0
        while True:
            #it += 1
            node_, action = copy.deepcopy(node.find_random_child(with_action = True))
            
            #if node_ == 'error':
            #    print('Error at height:', len(path)-1, ', known tree length:', it)
            #    self.children[action_path[-2]].remove(action_path[-1])
            #    return 'no_reward'
            
            side_path.append(node)
            side_action_path.append(action)
            
            if node_.is_terminal(): 
                side_path.append(node_)
                reward = node_.reward()
                self.s.append(reward)
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_path = copy.deepcopy(side_path)
                    self.best_action_seq = [a[1:] for a in action_path[1:]] + side_action_path #DA AGGIUNGERE IL RESTO DEL PATH
                    
                return reward
            
            node = copy.deepcopy(node_)
            
            
            
    def _backpropagate(self, action_path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(action_path):
            self.N[node] += 1
            self.Q[node] += reward
            #if reward > self.Q[node]:
            #    self.Q[node] = copy.copy(reward)
            

    def _uct_select(self, node, e_weight):
        "Select a child of node, balancing exploration & exploitation"
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        log_N_vertex = math.log(self.N[node]+1)
        def uct(n):   
            "Upper confidence bound for trees"
            return self.Q[n] / (self.N[n]+1) + e_weight * math.sqrt(log_N_vertex / (self.N[n]+1))
        return max(self.children[node], key=uct)
    
    def do_grid(self, n, PA, obstacles):
        w = [['__']*n]*n    
        w = np.array(w)
        for l in PA:#adding the solar panel
            w[l[0]][l[1]] = 'PA'
        for obs in obstacles:#adding the solar panel
            w[obs[0]][obs[1]] = '**'
        grid = w
        return grid

    def n_to_pa_(self, PA, fitt = 0):
        # We assign one and only one number to each panel:
        n_to_pa = {}
        pa_to_n = {}
        h = 1
        for p in PA:
            p = tuple(p)
            n_to_pa[h] = p
            pa_to_n[p] = h
            h += 1
        if fitt == 0:
            return n_to_pa
        else:
            return pa_to_n
        
    def find_random_start(self, C, n, PA, obstacles, robots, max_steps, with_current_actions = False):
        
        grid = self.do_grid(n, PA, obstacles)
        n_to_pa = self.n_to_pa_(PA)
        total_t = 3*max_steps
        nox = Node(PA, robots, C, grid, max_steps, initializing = True)
        t, env, current_insp_robot, current_rech_robot, free_panels, urgencies = nox.initialize_env()
        node = Node(PA, robots, C, grid, max_steps, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, t)
    
        
        while True:
            no = copy.deepcopy(node)
            current_actions = {}
            err = False
            
            while no.current_t() < max_steps/3:
                                
                copy_no = no.make_copy()
                no_ = copy_no.find_random_child()
                if no_ == 'error':
                    err = True
                    break
                
                if with_current_actions:
                    
                    copy_no = no.make_copy()
                    no_, (rob,action) = copy_no.find_random_child(with_action = True)
                    t = copy_no.current_t()
                    previs = copy_no.current_env(t)
                    durata = 0
                    while rob in previs[0]:
                        durata += 1
                        previs = copy_no.current_env(t + durata)
                    current_actions[rob] = [action , t + durata]
                
                no = copy.deepcopy(no_)       
                    
            if err == False:
                no.bring_to_start(total_t)
                envi = no.env[1][0]     
                if no.env[1][1] != {}:
                    if not with_current_actions:
                        return no   
                    return no, current_actions


# # Functions

# In[4]:


def initialize_random(tree, C, n, PA, obstacles, robots, max_steps, with_current_actions = False):
    
    if  with_current_actions == False:
        initial_node = tree.find_random_start(C, n, PA, obstacles, robots, max_steps)
        #print(initial_node.initial_panel())
        return initial_node
    
    initial_node, current_actions = tree.find_random_start(C, n, PA, obstacles,
                                                           robots, max_steps, with_current_actions = True)
    return initial_node, current_actions


# In[5]:


def do_rollout(initial_node, n_rollouts, alpha, other_actions = None,
               NN = False, model = None, NN_game_rate = 0.5, NN_mooves_rate = 0.9, disp = 500, pickling = True):
    #if you want to stop before run time --> pickling =True,   scores, p, actions = depickle_and_plot()
    
    
    node = copy.deepcopy(initial_node)
    exploration_weight = 1
    tree = MCTS()
    tt = time.time()

    for _ in range(n_rollouts):
        
        while True:
            try:
                
                tree.do_rollout(node, exploration_weight, NN, other_actions, model, NN_game_rate, NN_mooves_rate)
                break
            except (IndexError, ValueError, KeyError, TypeError):
                continue
        
        
        if _ < n_rollouts/10:
            scores = tree.s
            exploration_weight = -np.mean(scores)
        else:
            if -exploration_weight < max(scores)/2:
                exploration_weight **= alpha

        if _ % disp == 0:
            print('iteration:',_,'exploration weight:', exploration_weight, ', time: ', round(time.time()-tt,1) )
            
            if pickling:
                # The sequence of nodes and the sequence of actions:
                p = tree.best_p()
                actions = tree.best_actions()
                scores_ = tree.scores()

                # pickling p and actions:
                pickling_on = open("p.pickle","wb")
                pickle.dump(p, pickling_on)
                pickling_on.close()
                pickling_on = open("actions.pickle","wb")
                pickle.dump(actions, pickling_on)
                pickling_on.close()
                pickling_on = open("scores.pickle","wb")
                pickle.dump(scores_, pickling_on)
                pickling_on.close()

    print('finished in', time.time()-tt )
    
    
    p = tree.best_p()
    actions = tree.best_actions()
    scores = tree.scores()
    return p, actions, scores


# In[6]:


def depickle_and_plot(plot = True, scores = None):

    
    # depickling if not p, action, scores:
    if scores == None:
        pickle_off = open("scores.pickle", 'rb')
        scores = pickle.load(pickle_off)
        pickle_off = open("p.pickle", 'rb')
        p = pickle.load(pickle_off)
        pickle_off = open("actions.pickle", 'rb')
        actions = pickle.load(pickle_off)

    if plot == True:
        #plotting
        plt.plot(scores, "o", markersize=0.55)
        plt.title("Scores")
        plt.xlabel('rollout number')
        plt.ylabel('Score')
        plt.show()
        print('Best score:', max(scores), 'achived in iteration:', np.argmax(scores))
        #tree.print_n_prunings()
    
    if scores == None:
        return p, actions, scores


# In[7]:


def matrix_visualization(p, actions, total_t):
    

    xxx = 0
    n_coppie_utili = 0
    n_to_pa = p[1].n_to_pa_()
    
    for i in range(1,len(p)):
        nodo_ = p[i]
        nodo = p[i-1]
        t_ = nodo_.current_t()
        t = nodo.current_t()


        if nodo.is_terminal():
            break

        for j in range(t,min(t_, total_t)):

            if j == t:

                print('###################### ',i-1 ,'-th NODE . time step:', j, '######################################') 

                if i < len(p)-1:
                    action = actions[i-2]
                    if action[1] == 'wait_one':
                        a = 'waits'
                    elif action[1] == 0:
                        a = 'recharges'
                    else:
                        a = 'inspects panel ' + str(n_to_pa[action[1]])
                    print('CURRENT ACTION: ', action[0], a)

                    #IMPORTANT OBS: bisogna cambiare: must wait non viene capito----> check se un'azione non è possibile(--> must wait).
                    # scartare tutte le coppie (stato, opt action) dove l'opt action non è fattibile.
                    fitt = nodo.current_env(t-1) 
                    if a not in ['waits', 'recharges'] and fitt[0][action[0]][2] == 0:
                        print(fitt[0][action[0]][2] )
                        print('AZIONE NO BUENA')
                    else:
                        n_coppie_utili += 1
                        #fai lo storing della coppia (nodo, action) dove action[0] è il rob e 
                        # action[1] va da 0 a len(PA) + 1

            s = nodo.current_env(j)
            print('PANELS:', s[1])
            print('ROBOTS:', s[0])
            nodo.visualize(j)
            print('          ')
            for pa in s[1]:
                xxx += s[1][pa]**2

            #CHECKING ERRORS (commented because no errors occurr)
            if j < total_t-1:
                if j == t_- 1:
                    s_f = nodo_.current_env(t_)
                else:
                    s_f = nodo.current_env(j+1) 
                for r in s[0]:
                    if s[0][r][0:2] != s_f[0][r][0:2]:
                        if not nodo.be_neighbours(s[0][r][0:2], s_f[0][r][0:2]): # a robot does more than one step
                            print(r)
                            print('ERROR1: a robot does more than one step')
                            sys.exit()  
                        if s[0][r][2] == 0:  # a discharged robot mooves
                            print(r)
                            print('ERROR4: a discharged robot mooves')
                            sys.exit()
                for pa in s[1]:
                    if s_f[1][pa] - s[1][pa] > 1:  # a panel urgency increases of more than one unit per time step
                        print('ERROR2: a panel urgency increases of more than one unit per time step')
                        sys.exit()
                    if s[1][pa] == 0: # a non inspected panel's urgency becomes 0
                        if [r for r in s[0] if s[0][r][3] == 'i' and nodo.be_neighbours(s[0][r][0:2], pa)] == []:
                            print('ERROR3: a non inspected panel s urgency becomes 0')
                            sys.exit()

    print('TOTAL SCORE:', -xxx)
    print('coppie utili', n_coppie_utili)


# # Playing (Pygame).

# In[8]:



#PLOTTING FUNCTION:
def print_env(tile_size, env, time_step, gameDisplay, n, initial_node, obstacles, action=None, image_edge = 1000):#environment, j, gameDisplay, action)
    
    pa_to_n = initial_node.pa_to_n_()
    PA = initial_node.PA_()
    robots = initial_node.robots_()  
    
    image_edge = 1000
    
    #Loading images:
    insp_rob = pygame.image.load('/home/emilio/Documents/Tesi ISR tentativi utili/immagini_env/robot2.jpg')
    insp_rob  = pygame.transform.scale(insp_rob, (image_edge/n, image_edge/n))
    rech_rob = pygame.image.load('/home/emilio/Documents/Tesi ISR tentativi utili/immagini_env/robot3.jpg')
    rech_rob  = pygame.transform.scale(rech_rob, (image_edge/n, image_edge/n))
    panel = pygame.image.load('/home/emilio/Documents/Tesi ISR tentativi utili/immagini_env/panel.png')
    panel = pygame.transform.scale(panel, (image_edge/n, image_edge/n))
    brick = pygame.image.load('/home/emilio/Documents/Tesi ISR tentativi utili/immagini_env/bricks.png')
    brick = pygame.transform.scale(brick, (image_edge/n, image_edge/n))    
    grass = pygame.image.load('/home/emilio/Documents/Tesi ISR tentativi utili/immagini_env/grass.jpeg')
    grass = pygame.transform.scale(grass, (image_edge/n, image_edge/n))    
    
    gameDisplay.fill((255, 255, 255))
    #Environment:
    occupied = []
    for rech in [rob for rob in env[0] if env[0][rob][3] == 'r']:
        i,j =  env[0][rech][0:2]
        occupied.append([i,j])
        gameDisplay.blit(rech_rob, (j*tile_size,i*tile_size))
    for insp in [rob for rob in env[0] if env[0][rob][3] == 'i']:
        i,j =  env[0][insp][0:2]
        occupied.append([i,j])
        gameDisplay.blit(insp_rob, (j*tile_size,i*tile_size))
        
        font = pygame.font.SysFont('CFF', 30)
        text = font.render(str(insp)[-1] , True, (255,0,0), (255,255,255))
        gameDisplay.blit(text, (j*tile_size,i*tile_size))

        #font = pygame.font.SysFont('CFF', 25)
        #text = font.render(str(env[0][insp][2]), True, (0,0,0) )
        #gameDisplay.blit(text, ((j+0.35)*tile_size,(i+0.3)*tile_size))
    for obstacle in obstacles: 
        i, j = obstacle[0:2]
        occupied.append([i,j])
        gameDisplay.blit(brick, (j*tile_size,i*tile_size))
    for pa in PA:
        i, j = pa[0:2]
        occupied.append([i,j])
        gameDisplay.blit(panel, (j*tile_size,i*tile_size))
        font = pygame.font.SysFont('CFF', 30)
        text = font.render(str(env[1][pa]), True, (255,255,255) )
        gameDisplay.blit(text, ((j+0.3)*tile_size,(i+0.25)*tile_size))
        font = pygame.font.SysFont('CFF', 30)
        text = font.render(str(pa_to_n[pa]), True, (255,0,0) )
        gameDisplay.blit(text, (j*tile_size,i*tile_size))
    for i in range(n):
        for j in range(n):
            if [i,j] not in occupied:
                gameDisplay.blit(grass, (j*tile_size,i*tile_size))
    #Description:
    i = 40
    pygame.draw.rect(gameDisplay, (0,0,0), (1010,i-15,1000,i+10),0)
    font = pygame.font.SysFont('CFF', 32)
    text = font.render('Time step:  ' + str(time_step), True, (255,160,0) )
    gameDisplay.blit(text, (1050, i))
    i = 130
    font = pygame.font.Font('freesansbold.ttf', 30)
    text = font.render('INSPECTOR CHARGES:', True, (101,67,33), (255,255,255))
    gameDisplay.blit(text, (1045,i))
    i = 180
    
    for insp in [rob for rob in env[0] if env[0][rob][3] == 'i']:
        font = pygame.font.Font('freesansbold.ttf', 18)
        text = font.render( str(insp)[-1] + " inspector's charge = " + str(env[0][insp][2]), True, (101,67,33) )
        gameDisplay.blit(text, (1045,i))
        i += 35
        
    
    
def display_actions(game_actions, gameDisplay):
    i = 300
    font = pygame.font.Font('freesansbold.ttf', 30)
    text = font.render('PRESS A NUMBER:', True, (101,67,33), (255,255,255))
    gameDisplay.blit(text, (1045,i))
    i = 350
    for testo in game_actions:
        font = pygame.font.Font('freesansbold.ttf', 18)
        text = font.render(testo, True, (101,67,33), (255,255,255))
        gameDisplay.blit(text, (1045,i))
        i += 35
        
def display_finish(human_score, machine_score, gameDisplay, image_edge = 1000):
        if human_score > machine_score:
            q = '   >   '
            testol = '- You won -'
        else:
            q = '   <   '
            testol = '- The machine won -'
        
        gameDisplay.fill((0, 0, 0))
        font = pygame.font.SysFont('CFF', 40)
        
        
        testo = '***********************************************************************************'
        text = font.render(testo, True, (255,255,52) )
        text_rect = text.get_rect(center=(3*image_edge/4, image_edge*0.7/2))
        gameDisplay.blit(text, text_rect)
        
        testo = 'HUMAN  ' + str(human_score) + q + str(machine_score) + '  MACHINE'
        text = font.render(testo, True, (255,255,52) )
        text_rect = text.get_rect(center=(3*image_edge/4, image_edge*0.8/2))
        gameDisplay.blit(text,text_rect)
        
        testo = '******************************************************************'
        text = font.render(testo, True, (255,255,52) )
        text_rect = text.get_rect(center=(3*image_edge/4, image_edge*0.9/2))
        gameDisplay.blit(text, text_rect)
        
        text = font.render(testol, True, (255,255,52) )
        text_rect = text.get_rect(center=(3*image_edge/4, image_edge*1/2))
        gameDisplay.blit(text, text_rect)
        


# In[9]:


def visualize_sequence(p, actions, max_steps, n, obstacles):
    
    image_edge = 1000
    pygame.init()     
    tile_size = image_edge/n
    gameDisplay = pygame.display.set_mode((3/2 * image_edge, image_edge ))
    pygame.display.set_caption('Multi robot environment')
    finished = False
    i = 0
    while not finished:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True
            if event.type == pygame.KEYDOWN: #pressing any button
                if i < len(p)-1:
                    i += 1 
                    nodo_ = p[i]
                    nodo = p[i-1]
                    t_ = nodo_.current_t()
                    t = nodo.current_t()
                    if nodo.is_terminal():
                        time.sleep(100)
                        pygame.quit()
                    if actions:
                        action = actions[i-2]
                    for j in range(t, min(t_, max_steps)):
                        #if i < len(p)-1: 
                        environment = nodo.current_env(j)
                        #Environment displaying:
                        print_env(tile_size, environment, j, gameDisplay, n, p[1], obstacles)
                        
                        #pygame.image.save_extended(gameDisplay, prototipol) 
                        
                        pygame.display.update()
                        time.sleep(0.2)                


# In[10]:


def human_perform(initial_node, max_steps, n, obstacles, machine_best_score):
    
    
    pa_to_n = initial_node.pa_to_n_()
    n_to_pa = initial_node.n_to_pa_()
    PA = initial_node.PA_()
    robots = initial_node.robots_()
    C = initial_node.C_()
    grid = initial_node.grid_()
    max_steps = initial_node.max_steps_()
    
    
    
    image_edge =1000
    
    pygame.init()     
    tile_size = image_edge/n
    gameDisplay = pygame.display.set_mode((3/2 * image_edge, image_edge ))
    pygame.display.set_caption('Multi robot environment')


    nodo = copy.deepcopy(initial_node)
    t = nodo.current_t()
    finished = False
    first_node = True

    while not finished:

        possible_actions_ = nodo.find_children(0)
        possible_actions_ = [m[1:] for m in possible_actions_]
        #removing eventual duplicates:
        possible_actions = []
        for el in possible_actions_:
            if el not in possible_actions:
                possible_actions.append(el)
        
        
        index = 0
        game_actions = []
        for action in possible_actions:
            if action[1] == 'wait_one':
                a = ' waits'
            elif action[1] == 0:
                a = ' recharges'
            else:
                a = ' inspects panel ' + str(pa_to_n[n_to_pa[action[1]]])
            game_actions.append('Press ' + str(index) + ' ---> ' + ' inspector ' + str(action[0])[-1] + str(a))
            index += 1



        environment = nodo.current_env(t)
        if first_node:
            print_env(tile_size, environment, t, gameDisplay, n, initial_node, obstacles)
            first_node = False
        display_actions(game_actions, gameDisplay)
        pygame.display.update()

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


            if event.type == pygame.KEYDOWN:         

                try:
                    chosen_index = int(event.unicode)
                    rob, action = possible_actions[chosen_index]
                    if action == 'wait_one':
                        ti, env, current_insp_robot, current_rech_robot, free_panels, urgencies = nodo.compute_hl(rob, action, wait_one = True)
                    else:
                        ti, env, current_insp_robot, current_rech_robot, free_panels, urgencies = nodo.compute_hl(rob, action)
                    nodo_ = Node(PA, robots, C, grid, max_steps, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, ti)
                    #            PA, robots, C, grid, max_steps, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies=None, t=None, initializing = False):

                    t_ = nodo_.current_t()
                    if t_ >= max_steps:
                        ##
                        human_score = nodo_.reward()
                        display_finish(human_score, machine_best_score, gameDisplay)
                        pygame.display.update()
                        time.sleep(100)
                        #sys.exit()
                    for j in range(t, min(t_, max_steps)):
                        environment = nodo.current_env(j)
                        #Environment displaying:
                        print_env(tile_size, environment, j, gameDisplay, n, initial_node, obstacles)
                        pygame.display.update()
                        time.sleep(0.2)
                    nodo = copy.deepcopy(nodo_)
                    t = nodo_.current_t()

                except (ValueError, IndexError): #to be sure that the imput index is correct
                    continue



# In[11]:


def convert_aux(current_rob, action, nodo, other_actions):
    # returns a tuple ([x1, ... , xn], optimal action), where x1,x2, ... are respectively:
    # - coordinates and charge of the current available robot
    # - coordinates and charge of the other inspecting robots, their current task, number of time steps to finish it
    # - coordinates of the rechargers
    # - urgencies of the panels.
    
    t = copy.deepcopy(nodo.current_t())
    t = t-1
    env = copy.deepcopy(nodo.current_env(t))
    x1, x2, x3 = env[0][current_rob][0:3]
    x = [x1, x2, x3]
    
    for r in [r for r in env[0] if r != current_rob and env[0][r][3] == 'i']: #inspectors
        a, b, c = env[0][r][0:3]
        x = x + [a, b, c]
        # if the action is not finished yet and is not a waiting action ('wait_one'):
        if other_actions[r][0] != 'wait_one' and other_actions[r][1] - t > 0: 
            x.append(other_actions[r][0])
            x.append(other_actions[r][1]-t)
        else: # waiting is coded by the integer len(PA)+1
            x = x + [len(env[1])+1,-1]
        
    for r in [r for r in env[0] if env[0][r][3] == 'r']: #rechargers
        a, b = env[0][r][0:2]
        x = x + [a, b]
        
    for pa in env[1]:
        x.append(env[1][pa])
    
    if action == 'wait_one': # waiting is coded by the integer len(PA)+1
        action = len(env[1])+1

    if action != 'no_action':
        return (x, action)
    return x


# In[12]:


def convert(p, actions, max_steps, train_ratio,  stampa = False):
    # This functions returns a list of couples state - optimal moove. 
    
    
    current_actions = {}
    n_to_pa = p[1].n_to_pa_()
    
    if stampa:
        print('n_to_pa: ', n_to_pa)
    
    train_oc = []
    test_oc = []
    
    for i in range(1,len(p)):
        nodo_ = p[i]
        nodo = p[i-1]
        t_ = nodo_.current_t()
        t = nodo.current_t()


        if nodo.is_terminal():
            break

        for j in range(t,min(t_, max_steps)):

            if j == t:
                
                if stampa:
                    print('###################### ',i-1 ,'-th NODE . time step:', j, '######################################') 

                if i < len(p)-1:
                    action = actions[i-2]
                    az = copy.copy(action[1])
                    
                    if az == 'wait_one':
                        a = 'waits'
                    elif az == 0:
                        a = 'recharges'
                    else:
                        a = 'inspects panel ' + str(n_to_pa[az])

                    #IMPORTANT OBS: bisogna cambiare: must wait non viene capito----> check se un'azione non è possibile(--> must wait).
                    # scartare tutte le coppie (stato, opt action) dove l'opt action non è fattibile.
                    fitt = nodo.current_env(t-1) 
                    if a not in ['waits', 'recharges'] and fitt[0][action[0]][2] == 0:
                        az = 0
                        a = 'waits'
                    
                    if stampa:
                        print('CURRENT ACTION: ', action[0], a)
                    
                    # STORING:  current_actions is a dictionary, 
                    #           key:robot, value:(action, how many time steps to complete it)

                    previs = nodo.current_env(t)
                    durata = 0
                    while action[0] in previs[0]:
                        durata += 1
                        previs = nodo.current_env(t + durata)

                    current_actions[action[0]] = [az , t + durata]    
                    
                    if stampa:
                        for r in current_actions:
                            if current_actions[r][1] - t > 0: # l'azione non è ancora finita:
                                print('azione corrente di ', r, ' è :', current_actions[r][0], ' e durerà ancora'
                                      , current_actions[r][1] - t)
                            else:
                                print(r, 'sta aspettando istruzioni')


                    try: # we need to skip the first time steps where current_actions is not completed yet
                        momentaneo = convert_aux(action[0], az, nodo, current_actions)
                        
                        if random.random() > train_ratio:
                            train_oc.append(momentaneo)
                        else:
                            test_oc.append(momentaneo)
                            
                        if stampa:
                            print('to store: ',momentaneo)
                    except KeyError:
                        if stampa:
                            print('current_actions not yet completed')
                        continue

            s = nodo.current_env(j)
            
            if stampa:
                print('PANELS:', s[1])
                print('ROBOTS:', s[0])
                nodo.visualize(j)
                print('          ')

                
            #CHECKING ERRORS 
            if j < max_steps-1:
                if j == t_- 1:
                    s_f = nodo_.current_env(t_)
                else:
                    s_f = nodo.current_env(j+1) 
                for r in s[0]:
                    if s[0][r][0:2] != s_f[0][r][0:2]:
                        if not nodo.be_neighbours(s[0][r][0:2], s_f[0][r][0:2]): # a robot does more than one step
                            print(r)
                            print('ERROR1: a robot does more than one step')
                            #sys.exit()  
                            return
                        if s[0][r][2] == 0:  # a discharged robot mooves
                            print(r)
                            print('ERROR4: a discharged robot mooves')
                            #sys.exit()
                            return
                for pa in s[1]:
                    if s_f[1][pa] - s[1][pa] > 1:  # a panel urgency increases of more than one unit per time step
                        print('ERROR2: a panel urgency increases of more than one unit per time step')
                        #sys.exit()
                        return
                    if s[1][pa] == 0: # a non inspected panel's urgency becomes 0
                        if [r for r in s[0] if s[0][r][3] == 'i' and nodo.be_neighbours(s[0][r][0:2], pa)] == []:
                            print('ERROR3: a non inspected panel s urgency becomes 0')
                            #sys.exit()
                            return

    return train_oc, test_oc
        
        


# In[13]:


def clean_all(DB):
    train_oc = []
    test_oc = []
    pickling_on = open(str(DB)+"_train.pickle","wb")
    pickle.dump(train_oc, pickling_on)
    pickling_on.close()
    pickling_on = open(str(DB)+"_test.pickle","wb")
    pickle.dump(test_oc, pickling_on)
    pickling_on.close()
    return


# In[14]:


def store_couples(p, actions, max_steps, train_ratio, DB):
    
    new_train_oc, new_test_oc = convert(p, actions, max_steps, train_ratio)
    try: 
        # adjourining:
        pickle_off = open(str(DB)+"_train.pickle", 'rb')
        train_oc = pickle.load(pickle_off)
        train_oc += new_train_oc
        
        pickle_off = open(str(DB)+"_test.pickle", 'rb')
        test_oc = pickle.load(pickle_off)
        test_oc += new_test_oc
        
    except FileNotFoundError: #if it's the 1st time ---> create the DB
        train_oc = new_train_oc
        test_oc = new_test_oc
        
    # pickling:
    pickling_on = open(str(DB)+"_train.pickle", "wb")
    pickle.dump(train_oc, pickling_on)
    pickling_on.close()

    pickling_on = open(str(DB)+"_test.pickle","wb")
    pickle.dump(test_oc, pickling_on)
    pickling_on.close()
    
    


# In[15]:


def enlarge_DB(C, n, PA, obstacles, robots, max_steps,
               n_rollouts, alpha, k, disp, DB, NN_game_rate = 0.9, NN_mooves_rate = 0.9,
               model = None, train_ratio = 0.3):
    
    it = 0
    while it < k-1:
        
        try:
            tree = MCTS()
            if model:
                initial_node, other_actions = initialize_random(tree, C, n, PA, obstacles, robots,
                                                                max_steps, with_current_actions = True)
                p, actions, scores = do_rollout(initial_node, n_rollouts, alpha, other_actions,
                                                True, model, NN_game_rate = NN_game_rate,
                                                NN_mooves_rate = NN_mooves_rate, disp = disp, pickling = False)
            else:
                initial_node = initialize_random(tree, C, n, PA, obstacles, robots, max_steps)       
                p, actions, scores = do_rollout(initial_node, n_rollouts, alpha, disp)
            store_couples(p, actions, max_steps, train_ratio, DB)
            
            it += 1
            print('ITERATION ', it, 'COMPLETED')

        except (IndexError, ValueError, KeyError, TypeError): 
            #print(traceback.format_exc())
            continue


# In[17]:


def DB_display(DB, first_50 = False):
    pickle_off = open(str(DB)+"_train.pickle", 'rb')
    train_oc = pickle.load(pickle_off)
    pickle_off = open(str(DB)+"_test.pickle", 'rb')
    test_oc = pickle.load(pickle_off)
    print('Train set dimension: ', len(train_oc))
    print('Test set dimension: ', len(test_oc))
    if first_50:
        print('50 train set examples: ')
        print(train_oc[:50])
    


# In[16]:


def get_train_and_test_set(DB):
    
    
    pickle_off = open(str(DB)+"_train.pickle", 'rb')
    train_oc = pickle.load(pickle_off)
    pickle_off = open(str(DB)+"_test.pickle", 'rb')
    test_oc = pickle.load(pickle_off)
    n_labels = max([s[1] for s in train_oc]) # Ignoring the 'wait' option (that is coded as the last one).
    
    '''
    X_train = [s[0] for s in train_oc if s[1] != n_labels]
    Y_train = [s[1] for s in train_oc if s[1] != n_labels]
    X_test = [s[0] for s in test_oc if s[1] != n_labels]
    Y_test = [s[1] for s in test_oc if s[1] != n_labels]
    '''
    
    min_size_train = min( [len([s[0] for s in train_oc if s[1]==j]) for j in range(n_labels)] ) 
    min_size_test = min( [len([s[0] for s in test_oc if s[1]==j]) for j in range(n_labels)] )
    
    
    added_train, added_test = np.zeros(n_labels), np.zeros(n_labels)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    
    
    for i in range(len(train_oc)):
        
        if train_oc[i][1] < n_labels and added_train[train_oc[i][1]] < min_size_train:
            X_train.append(train_oc[i][0])
            Y_train.append(train_oc[i][1])
            added_train[train_oc[i][1]] += 1
    
    for i in range(len(test_oc)):
        
        if test_oc[i][1] < n_labels and added_test[test_oc[i][1]] < min_size_test:
            X_test.append(test_oc[i][0])
            Y_test.append(test_oc[i][1])
            added_test[test_oc[i][1]] += 1
    
    
    return X_train, Y_train, X_test, Y_test


# In[1]:


def create_model(X_train, Y_train, X_test, Y_test, Y_test1, EPOCHS = 200,
                 shape = [30,100,50],
                 activation_functions = ['LeakyReLU', 'LeakyReLU'],
                 batch_size = 16, dropout = 0.1, learning_rate = 0.00015):

    if len(shape) != len(activation_functions)+1:
        print('Error in the architecture, must be len(shape) = len(activation_functions)+1')
    
    
    model = Sequential()
    model.add(Dense(shape[0], input_shape = (X_train.shape[1],)))    # Input layer => input_shape must be explicitly designated       
    model.add(Dropout(dropout), )
    
    for i in range(len(activation_functions)):

        model.add(Dense(shape[i+1], activation = activation_functions[i]))    # Input layer => input_shape must be explicitly designated       
        model.add(Dropout(dropout), )
    
    model.add(Dense(Y_train.shape[1], activation='softmax'))                          # Output layer => output dimension = 1 since it is a regression problem
    model.add(Dropout(dropout), )
    
    # Activation: sigmoid, softmax, tanh, relu, LeakyReLU. 
    #Optimizer: SGD, Adam, RMSProp, etc. # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    
    optimizer = optimizers.Adam(learning_rate)
    model.compile(loss='categorical_crossentropy',#from_logits=True),
                optimizer=optimizer,
                metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
    
    
    #MODEL SUMMARY.
    print('Here is a summary of this model: ')
    model.summary()
    history = model.fit(
        X_train, 
        Y_train,
        batch_size = batch_size,
        epochs=EPOCHS, 
        verbose=1,
        shuffle=True,
        steps_per_epoch = int(X_train.shape[0] / batch_size) ,
        validation_data = (X_test, Y_test),   
    )

    #PLOTS.
    #plotting accuracy and loss:
    '''plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Cross-Validation'], loc='upper left')
    plt.show()'''
    
    #plotting the confusion matrix:
    ax= plt.subplot()
    predict_results = model.predict(X_test)
    # predict_results = (predict_results.argmax())
    predict_results= predict_results.argmax(axis = 1)
    cm = confusion_matrix(Y_test1, predict_results)
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    # ax.xaxis.set_ticklabels(['Positive', 'Negative']); ax.yaxis.set_ticklabels(['Positive', 'Negative']);

    
    return model


# In[1]:


def NN_plays(model, initial_node, other_actions, NN_mooves_rate = 1):
    
    node = copy.deepcopy(initial_node)
    grid = node.grid_()
    n_to_pa = node.n_to_pa_()
    PA = node.PA_()
    robots = node.robots_()
    C = node.C_()
    max_steps = node.max_steps_()
    n_labels = len(PA)+1
    
    #PA, robots, C, grid, max_steps, env, n_to_pa,
    #free_panels, current_insp_robot, current_rech_robot, urgencies, ti)

    nodes_sequence = []
    actions_sequence = []
    
    while True:

        current_robots = node.current_insp()
        rob = current_robots[0]

        x = convert_aux(rob, 'no_action', node, other_actions)
        
        if random.random() < NN_mooves_rate:
        
            # transforming x so that it is readable by model.predict:
            a = {}
            for i in range(len(x)):
                a[i] = [x[i]]
            c = pd.DataFrame(a)
            prob = model.predict(c)
            prob = prob[0]
            
        else:
            prob = [random.random() for _ in range(n_labels)]
        
        
        possible_actions = node.find_children(0)   
        possible_actions = [m[1:] for m in possible_actions if m[1] == rob] #(n_of_children,rob,action)--->(rob,action)

        while True:

            if sum(prob) == 0: #then 'wait_one' 
                ti, env, current_insp_robot, current_rech_robot, free_panels, urgencies = node.compute_hl(rob, action, wait_one = True)
                node_ = Node(PA, robots, C, grid, max_steps, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, ti)
                nodes_sequence.append(node)
                actions_sequence.append((rob, 'wait_one'))
                
                #adjourning other_actions
                t = node.current_t()
                previs = node.current_env(t)
                durata = 0
                while rob in previs[0]:
                    durata += 1
                    previs = node.current_env(t + durata)
                other_actions[rob] = [action , t + durata]

                break 

            action = np.argmax(prob) #choosing the best action

            if (rob, action) in possible_actions:

                try:
                    copia = node.make_copy()
                    ti, env, current_insp_robot, current_rech_robot, free_panels, urgencies = copia.compute_hl(rob, action)
                    node_ = Node(PA, robots, C, grid, max_steps, env, n_to_pa, free_panels, current_insp_robot, current_rech_robot, urgencies, ti)

                    #adjourning other_actions
                    t = copia.current_t()
                    previs = copia.current_env(t)
                    durata = 0
                    while rob in previs[0]:
                        durata += 1
                        previs = copia.current_env(t + durata)
                    other_actions[rob] = [action , t + durata]
                    nodes_sequence.append(copia)
                    actions_sequence.append((rob,action))
                    break

                except (ValueError, IndexError):
                    prob[action] = 0

            else:
                prob[action] = 0

        if node_ == 'error':
            break

        if node.is_terminal(): 
            reward = node_.reward()
            break

        node = copy.deepcopy(node_)

    return nodes_sequence, actions_sequence, reward


# In[ ]:


def random_agent(initial_node, iterations, machine_score,  disp):
    scores = []

    while len(scores) < iterations:
        node = copy.deepcopy(initial_node)

        while True:
            node_ = copy.deepcopy(node.find_random_child())
            if node_ == 'error':
                break
            if node_.is_terminal(): 
                reward = node_.reward()
                scores.append(reward)
                break
            node = copy.deepcopy(node_)
        if disp != None and len(scores) % disp == 0:
            print(str(len(scores)) + ' random games simulated, best score achieved: '
                  + str(max(scores)) + ', mean: ' + str(np.mean(scores)))
    
    print('Scores mean:',np.mean(scores))
    print('Best score:',max(scores))
    
    if machine_score:
        pepe = len([s for s in scores if s > machine_score])/iterations*100
        print('Percentage of better random games:', pepe, ' %')
        
    plt.plot(scores, "o", markersize = 0.85)
    if machine_score:
        plt.axhline(y = machine_score)
    plt.title("Scores")
    plt.xlabel('rollout number')
    plt.ylabel('Score')
    plt.show()
    #print('best random result:',max(scores), 'over a total of ', len(scores), 'random trials')
    #if machine_score:
    #    return pepe
    


# In[ ]:


def compare_models(models, iterations, C, n, PA, obstacles, robots, max_steps):
    
    model_scores = {}
    model_wins = {}
    
    for model in models:
        model_scores[model] = 0
        model_wins[model] = 0
        
    it = 0
    while it < iterations:
        
        try:
            tree = MCTS()
            initial_node, other_actions = initialize_random(tree, C, n, PA, obstacles, 
                                                               robots, max_steps, with_current_actions = True)
            scores = {}
            for model in models:
                p, actions_sequence, score = NN_plays(model, initial_node, other_actions, NN_mooves_rate = 1)
                scores[model] = score
            for model in models:
                model_scores[model] += scores[model]
            winning_model = max(scores.items(), key=operator.itemgetter(1))
            model_wins[winning_model[0]] += 1
                
        except (IndexError, ValueError, KeyError, TypeError):
            continue
        it += 1    
                
        if it % 10 == 0:
            print('Iteration '+ str(it) +' completed.')
    
    n_mod = 0
    for model in models:
        n_mod += 1
        print('Model '+str(n_mod)+' average score: '+str(model_scores[model]/iterations)
              +', best model in '+str(model_wins[model])+' iterations.')
        
    
            
           


# https://github.com/royjafari/DataAnalyticsForFun/blob/main/MLP%20Classification/MLP%20Classify%20-%20E.ipynb
# 
# https://github.com/zhailat/Introduction-to-machine-learning-Python/blob/master/Part%2007%20-%20Constructing%20a%20Multi-Class%20Classifier%20Using%20Neural%20Network%20with%20Python%20(Tensorflow%20%26%20Keras)/Ex04-NN-multi-class.ipynb
# 
# https://www.youtube.com/watch?v=oOSXQP7C7ck
