import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np 
import math
import csv 

class Natural_Neighbor(object):

    def __init__(self): 
        self.nan_edges = {}
        self.all_data = []
        self.nan_num = {}
        self.repeat = {}
        self.target = []
        self.data = []
        self.knn = {}
    
    def load(self, filename):
        aux = []
        with open(filename, 'r') as dataset: 
            data = list(csv.reader(dataset))
            data.pop(0)
            for inst in data: 
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                aux.append(row)
        self.data = np.array(aux)
        
    def asserts(self): 
        for j in range(len(self.data)): 
            self.knn[j] = set()
            self.nan_edges[j] = set()
            self.nan_num[j] = 0
            self.repeat[j] = 0
    
    def count(self): 
        nan_zeros = 0 
        for x in self.nan_num: 
            if self.nan_num[x] == 0: 
                nan_zeros += 1 
        return nan_zeros
    
    def findKNN(self, inst, r, tree): 
        dist, ind = tree.query([inst], r+1)
        return np.delete(ind[0], 0)

    def algorithm(self):
        # ASSERT
        tree = KDTree(self.data)
        self.asserts()
        flag = 0 
        r = 1 

        while(flag == 0): 
            for i in range(len(self.data)): 
                knn = self.findKNN(self.data[i], r, tree)
                n = knn[-1]
                self.knn[i].add(n)
                if(i in self.knn[n] and i not in self.nan_edges[n]): 
                    self.nan_edges[i].add(n) 
                    self.nan_edges[n].add(i)  #checar se precisa disso aqui
                    self.nan_num[i] += 1
                    self.nan_num[n] += 1 
            
            cnt = self.count()
            rep = self.repeat[cnt]
            self.repeat[cnt] += 1
            if(cnt == 0 or rep >= math.sqrt(r - rep)): 
                flag = 1 
            else: 
                r += 1 
        return r

