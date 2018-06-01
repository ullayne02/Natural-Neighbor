from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np 
import math

class Natural_Neighbor(object):
    def __init__(self): 
        self.nan_edges = {}
        self.nan_num = {}
        self.repeat = {}
        self.data = []
        self.knn = {}
    
    def asserts(self): 
        i = 0 
        for x in self.data: 
            self.knn[x] = set()
            self.nan_edges[x] = set()
            self.nan_num[x] = 0
            self.repeat[i] = 0
            i += 1 
    
    def count(self): 
        nan_zeros = 0 
        for x in self.nan_num: 
            if self.nan_num[x] == 0: 
                nan_zeros += 1 
    
    def findKNN(self, inst, r, tree): 
        _, ind = tree.query(inst, r+1)
        result = []
        ind.pop(0)
        for i in ind: 
            result.append(self.data[i])
        return result

    def algorithm(self):
        # ASSERT
        tree = KDTree(self.data)
        self.asserts()
        flag = 0 
        r = 1 
        
        while(flag == 0): 
            for x in self.data: 
                knn = self.findKNN(x, r, tree)
                for n in knn: 
                    self.knn[x].add(n)
                    if(x in self.knn[n] and x not in self.nan_edges[n]): 
                        self.nan_edges[n].add(x)
                        self.nan_edges[x].add(n) #checar se precisa disso aqui
                        self.nan_num[x] += 1
                        self.nan_num[n] += 1
            
            cnt = self.count()
            self.repeat[cnt] += 1
            rep = self.repeat[cnt]
            if(cnt == 0 or rep >= math.sqrt(r - rep)): 
                flag = 1 
            r += 1 

            return r-1 
