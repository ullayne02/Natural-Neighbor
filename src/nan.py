from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np 

class Natural_Neighbor(object):
    def __init__(self): 
        self.nan_edges = set()
        self.nan_num = {}
        self.data = []
        self.knn = {}
    
    def algorithm(self):
        # ASSERTS
        flag = 0 
        r = 1 
        tree = KDTree(self.data)

        while(flag == 0): 
            for x in self.data: 
                knn = self.findKNN(x, r, tree)
    
    def findKNN(self, inst, r, tree): 
        _, ind = tree.query(inst, r+1)
        result = []
        ind.pop(0)
        for i in ind: 
            result.append(self.data[i])
        return result

