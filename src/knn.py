from sklearn.model_selection import StratifiedKFold 
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import operator

class Knn(object):
    def __init__(self, n_neighbors, data, target, all_data): 
        self.data = data
        self.target = target
        self.all_data = all_data
        self.n_neighbors = n_neighbors 
        self.testing_set = []
        self.traning_set = []
    
    def split_kcross(self, dataset, targ, all_data): 
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        for x in targ: 
            self.target.add(x)
        for x, y in skf.split(dataset, targ): 
            aux_traning = [] 
            aux_testing = []
            aux_testing = [all_data[t] for t in y]
            aux_traning = [all_data[t] for t in x]
            self.traning_set.append(aux_traning)
            self.testing_set.append(aux_testing)

    def get_nearest_neighbors(self, test_inst, traning_set, k): 
        tree = KDTree(traning_set)
        _, ind = tree.query([test_inst], r+1)
        near_neighbor = [self.all_data[i] for i in ind]
        return near_neighbor

    def get_response(self, neighbors): 
        votes = {}
        for x in neighbors: 
            if(x[-1] not in votes.keys()): 
                votes.update({x[-1]: 1})
            else: 
                votes[x[-1]] += 1
        sort = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        return sort[0][0]
    
    def get_acurracy(self, testing_set, prediction):
        tp = 0
        testing_size = len(testing_set)
        for i in range(testing_size): 
            if(testing_set[i][-1] == prediction[i][-1]): 
                tp += 1 
        return tp/float(testing_size) 
    
    #def get_precision(self, )