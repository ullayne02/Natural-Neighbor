import enn.enn
import numpy as np 
from sklearn.model_selection   import StratifiedKFold 
from scipy.spatial.distance import euclidean
from enn.enn import ENN
import math

class Enn(object):
    def __init__(self, data, target): 
        self.data = data
        self.target = target
        self.testing_set = []
        self.traning_set = []
        self.target_test_set = []
        self.target_traning_set = []
    
    def split_kcross(self): 
        skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
        for x, y in skf.split(self.data, self.target): 
            aux_testing = [self.data[t] for t in y]
            aux_traning = [self.data[t] for t in x]
            target_test_set = [self.target[t] for t in y]
            target_train_set = [self.target[t] for t in x]
            self.traning_set.append(aux_traning)
            self.testing_set.append(aux_testing)
            self.target_test_set.append(target_test_set)
            self.target_traning_set.append(target_train_set)

    def get_accuracy(self, testing_set, predictions):
        c = 0 
        for i in range(len(testing_set)): 
            if(testing_set[i] ==  predictions[i]):
                c += 1
        return c/float(len(testing_set))*100.0 

    def main(self,nane): 
        self.split_kcross() 
        
        K = [1, 3, 5, int(math.sqrt(len(self.data))), nane]
        acuraccy = []
        for k in K: 
            clf = ENN(k=k, distance_function=euclidean)
            acc = []
            for i in range(len(self.traning_set)): 
                clf.fit(np.array(self.traning_set[i]),np.array(self.target_traning_set[i]))
                pred = clf.predict(self.testing_set[i])
                acc.append(self.get_accuracy(self.target_test_set[i], pred))
            acuraccy.append(sum(acc)/10)
            #print(pred)
        return acuraccy
