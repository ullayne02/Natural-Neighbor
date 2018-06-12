from sklearn.model_selection   import StratifiedKFold 
from sklearn.metrics   import average_precision_score
from sklearn.metrics   import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics   import accuracy_score
from sklearn.neighbors import KDTree
import operator
import math
import numpy as np
import pandas as pd

class Knn(object):
    def __init__(self, data, target): 
        self.data = data
        self.target = target
        self.testing_set = []
        self.traning_set = []
        self.target_test_set = []
        self.target_traning_set = []
    
    # Divide a base de dados de entrada em 10 folds balanceados
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

    # Retorna os indicies dos k vizinhos mais proximos a uma determinada instancia
    def get_nearest_neighbors(self, test_inst, traning_set, train_targ, k): 
        tree = KDTree(traning_set)
        _, ind = tree.query([test_inst], k+1)
        ind = list(ind[0])
        ind.pop(0)
        near_neighbor = [train_targ[i] for i in ind]
        return near_neighbor

    # Retorna a classe com maior quantidade de vizinhos associada 
    def get_response(self, neighbors): 
        votes = {}
        for x in neighbors: 
            if(x not in votes.keys()): 
                votes.update({x: 1})
            else: 
                votes[x] += 1
        sort = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        return sort[0][0]

    # Retorna a acuracia 
    def get_accuracy(self, testing_set, predictions):
        c = 0 
        for i in range(len(testing_set)): 
            if(testing_set[i] ==  predictions[i]):
                c += 1
        return c/float(len(testing_set))*100.0 
    
    # Retorna a precisao pelo metodo macro avarage 
    def get_precision(self, predict, target, labels):   #macro avarage
        cm = confusion_matrix(predict, target, labels=labels)
        df = pd.DataFrame(cm)
        i = 0
        tp = 0
        tp_fp = 0
        for row in cm:
            tp += row[i]
            tp_fp += df[i].sum()
            i+=1  
        return tp/tp_fp

    # Executa o KNN com K = {1, 3, 5, sqrt(n), NaN} 
    def main(self, nane): 
        self.split_kcross()
        data_size = int(math.sqrt(len(self.data)))
        K = [1, 3, 5, data_size, nane] 
        
        all_class = set()
        for x in self.target: 
            all_class.add(x)
        #print(all_class)
        precision_all = []
        acurracy_all = []
        for k in K: 
            
            accuracy_by_kcross = []
            precisin_by_kcross = []
            for i in range(len(self.testing_set)):
                prediction = []
                train_set = self.traning_set[i]
                test_set = self.testing_set[i]
                train_targ = self.target_traning_set[i]
                test_tar = self.target_test_set[i]
                for t in test_set:  
                    neighbors = self.get_nearest_neighbors(t, train_set,train_targ, k)
                    prediction.append(self.get_response(neighbors))
                accuracy = self.get_accuracy(test_tar, prediction)
                precision = self.get_precision(prediction, test_tar, list(all_class))
                precisin_by_kcross.append(precision)
                accuracy_by_kcross.append(accuracy)

            size = len(accuracy_by_kcross)
            precision_all.append(sum(precisin_by_kcross)/size)
            acurracy_all.append(sum(accuracy_by_kcross)/size)
        
        return (K, precision_all, acurracy_all)

