import math
import enn.enn
import numpy as np 
from enn.enn import ENN
from scipy.spatial.distance import euclidean
from sklearn.model_selection   import StratifiedKFold


class Enn(object):
    def __init__(self, data, target): 
        self.data = data                # Conjunto de instancias 
        self.target = target            # Conjunto de classes 
        self.testing_set = []           # Conjunto de instancias de teste dividido por fold
        self.traning_set = []           # Conjunto de instancias de treninamento dividido por fold
        self.target_test_set = []       # Conjunto de classes de teste dividido por fold
        self.target_traning_set = []    # Conjunto de classes de treinamento dividido por fold
    
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
    
    # Retorna a acuracia 
    def get_accuracy(self, testing_set, predictions):
        c = 0 
        for i in range(len(testing_set)): 
            if(testing_set[i] ==  predictions[i]):
                c += 1
        return c/float(len(testing_set))*100.0 

    # Executa o KNN com K = {1, 3, 5, sqrt(n), NaN} 
    def main(self,nane): 
        # Asserts 
        self.split_kcross() 
        data_size = int(math.sqrt(len(self.data)))
        K = [1, 3, 5, data_size, nane]
        acuraccy = []
        
        for k in K: 
            clf = ENN(k=k, distance_function=euclidean)
            acc = []
            for i in range(len(self.traning_set)): 
                clf.fit(np.array(self.traning_set[i]),np.array(self.target_traning_set[i]))
                pred = clf.predict(self.testing_set[i])
                acc.append(self.get_accuracy(self.target_test_set[i], pred))
            acuraccy.append(sum(acc)/10)
            print(sum(acc)/10)
        return acuraccy
