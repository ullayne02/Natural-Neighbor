from src import nan
from src import knn
from src import enn


def main(): 
    nat_neighbor = nan.Natural_Neighbor()
    nat_neighbor.load('datasets/4.4/breast-cancer-wisconsin.csv') 
    nane = nat_neighbor.algorithm()
    print(nane)
    
    # Algoritmo KNN com o NaN 
    a = knn.Knn(nat_neighbor.data, nat_neighbor.target)
    print(a.main(nane)) #retorna a acuracia e precisao 

    # Algoritmo ENN com NaN
    b = enn.Enn(nat_neighbor.data, nat_neighbor.target)
    print(b.main(nane)) #retorna a acuracia 

if __name__ == '__main__':
	main()