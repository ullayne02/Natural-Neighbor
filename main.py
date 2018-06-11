from src import nan
from src import knn
from src import enn


def main(): 
    nat_neighbor = nan.Natural_Neighbor()
    nat_neighbor.load('iris.csv')
    nane = nat_neighbor.algorithm()
    print(nane)
    
    #a = knn.Knn(nat_neighbor.data, nat_neighbor.target)
    #print(a.main(17))

    b = enn.Enn(nat_neighbor.data, nat_neighbor.target)
    print(b.main(nane))

if __name__ == '__main__':
	main()