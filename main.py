from src import nan
from src import knn


def main(): 
    nat_neighbor = nan.Natural_Neighbor()
    nat_neighbor.load('iris.csv')
    nane = nat_neighbor.algorithm()
    print(nane)
    
    a = knn.Knn(nat_neighbor.data, nat_neighbor.target)
    print(a.main(nane))


if __name__ == '__main__':
	main()