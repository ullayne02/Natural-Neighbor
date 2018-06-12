import nan
import csv
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.neighbors import KDTree

datas = [("tests/exp1/datasets/spiral3D.csv", "Spiral3D", [3, 6])]

for path, name, ks in datas:
    with open(path) as fl:
        data = list(csv.reader(fl, delimiter=','))[1:]
        for i, _ in enumerate(data):
            data[i] = list(map(lambda x: float(x), data[i]))
        X = list(map(lambda x: x[:-1], data))
        Y = list(map(lambda x: x[-1], data))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        NaN = nan.Natural_Neighbor()
        NaN.read(X, Y)
        num, edges, _ = NaN.algorithm()
        minX, minY, minZ, maxX, maxY, maxZ = 100, 100, 100, 0, 0, 0
        for h, n in edges:
            minX = min(minX, X[h][0], X[n][0])
            minY = min(minY, X[h][1], X[n][1])
            minZ = min(minZ, X[h][2], X[n][2])
            maxX = max(maxX, X[h][0], X[n][0])
            maxY = max(maxY, X[h][1], X[n][1])
            maxZ = max(maxZ, X[h][2], X[n][2])
            x = [X[h][0], X[n][0]]
            y = [X[h][1], X[n][1]]
            z = [X[h][2], X[n][2]]
            ax.plot_wireframe(x, y, z, color='black')
        
        ax.set_xlim(minX, maxX)
        ax.set_ylim(minY, maxY)
        ax.set_zlim(minZ, maxZ)
        ax.set_title(name + " - NaNE: " + str(num))
        ax.view_init(30, 50)
        
        # Save the figure and show
        plt.tight_layout()
        fig.savefig("tests/exp1/graphs/" + name + '/Graph - NaN: ' + str(num) + '.png')
        print(name + " graph generated! NaN:", num)
        
        tree = KDTree(X)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for k in ks:
            for h, i in enumerate(X):
                ns = tree.query([i], k=k+1, return_distance=False)
                for n in list(ns[0])[1:]:
                    x = [X[h][0], X[n][0]]
                    y = [X[h][1], X[n][1]]
                    z = [X[h][2], X[n][2]]
                    ax.plot_wireframe(x, y, z, color='black')
            ax.set_xlim(minX, maxX)
            ax.set_ylim(minY, maxY)
            ax.set_zlim(minZ, maxZ)
            ax.set_title(name + " - kNN: " + str(k))
            ax.view_init(30, 50)
            
            # Save the figure and show
            plt.tight_layout()
            fig.savefig("tests/exp1/graphs/" + name + '/Graph - kNN: ' + str(k) + '.png')
            print(name + " graph generated! kNN:", k)