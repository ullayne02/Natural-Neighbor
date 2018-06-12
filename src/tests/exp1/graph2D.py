import nan
import csv
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import KDTree

datas = [
    ("tests/exp1/datasets/aggregation.csv", "Aggregation", [1, 9]),
    ("tests/exp1/datasets/pathbased.csv", "Pathbased", [2, 12]),
    ("tests/exp1/datasets/compound.csv", "Compound", [4, 8]),
    ("tests/exp1/datasets/spiral.csv", "Spiral", [4, 7]),
    ("tests/exp1/datasets/flame.csv", "Flame", [3, 5]),
    ("tests/exp1/datasets/jain.csv", "Jain", [2, 8])]

for path, name, ks in datas:
    with open(path) as fl:
        data = list(csv.reader(fl, delimiter=','))[1:]
        for i, _ in enumerate(data):
            data[i] = list(map(lambda x: float(x), data[i]))
        X = list(map(lambda x: x[:-1], data))
        Y = list(map(lambda x: x[-1], data))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        NaN = nan.Natural_Neighbor()
        NaN.read(X, Y)
        num, edges, _ = NaN.algorithm()
        minX, minY, maxX, maxY = 100, 100, 0, 0
        for h, n in edges:
            minX = min(minX, X[h][0], X[n][0])
            minY = min(minY, X[h][1], X[n][1])
            maxX = max(maxX, X[h][0], X[n][0])
            maxY = max(maxY, X[h][1], X[n][1])
            x = [X[h][0], X[n][0]]
            y = [X[h][1], X[n][1]]
            ax.add_line(Line2D(x, y, color='black'))
        
        ax.set_xlim(minX-(0.03*maxX), maxX+(0.03*maxX))
        ax.set_ylim(minY-(0.03*maxY), maxY+(0.03*maxY))
        ax.set_title(name + " - NaNE: " + str(num))
        
        # Save the figure and show
        plt.tight_layout()
        fig.savefig("tests/exp1/graphs/" + name + '/Graph - NaN: ' + str(num) + '.png')
        print(name + " graph generated! NaN:", num)
        
        tree = KDTree(X)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k in ks:
            for h, i in enumerate(X):
                ns = tree.query([i], k=k+1, return_distance=False)
                for n in list(ns[0])[1:]:
                    x = [X[h][0], X[n][0]]
                    y = [X[h][1], X[n][1]]
                    ax.add_line(Line2D(x, y, color='black'))
            ax.set_xlim(minX-(0.03*maxX), maxX+(0.03*maxX))
            ax.set_ylim(minY-(0.03*maxY), maxY+(0.03*maxY))
            ax.set_title(name + " - kNN: " + str(k))
            
            # Save the figure and show
            plt.tight_layout()
            fig.savefig("tests/exp1/graphs/" + name + '/Graph - kNN: ' + str(k) + '.png')
            print(name + " graph generated! kNN:", k)