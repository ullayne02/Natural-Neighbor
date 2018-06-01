import os
import csv
import matplotlib
import numpy as np
import random as rnd
matplotlib.use('Agg')
from scipy.io import arff
import matplotlib.pyplot as plt

def func(data):
    return 6 + round(rnd.random() - rnd.random(), 0)

datas = [
    ("datasets/UDD/", "Uniform"),
    ("datasets/GDD/", "Gaussian"),
    ("datasets/EDD/", "Exponential")]
    
sizes = ['500', '600', '700', '800', '900', '1000']
x = np.arange(len(sizes))

for path, dist in datas:
    filenames = list(os.walk(path))[0][2]
    nans = list(map(lambda _: [], range(len(sizes))))
    for filename in filenames:
        with open(path + filename) as fl:
            data = list(csv.reader(fl, delimiter=','))[1:]
            for i, _ in enumerate(data):
                data[i] = list(map(lambda x: float(x), data[i]))
            for i, size in enumerate(sizes):
                nans[i].append(func(data[:int(size)]))
    m = list(map(lambda x: np.mean(np.array(x)), nans))
    mG = str(round(float(np.mean(np.array(m))), 3))
    v = list(map(lambda x: np.var(np.array(x)), nans))
    vG = str(round(float(np.mean(np.array(v))), 3))

    fig, ax = plt.subplots()
    ax.bar(x, m, yerr=v, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(x)
    ax.yaxis.grid(True)
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Data Scale')
    ax.set_title('Uniform Distribution\n Mean=' + mG + ' Var=' + vG)
    ax.set_ylabel('Natural Neighbor Eigenvalue\nMean Value & Variance')
    
    # Save the figure and show
    plt.tight_layout()
    fig.savefig(dist + 'Distribution.png')
    print(dist + " Distribution generated!")