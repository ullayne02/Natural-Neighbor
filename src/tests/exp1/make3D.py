import csv
import math

with open("tests/exp1/datasets/spiral.csv") as fl:
    data = list(csv.reader(fl, delimiter=','))[1:]
    
for i, _ in enumerate(data):
    data[i] = list(map(lambda x: float(x), data[i]))
X = list(map(lambda x: x[:-1], data))
Y = list(map(lambda x: x[-1], data))
minX = min(map(lambda x: x[0], X))
minY = min(map(lambda x: x[1], X))
maxX = max(map(lambda x: x[0], X))
maxY = max(map(lambda x: x[1], X))


groups = []
classes = set(Y)
for c in classes:
    groups.append([ i for i, _ in enumerate(X) if Y[i] == c ])
g = groups[2]

for i, _ in enumerate(X):
    X[i][0] = ((X[i][0] - minX)/(maxX - minX)) - 0.5
    X[i][1] = ((X[i][1] - minY)/(maxY - minY)) - 0.5

with open("tests/exp1/datasets/spiral3D.csv", 'w') as fl:
    theta = [math.pi*(-1), math.pi*0.1, math.pi*0.1]
    for c, t in enumerate(theta):
        base = 1
        step = 1.0/len(g)
        for i in g:
            X[i][0] = X[i][0]*math.cos(t) - X[i][1]*math.sin(t)
            X[i][1] = X[i][0]*math.sin(t) + X[i][1]*math.cos(t)
            fl.write(str(X[i])[1:-1] + "," + str(base) + "," + str(c+1) + "\n")
            base -= step