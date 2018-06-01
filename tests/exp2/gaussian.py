import random as rnd
import math

dim = 1     # Dimensão inicial
nDS = 200   # Numero de datasets a serem gerados 
nDM = 200   # Numero de atributos a serem gerados
nEX = 1000  # Numero de instancias por dataset
print("Generating 200 gaussian datasets with 1000 instances each one:")
for n in range(nDS):
    n += 1
    with open("datasets/GDD/" + str(n) + ".arff", "a") as fl:
        fl.write("@RELATION dataset" + str(n) + "\n\n")
        for d in range(dim):
            d += 1
            fl.write("@ATTRIBUTE attr" + str(d) + " NUMERIC" + "\n")
        fl.write("\n@DATA\n")
        for _ in range(nEX):
            ls = []
            for _ in range(dim):
                r1 = max(round(rnd.random(), 5), 0.00001)
                r2 = max(round(rnd.random(), 5), 0.00001)
                # Metodo de geração de simulações com Box-Muller
                box_mul = math.sqrt(-2*math.log1p(r1-1))*math.cos(2*math.pi*r2)
                ls.append(round((box_mul/9.8) + 0.5, 4))
            fl.write(str(ls)[1:-1] + "\n")
        fl.close()
    if(dim % nDM == 0):
        dim = 0
    dim += 1
    if(n % 10 == 0):
        print(str(n) + " gaussian datasets were generated")
