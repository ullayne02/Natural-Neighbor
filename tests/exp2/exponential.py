import random as rnd

dim = 2     # Dimensão inicial
nDS = 200   # Numero de datasets a serem gerados 
nDM = 201   # Numero de atributos a serem gerados
nEX = 1000  # Numero de instancias por dataset

print("Generating 200 exponential datasets with 1000 instances each one:")
for n in range(nDS):
    n += 1
    with open("datasets/EDD/" + str(n) + ".arff", "a") as fl:
        fl.write("@RELATION dataset" + str(n) + "\n\n")
        for d in range(dim):
            d += 1
            fl.write("@ATTRIBUTE attr" + str(d) + " NUMERIC" + "\n")
        fl.write("\n@DATA\n")
        for _ in range(nEX):
            ls = []
            for _ in range(dim):
                ls.append(round(rnd.random()*rnd.random(), 4))
            fl.write(str(ls)[1:-1] + "\n")
        fl.close()
    if(dim % nDM == 0):
        dim = 1
    dim += 1
    if(n % 10 == 0):
        print(str(n) + " exponential datasets were generated")