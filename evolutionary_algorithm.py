import psim
import random, copy
import numpy as np
from multiprocessing import Process, Value, Array
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(num=None, figsize=(16, 4), dpi=80, facecolor='w', edgecolor='k')

def visualizeAI(parameters):
    arrays = []
    arrays.append(parameters[:4])
    arrays.append(parameters[4])
    arrays.append(parameters[5])
    for a in arrays:
        print(a)
        plt.figure(figsize=(10.0,2.0))
        plt.xticks(np.arange(0,15))
        plt.yticks(np.arange(0,15))
        plt.bar(np.arange(0, len(a)), a)
        axes = plt.gca()
        axes.set_ylim([0,3])
        plt.show()
        
def visualizeAverageAI(ais):
    means = []
    errs = []
    for i in range(4):
        values = []
        for ai in ais:
            if (len(ai)==3):
                values.append(ai[0][i])
            else:
                values.append(ai[i])
        means.append(np.mean(values))
        errs.append(np.std(values))
    plt.figure(figsize=(10.0,2.0))
    plt.xticks(np.arange(0,15))
    plt.yticks(np.arange(0,15))
    plt.bar(np.arange(0, len(means)), means,yerr=errs,align='center', alpha=0.5, ecolor='black', capsize=10)
    axes = plt.gca()
    axes.set_ylim([0,3])
    plt.show()
    
    for i in range(2):
        means = []
        errs = []
        for j in range(11):
            values = []
            for ai in ais:
                if(len(ai)==3):
                    values.append(ai[1+i][j])
                else:
                    values.append(ai[4+i][j])
            means.append(np.mean(values))
            errs.append(np.std(values))
        plt.figure(figsize=(10.0,2.0))
        plt.xticks(np.arange(0,15))
        plt.yticks(np.arange(0,15))
        plt.bar(np.arange(0, len(means)), means,yerr=errs,align='center', alpha=0.5, ecolor='black', capsize=10)
        axes = plt.gca()
        axes.set_ylim([0,3])
        plt.show()    

def createRandomAI():
    parameters = []
    parameters.append(round(random.uniform(0,1),2))#overlapmultiplier
    parameters.append(round(random.uniform(0,1),2))#fortifymultiplier
    parameters.append(round(random.uniform(0,2),2))#emptybonus
    parameters.append(round(random.uniform(0,2),2))#spreadthreshold
    stations1=[]
    stations2=[]
    for i in range(11):
        stations1.append(round(random.uniform(0.5,1.5),2))
        stations2.append(round(random.uniform(0.5,1.5),2))
    parameters.append(stations1)
    parameters.append(stations2)
    return parameters

def transformAI(ai):
    newAI = []
    for i in range(4):
        newAI.append(ai[0][i])
    newAI.append(ai[1])
    newAI.append(ai[2])
    return newAI

def visualizeAverageAI(ais,showErrs=True):
#     if len(ais[0] > 3):
#         for ai in ais:
#             ai = transformAI(ai)
    means = []
    errs = []
    for i in range(4):
        values = []
        for ai in ais:
            values.append(ai[0][i])
        means.append(np.mean(values))
        errs.append(np.std(values))
    plt.figure(figsize=(10.0,2.0))
    plt.xticks(np.arange(0,15))
    plt.yticks(np.arange(0,15))
    plt.title("General computer player parameters")
    if showErrs:
        plt.bar(["overlap multiplier","fortify multiplier","empty bonus","spread treshold"], means,yerr=errs,align='center', alpha=0.5, ecolor='black', capsize=10)
    else:
        plt.bar(["overlap multiplier","fortify multiplier","empty bonus","spread treshold"], means,align='center', alpha=0.5, ecolor='black', capsize=10)
    axes = plt.gca()
    axes.set_ylim([0,3])
    plt.show()
    
    for i in range(2):
        means = []
        errs = []
        for j in range(11):
            values = []
            for ai in ais:
                values.append(ai[1+i][j])
            means.append(np.mean(values))
            errs.append(np.std(values))
        plt.figure(figsize=(10.0,2.0))
        plt.xticks(np.arange(0,15))
        plt.yticks(np.arange(0,15))
        plt.title("Round "+str(i+1)+" station preferences")
        if showErrs:
            plt.bar(np.arange(0, len(means)), means,yerr=errs,align='center', alpha=0.5, ecolor='black', capsize=10,)
        else:
            plt.bar(np.arange(0, len(means)), means,align='center', alpha=0.5, ecolor='black', capsize=10,)
        axes = plt.gca()
        axes.set_ylim([0,3])
        plt.show()
    
def mutateAI(parameters,maxMutation=0.1):
    for i in range(4):
        mutation = random.uniform(-maxMutation,maxMutation)
        parameters[i] += mutation
        parameters[i] = round(parameters[i],2)
    for i in range(11):
        mutation = random.uniform(-maxMutation,maxMutation)
        parameters[4][i] += mutation
        parameters[4][i] = round(parameters[4][i],2)
    for i in range(11):
        mutation = random.uniform(-maxMutation,maxMutation)
        parameters[5][i] += mutation
        parameters[5][i] = round(parameters[5][i],2)
    return parameters
def crossAI(p1,p2):
    newparam = []
    for i in range(4):
        newparam.append(round(random.uniform(p1[i],p2[i]),2))
    stations1 = []
    for i in range(11):
        stations1.append(round(random.uniform(p1[4][i],p2[4][i]),2))
    newparam.append(stations1)
    stations2 = []
    for i in range(11):
        stations2.append(round(random.uniform(p1[5][i],p2[5][i]),2))
    newparam.append(stations2)
    return newparam
    
popSize = 5             #the number of AIs in the genetic pool at any given moment
nrOfGenerations = 100     #how mony iterations the algorithm runs for
nrOfCrossovers  = 1     #how many new AIs are spawned by crossing the top AIs
nrOfRandomAIs = 1       #how many random AIs are spawned 
mutationStrength = 0.1  #the maximum fraction by which a parameter can be increased or decreased through mutation
nrOfEliminations = nrOfCrossovers+nrOfRandomAIs 

#assert nrOfEliminations <= popSize

population = []

processPool = [[None for x in range(popSize)] for y in range(popSize)] 
outcomes = [[Array('i', range(2)) for x in range(popSize)] for y in range(popSize)] 

for i in range(popSize):
    population.append(createRandomAI())
    print(population[i])


gens = 0
bestAIUnchanged = 0
previousBest = []
while bestAIUnchanged < 3:
    gens += 1
    print(gens)
    scores = [0]*popSize
    for p in range(popSize):
        for q in range(popSize):
            if not p == q:
                outcome = outcomes[p][q]
                processPool[p][q] = Process(target=psim.playGame,args=(population[p], population[q], False, outcome))
                processPool[p][q].start()
    for p in range(popSize):
        for q in range(popSize):
            if not p == q:
                processPool[p][q].join()
                outcome = outcomes[p][q]
                if outcome[0] > outcome[1]:
                    scores[p] += 1
                elif outcome[1] > outcome[0]:
                    scores[q] += 1
                else:
                    scores[p] += 0.5
                    scores[q] += 0.5
    print(scores)
    newPop = []
    for j in range(nrOfCrossovers):
        bestIndex = scores.index(max(scores))
        bestAI = population.pop(bestIndex)
        visualizeAI(bestAI)
        if j == 0 and bestAI == previousBest:
            bestAIUnchanged += 1
            print('consecutive winner')
        else:
            bestAIUnchanged = 0
            previousBest = bestAI
        scores.pop(bestIndex)
        secondIndex = scores.index(max(scores))
        secondBestAI = population.pop(secondIndex)
        scores.pop(secondIndex)
            
        newPop.append(mutateAI(crossAI(bestAI,secondBestAI),mutationStrength))
        newPop.append(bestAI)
        newPop.append(secondBestAI)
        
    for i in range(nrOfRandomAIs):
        newPop.append(createRandomAI())
        
    while len(newPop) < popSize:
        best = 0
        for i in range(len(scores)):
            if scores[i] > scores[best]:
                best = i
        scores.pop(best)
        newPop.append(population.pop(best))
        
    assert len(newPop) == popSize
    population = newPop
    for p in population:
        print(p)
