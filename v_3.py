import random
import numpy

import struct
size=1000
iteration_number=10
#best_score_progress=[]
best_score_progress=[]
avg_score_progress=[]
mean_avg_score_progress=[]
mean_best_score_progress=[]
std_avg_score_progress=[]
std_best_score_progress=[]
def create_starting_population(low,high,size):
    # Set up an initial array of all zeros
    population = numpy.random.uniform(low, high, size)
    population=numpy.unique(population)
    population=numpy.around(population,decimals=4)
    
    return population
def calc_fitness(x1,x2):
    
    fitness= 21.5 + x1 * numpy.sin(4 * numpy.pi * x1) + x2 * numpy.sin(20 * numpy.pi *x2)
    
    return fitness
def parent_selection(scores,x,y):
    n=numpy.sort(scores)
    
    idx=numpy.where(n == numpy.amax(n))
    
    parent_1=numpy.unique(x[len(idx):int(len(x)/2)])
    parent_2=numpy.unique(y[len(idx):int(len(y)/2)])
    return parent_1,parent_2

def crossover(parent_1,parent_2):
    chromosome_length=len(parent_1)
    crossover_point = random.randint(1,chromosome_length-1)
    
    # Create children. np.hstack joins two arrays
    child_1 = numpy.hstack((parent_1[0:crossover_point],
                        parent_2[crossover_point:]))
    
    child_2 = numpy.hstack((parent_2[0:crossover_point],
                        parent_1[crossover_point:]))
   
    return child_1,child_2
def mutate(individual, mutationRate):
    x=1/mutationRate
    for i in range(int(x)):
        child_1=individual[0:i]-random.random()
        #child_1 = numpy.hstack((individual[0:i],random.random()))
        #individual[i+1]=individual[i]-random.random()
        #print(i,individual[i])
    return child_1

maximum_generation=30
x=create_starting_population(-12.0,12.0,size)
y=create_starting_population(-6,6,size)
for iteration in range (iteration_number):
    

    for generation in range(maximum_generation):
    # Create an empty list for new population
    
    #x=create_starting_population(-12,12,10)
    #y=create_starting_population(-6,6,10)

    #for i in range(int(size/2)):
        #x=create_starting_population(-12,12,10)
        #y=create_starting_population(-6,6,10)
        x1 = []
        x2=[]
    #print("x:",x)
    #print("y:",y)
        scores=calc_fitness(x,y)
        k = parent_selection(scores,x,y)
    #print("parents",k)
    #print("parents",parents)
        #print("parents_shape",parents.shape)
        #parent_1=numpy.ndarray(shape=1)
        n=crossover(k[0],k[1])
    #print("child",n)
        j=mutate(n[0],0.025)
        j_1=mutate(n[1],0.025)
    #print("mutate_child:",j,j_1)
        f=numpy.concatenate((j,k[0],n[0]))
    #print("j:",j)
    #print("k[0]:",k[0])
    #print("n[0]:",n[0])
    #print("con_f",f)
        f_1=numpy.concatenate((j_1,k[1],n[1]))
            #h=numpy.unique(f)
    #h_1=numpy.unique(f_1)
    #mn=numpy.concatenate((h,x[0:len(h-size)]))
    #mn_1=numpy.concatenate((h_1,y[0:len(h_1-size)]))
    #ms=numpy.unique(mn)
    #ms_1=numpy.unique(mn_1)
    #x=f
    #y=f_1
        if len(f)<len(f_1):
            f=numpy.concatenate((f,x[0:(len(f_1)-len(f))]))
       
        if len(f)>len(f_1):
            f_1=numpy.concatenate((f_1,y[0:(len(f)-len(f_1))]))
        x=f
        y=f_1
        #x=numpy.concatenate((x[len(f):-1],f))
        #y=numpy.concatenate((y[len(f_1):-1],f_1))
    


        scores = calc_fitness(x,y)
        best_score=numpy.max(scores)
    idx=numpy.where(scores == numpy.amax(scores))
    print("co-ordinates:",idx)
    avg_score=numpy.average(scores)
    mean_avg_score=numpy.mean(avg_score_progress)
    mean_avg_score_progress.append(mean_avg_score)
    mean_best_score=numpy.mean(best_score_progress)
    mean_best_score_progress.append(mean_best_score)
    std_avg_score=numpy.std(avg_score_progress)
    std_best_score=numpy.std(best_score_progress)
    std_best_score_progress.append(std_best_score)
    std_avg_score_progress.append(std_avg_score)
    best_score_progress.append(best_score)
    avg_score_progress.append(avg_score)
    #scores=calc_fitness(x,y)
    #best_score=numpy.max(scores)
    #best_score_progress.append(best_score)
    #print("generation:",generation)
print("best score over iterations:",best_score_progress)   
print("average score over iteration:",avg_score_progress)
print("mean of average scores:",mean_avg_score_progress)
print("standard deviation of average score:",std_avg_score_progress)
print("mean of best scores:",mean_best_score_progress)
print("standard deviation of best score:",std_best_score_progress)
            
# GA has completed required generation
print ('End best score, % target: ', best_score)



# Plot progress
import matplotlib.pyplot as plt
plt.plot(mean_best_score_progress,label='mean_best_score')
plt.plot(mean_avg_score_progress,label='mean_avg_score')
plt.plot(std_avg_score_progress ,label='standard_deviation_avg_score')
plt.plot(std_best_score_progress, label='standard_deviation_best_score')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('PE')
plt.show()