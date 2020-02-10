import numpy
import random

maximum_generation=30
size=10
mutation_probability=0.025

pop_size = 1000 # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
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
import struct
def decimalToBinary(n): 
    return bin(n).replace("0b","")
def float_to_bin(num):
    return format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')

def bin_to_float(binary):
    return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]


def parent_selection(scores,x,y):
    numpy.sort(scores)
    idx=numpy.where(scores == numpy.amax(scores))
    parent_1=numpy.unique(x[idx])
    parent_2=numpy.unique(y[idx])
    return parent_1,parent_2

def crossover(parent_1,parent_2):
    paren_1=float_to_bin(parent_1)
    #paren_1=decimalToBinary(parent_1)
    #paren_2=bin((parent_2))
    paren_2=float_to_bin(parent_2)
    chromosome_length=len(paren_1)
    crossover_point = random.randint(1,chromosome_length-1)
    
    # Create children. np.hstack joins two arrays
    child_1 = numpy.hstack((paren_1[0:crossover_point],
                        paren_2[crossover_point:]))
    
    child_2 = numpy.hstack((paren_2[0:crossover_point],
                        paren_1[crossover_point:]))
    #print("child_11",child_1)
    ch_1=child_1[0]+child_1[1]
    ch_2=child_2[0]+child_2[1]
    ch_1=bin_to_float(ch_1)
    ch_2=bin_to_float(ch_2)
    return ch_1,ch_2

def mutation(ch,mutation_probability):
    random_value = numpy.random.uniform(-1.0, 1.0, 1)
    if (random_value > mutation_probability): 
        ch = ch - random_value
    return ch

#scores=calc_fitness(x,y)
#best_score=numpy.max(scores)
#avg_score=numpy.average(scores)
#mean_avg_score=numpy.mean(avg_score)
#mean_avg_score_progress.append(mean_avg_score)
#std_avg_score=numpy.std(avg_score)
#std_avg_score_progress.append(std_avg_score)
#mean_best_score=numpy.mean(best_score)
#mean_best_score_progress.append(mean_best_score)
#std_best_score=numpy.std(best_score)
#std_best_score_progress.append(std_best_score)
#best_score_progress.append(best_score)
#avg_score_progress.append(avg_score)
#print ('Starting best score, % target: ',best_score)
#print('Starting best score, % target: ',avg_score)
#print(x)
#print(y)
#print(parent_selection(scores,x,y))
num_iteration=10
x=create_starting_population(-12.0,12.0,1000)
y=create_starting_population(-6.0,6.0,1000)
for iteration in range(num_iteration):

    

    for generation in range(maximum_generation):
    # Create an empty list for new population
    
    #x=create_starting_population(-12,12,10)
    #y=create_starting_population(-6,6,10)

        for i in range(int(size/2)):
        #x=create_starting_population(-12,12,10)
        #y=create_starting_population(-6,6,10)
            x1 = []
            x2=[]
            scores=calc_fitness(x,y)
            parents = parent_selection(scores,x,y)
        #print("parents",parents)
        #print("parents_shape",parents.shape)
        #parent_1=numpy.ndarray(shape=1)
       
            parent_1=parents[0]
        #if (parent_1.shape==2):
         #   parent_1=parent[1]
        #else:
         #   parent_1=parent_1[0]
            parent_2=parents[1]
            parent_1=parent_1.astype(float)
        
        #print("parent_1",parent_1)
        #print("par_1",type(parent_1))
        #print("p_1",parent_1.size)
        #print("pa_1",parent_1.shape)
            parent_2=parent_2.astype(float)
            child_1, child_2 = crossover(parent_1, parent_2)
        #print("cd_1",type(child_1))
        
        
        x1.append(child_1)
        x2.append(child_2)
        x1=numpy.asarray(x1)
        x2=numpy.asarray(x2)
        x1=mutation(x1,mutation_probability)
        x2=mutation(x2,mutation_probability)
        x=numpy.sort(x)
        y=numpy.sort(y)
        x=x[:len(x)-2]
    #print(x.shape)
        y=y[:len(y)-2]
        x=numpy.concatenate((x,parent_1,x1))
        y=numpy.concatenate((y,parent_2,x2))
        parent_1=numpy.unique(parent_1)
    
    #x=numpy.where(x==parent_1, x1[0], x)
    #y=numpy.where(y==parent_2, x2[0], y)
        
        #x.append(child_1)
        #y.append(child_2)
    
    # Replace the old population with the new one
    #x = child_1
    #y=child_2
    # Apply mutation
    #mutation_rate = 0.002
    #population = randomly_mutate_population(population, mutation_rate)
    
    # Score best solution, and add to tracker
    
    scores = calc_fitness(x,y)
    best_score=numpy.max(scores)
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
        
    
    print("gen:",generation)
    
    #print("x:",x.shape)
    #print("y:",y)
    print(parents)
    print("best_score:",best_score)
    #print("avg_score",avg_score)
    
    print("parent_1",parent_1)
    print("parent_2",parent_2)
    #print("x1:",x1)
    #print("x2:",x2)
    parent_1=numpy.unique(parent_1)
    parent_1=parent_1.reshape(1,-1)
    #for j in range(i):
     #   if (j<0):
      #      parent_1=parent_1[j]
    parent_1=numpy.delete(parents,1)
    idx=numpy.where(scores == numpy.amax(scores))
print("co-ordinates:",idx)
print("best score over itenerations:",best_score_progress)   
print("average score over iteneration:",avg_score_progress)
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