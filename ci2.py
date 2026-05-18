import random
import matplotlib.pyplot as plt

def fitness(x):
    return x * (1 - x)

def initialize_population(pop_size):
    return [random.uniform(0,1) for _ in range(pop_size)]

def select(population, num):
    return sorted(population, key=fitness, reverse=True)[:num]

def clone(selected, cloning_factor):
    results = []
    for i in selected:
        for _ in range(cloning_factor):
            results.append(i)
    return results

def mutate(clones):
    results = []
    for i in clones:
        cloning_factor = random.uniform(-1,1)
        new = i + cloning_factor
        new = max(0, min(1, new))
        results.append(new)
        
    return results

def replace(population, mutated):
    combined = population + mutated
    return sorted(combined, key=fitness, reverse=True)[:len(population)]


pop_size = 10
num_generations = 20
cloning_factor = 2

pop  = initialize_population(pop_size)

results = [(max(pop), fitness(max(pop)))]

for i in range(num_generations):
    selected = select(pop, 4)
    clones =  clone(selected, cloning_factor)
    mutated = mutate(clones)
    pop = replace(pop, mutated)
    results.append((max(pop), fitness(max(pop))))

print(f"Best Solution: {results[-1][0]}")
print(f"Best Solution Fitness: {results[-1][1]}")
    
plt.plot([x[1] for x in results])
plt.show()