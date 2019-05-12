# Example usage
from genetic import *
import random
import pandas as pa
import sys
import itertools

df = pa.read_csv("data.csv", sep=";")
df = df[["gender", "age"]]
df2 = pa.get_dummies(df.reset_index(), columns=["gender"])

data = df2.values.tolist()
data = data[:100]

nb_iteration = 30

N_POPU = len(data)
N_GROUP = 2000
GROUP_SIZE = 10

def random_bit():
    return int(random.getrandbits(1))

def generate_individual(n_popu):
    size = int(n_popu/2)
    individual1 = [0] * size
    individual2 = [0] * size
    n_participants = int(GROUP_SIZE/2)

    bit_positions_first = random.sample(range(0, size), n_participants)
    for pos in bit_positions_first:
        individual1[pos] = 1

    bit_positions_second = random.sample(range(0, size), n_participants)
    for pos in bit_positions_second:
        individual2[pos] = 1

    return individual1 + individual2

def generate_population(n_group, n_popu):
    return [generate_individual(n_popu) for x in range(n_group)]

def score_group(group):
    individuals = np.array([data[i] for i in range(len(group)) if group[i] == 1])

    variance_columns = np.var(individuals, axis=0)

    variance_columns = np.delete(variance_columns, 0)

    n_items = individuals.size

    score = np.mean(variance_columns)

    if n_items > GROUP_SIZE:
        return score
    return score

def score_population(population, avg=True):
    sorted_popu = sorted([(score_group(group),group) for group in population])

    if avg:
        return np.mean([x[0] for x in sorted_popu])
    return [x[1] for x in sorted_popu]

def evolve(pop, data, retain=0.2, random_select=0.05, mutate=0.01):
    #score each group
    sorted_groups = score_population(pop, avg=False)

    #keep the parents
    retain_length = int(len(sorted_groups)*0.2)
    parents = sorted_groups[:retain_length]

    parents = [tuple(p) for p in parents]
    parents = list(set(parents))
    parents = [list(p) for p in parents]

    # promote genetic diversity
    for group in sorted_groups[retain_length:]:
        if random_select > random.random():
            parents.append(group)
    
    # mutate some individuals
    for group in parents:
        if mutate > random.random():
            size = int(len(group)/2)
            pos_to_mutate_a = random.randint(0, size-1)
            pos_to_mutate_b = random.randint(0, size-1)
            value_a = group[pos_to_mutate_a]
            value_b = group[pos_to_mutate_b]
            group[pos_to_mutate_a] = value_b
            group[pos_to_mutate_b] = value_a

    #make children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length

    children = []
    while len(children) < desired_length:
        male = choice(parents)
        female = choice(parents)
        if male != female:
            half = int(len(male) / 2)
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)

    #done!
    return parents

population = generate_population(N_GROUP, N_POPU)


score_history = [score_population(population),]
for i in range(nb_iteration):
    print(i)
    population = evolve(population, data)
    score_history.append(score_population(population))

for datum in score_history:
   print(datum)

for group in population[:10]:
    individuals = [i for i in range(len(group)) if group[i] == 1]

    subdf = df.ix[individuals]
    print(subdf)
