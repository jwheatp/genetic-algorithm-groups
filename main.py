# Example usage
import random
import pandas as pa
import sys
import itertools
import numpy as np

df = pa.read_csv("data.csv", sep=";")
df = df[["gender", "age", "company"]]

WEIGHTS = {
    "gender": {
        "weight": 0.6,
        "condition": "equality_elements"
    },
    "age": {
        "weight": 1
    },
    "company": {
        "weight": 0.4
    }
}

df2 = pa.get_dummies(df.reset_index(), columns=["gender", "company"])

data = df2.values.tolist()
data = data[:100]

N_ITERATIONS = 30
N_POPU = len(data)
N_GROUP = 2000
GROUP_SIZE = 10

def generate_group(n_popu):
    """
    Generate group vector.
    Size matters here.
    We have a population of size n_popu.
    So we'll get a vector of size n_popu with 0s and 1s:
    - 0 if the individual does not belong to the group
    - 1 otherwise
    This insure an individual won't appear several times in the same group.

    But as we want fixed-size groups, we use a trick:
    We initialize our group vector by spliting in two parts
    With two individuals (two 1s) randomly selected in the 1st part, and two in the second part.
    """
    part_size = int(n_popu/2) # length of each part of the vector
    n_part_participants = int(GROUP_SIZE/2) # how many 1s in each part
    
    individual = []
    for part in [0, 1]:
        part = [0] * part_size
        bit_positions = random.sample(range(0, part_size), n_part_participants)
        for pos in bit_positions:
            part[pos] = 1

        individual += part
    return individual

def generate_population(n_group, n_popu):
    """
    Generate a population by generating each group.
    """
    return [generate_group(n_popu) for x in range(n_group)]

def score_group(group):
    """
    Score a group.
    A group is a list of 0s and 1s.
    [0 0 1 0 1 0]
    Each of these bits says if the nth individual belongs to this group.
    An individual is identified by its index in the list, the same than in the data list.
    """
    # first, get each data vector given the index in the list
    # ie we get the enriched list of individuals
    individuals = np.array([data[i] for i in range(len(group)) if group[i] == 1])

    # magic happens here
    variance_columns = np.var(individuals, axis=0)
    variance_columns = np.delete(variance_columns, 0)

    # finally we compute the average variance as the score
    score = np.mean(variance_columns)

    return score

def score_population(population, avg=True):
    """
    Score a population as the mean score of all groups.
    """
    # compute the score for each group
    # we also sort the list, useful for later
    sorted_popu = sorted([(score_group(group),group) for group in population])

    # either we want the average score
    if avg:
        return np.mean([x[0] for x in sorted_popu])

    # or all the scores
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
        male = random.choice(parents)
        female = random.choice(parents)
        if male != female:
            half = int(len(male) / 2)
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)

    #done!
    return parents

population = generate_population(N_GROUP, N_POPU)

score_history = [score_population(population),]
for i in range(N_ITERATIONS):
    print(i)
    population = evolve(population, data)
    score_history.append(score_population(population))

for datum in score_history:
   print(datum)

for group in population[:10]:
    individuals = [i for i in range(len(group)) if group[i] == 1]

    subdf = df.ix[individuals]
    print(subdf)
