import random

from individual import Individual

def swap_mutation(ind: Individual, prob: float = 0.1):
    if random.random() < prob and len(ind.genome) >= 2:
        i, j = random.sample(range(len(ind.genome)), 2)
        ind.genome[i], ind.genome[j] = ind.genome[j], ind.genome[i]

def inversion_mutation(ind: Individual, prob: float = 0.1):
    if random.random() < prob and len(ind.genome) >= 2:
        start, end = sorted(random.sample(range(len(ind.genome)), 2))
        ind.genome[start:end+1] = ind.genome[start:end+1][::-1]


def insertion_mutation(ind: Individual, prob: float = 0.1):
    if random.random() < prob and len(ind.genome) >= 2:
        i, j = random.sample(range(len(ind.genome)), 2)
        gene = ind.genome.pop(i)
        ind.genome.insert(j, gene)


def gaussian_mutation(ind: Individual, prob: float = 0.1, sigma: float = 0.1):
    for i in range(len(ind.genome)):
        if random.random() < prob:
            ind.genome[i] += random.gauss(0, sigma)
            ind.genome[i] = max(0, min(1, ind.genome[i]))


def uniform_mutation(ind: Individual, prob: float = 0.1):
    for i in range(len(ind.genome)):
        if random.random() < prob:
            ind.genome[i] = random.random()

def polynomial_mutation(ind: Individual, prob: float = 0.1, eta: float = 20):
    for i in range(len(ind.genome)):
        if random.random() < prob:
            u = random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            ind.genome[i] += delta
            # thường ràng buộc [0,1] hoặc theo domain
            ind.genome[i] = max(0, min(1, ind.genome[i]))