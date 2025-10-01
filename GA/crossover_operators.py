import random
from typing import Tuple

from individual import Individual

def order_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    size = len(p1.genome)
    if size <= 1:
        return p1.copy(), p2.copy()
    
    c1_genome, c2_genome = [-1] * size, [-1] * size
    
    start, end = sorted(random.sample(range(size), 2))
    
    c1_genome[start:end+1] = p1.genome[start:end+1]
    c2_genome[start:end+1] = p2.genome[start:end+1]
    
    def fill_remaining(child_genome, parent_genome):
        pointer = end + 1
        for gene in parent_genome[end+1:] + parent_genome[:end+1]:
            if gene not in child_genome:
                child_genome[pointer % size] = gene
                pointer += 1
    
    fill_remaining(c1_genome, p2.genome)
    fill_remaining(c2_genome, p1.genome)
    
    return Individual(c1_genome), Individual(c2_genome)


def pmx_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    size = len(p1.genome)
    if size <= 1:
        return p1.copy(), p2.copy()
    
    c1_genome, c2_genome = p1.genome[:], p2.genome[:]
    
    start, end = sorted(random.sample(range(size), 2))
    
    mapping1, mapping2 = {}, {}
    for i in range(start, end + 1):
        mapping1[p2.genome[i]] = p1.genome[i]
        mapping2[p1.genome[i]] = p2.genome[i]
    
    for i in range(size):
        if i < start or i > end:
            while c1_genome[i] in mapping1:
                c1_genome[i] = mapping1[c1_genome[i]]
            while c2_genome[i] in mapping2:
                c2_genome[i] = mapping2[c2_genome[i]]
    
    c1_genome[start:end+1] = p2.genome[start:end+1]
    c2_genome[start:end+1] = p1.genome[start:end+1]
    
    return Individual(c1_genome), Individual(c2_genome)


def cycle_crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    size = len(p1.genome)
    if size <= 1:
        return p1.copy(), p2.copy()
    
    c1_genome, c2_genome = [-1] * size, [-1] * size
    
    visited = [False] * size
    cycle = 0
    
    for i in range(size):
        if not visited[i]:
            current = i
            while not visited[current]:
                visited[current] = True
                if cycle % 2 == 0:
                    c1_genome[current] = p1.genome[current]
                    c2_genome[current] = p2.genome[current]
                else:
                    c1_genome[current] = p2.genome[current]
                    c2_genome[current] = p1.genome[current]
                
                value = p2.genome[current]
                try:
                    current = p1.genome.index(value)
                except ValueError:
                    break
            
            cycle += 1
    
    return Individual(c1_genome), Individual(c2_genome)

def arithmetic_crossover(p1: Individual, p2: Individual, alpha: float = 0.5) -> Tuple[Individual, Individual]:
    c1_genome = []
    c2_genome = []
    
    for g1, g2 in zip(p1.genome, p2.genome):
        c1_genome.append(alpha * g1 + (1 - alpha) * g2)
        c2_genome.append((1 - alpha) * g1 + alpha * g2)
    
    return Individual(c1_genome), Individual(c2_genome)


def blend_crossover(p1: Individual, p2: Individual, alpha: float = 0.5) -> Tuple[Individual, Individual]:
    c1_genome = []
    c2_genome = []
    
    for g1, g2 in zip(p1.genome, p2.genome):
        min_val, max_val = min(g1, g2), max(g1, g2)
        range_val = max_val - min_val
        
        lower = min_val - alpha * range_val
        upper = max_val + alpha * range_val
        
        c1_genome.append(random.uniform(lower, upper))
        c2_genome.append(random.uniform(lower, upper))
    
    return Individual(c1_genome), Individual(c2_genome)

def sbx_crossover(p1: Individual, p2: Individual, eta: float = 2.0, prob: float = 1.0):
    size = len(p1.genome)
    c1, c2 = [], []
    for i in range(size):
        if random.random() <= prob:
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1.append(0.5 * ((1 + beta) * p1.genome[i] + (1 - beta) * p2.genome[i]))
            c2.append(0.5 * ((1 - beta) * p1.genome[i] + (1 + beta) * p2.genome[i]))
        else:
            c1.append(p1.genome[i])
            c2.append(p2.genome[i])
    return Individual(c1), Individual(c2)