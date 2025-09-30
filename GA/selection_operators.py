import random
import numpy as np

from individual import Individual
from population import Population

def tournament_selection(pop: Population, k: int = 3) -> Individual:
    candidates = random.sample(pop.individuals, min(k, len(pop.individuals)))
    if isinstance(candidates[0].fitness, dict):
        return max(candidates, key=lambda ind: ind.scalar_fitness).copy()
    else:
        return max(candidates, key=lambda ind: ind.fitness).copy()


def roulette_wheel_selection(pop: Population) -> Individual:
    if isinstance(pop.individuals[0].fitness, dict):
        fitnesses = [ind.scalar_fitness for ind in pop.individuals]
    else:
        fitnesses = [ind.fitness for ind in pop.individuals]
    
    min_fitness = min(fitnesses)
    shifted_fitness = [f - min_fitness + 1 for f in fitnesses]
    
    total_fitness = sum(shifted_fitness)
    if total_fitness == 0:
        return random.choice(pop.individuals).copy()
    
    probabilities = [f / total_fitness for f in shifted_fitness]
    selected = np.random.choice(pop.individuals, p=probabilities)
    return selected.copy()


def truncation_selection(pop: Population, top_percent: float = 0.5) -> Individual:
    sorted_pop = sorted(pop.individuals, key=lambda ind: ind.fitness, reverse=True)
    cutoff = max(1, int(len(sorted_pop) * top_percent))
    return random.choice(sorted_pop[:cutoff]).copy()