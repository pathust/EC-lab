import random
from typing import List, Optional

from individual import Individual

class Population:
    def __init__(self, size: int, problems=None):
        self.individuals: List[Individual] = []
        self.size = size
        self.problems = problems if problems else []
        
    def initialize(self, genome_generator):
        self.individuals = [Individual(genome_generator()) for _ in range(self.size)]
        
        if len(self.problems) == 1:
            for ind in self.individuals:
                ind.evaluate(self.problems[0])
        elif len(self.problems) > 1:
            for ind in self.individuals:
                ind.skill_factor = random.randint(0, len(self.problems) - 1)
                ind.evaluate(self.problems[ind.skill_factor], ind.skill_factor)
    
    def get_best(self, task_id: Optional[int] = None):
        if task_id is not None:
            valid_inds = [ind for ind in self.individuals if task_id in ind.fitness]
            if valid_inds:
                return max(valid_inds, key=lambda ind: ind.fitness[task_id])
        else:
            if isinstance(self.individuals[0].fitness, dict):
                return max(self.individuals, key=lambda ind: ind.scalar_fitness)
            else:
                return max(self.individuals, key=lambda ind: ind.fitness)
    
    def get_worst(self):
        if isinstance(self.individuals[0].fitness, dict):
            return min(self.individuals, key=lambda ind: ind.scalar_fitness)
        else:
            return min(self.individuals, key=lambda ind: ind.fitness)