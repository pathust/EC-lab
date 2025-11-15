from typing import Any, List, Optional
import copy

class Individual:
    def __init__(self, genome: List, skill_factor: Optional[int] = None):
        self.genome = genome
        self.fitness = {}
        # For MFEA
        self.skill_factor = skill_factor
        
    def evaluate(self, problem, task_id: Optional[int] = None):
        if task_id is not None:
            self.fitness[task_id] = problem.evaluate(self)
            return self.fitness[task_id]
        else:
            self.fitness = problem.evaluate(self)
            return self.fitness
    
    def copy(self):
        new_ind = Individual(copy.deepcopy(self.genome))
        new_ind.fitness = copy.deepcopy(self.fitness)
        new_ind.skill_factor = self.skill_factor
        return new_ind