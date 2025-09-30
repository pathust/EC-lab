import random
from typing import Callable, Dict, Tuple
import numpy as np

from problem import *
from population import Population
from GA.crossover_operators import order_crossover, arithmetic_crossover
from GA.mutation_operators import swap_mutation, gaussian_mutation
from GA.selection_operators import tournament_selection

class GeneticAlgorithm:
    def __init__(self, 
                 problem: Problem,
                 seed: int,
                 pop_size: int = 100,
                 generations: int = 500,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 elitism: int = 2,
                 crossover_op: Callable = None,
                 mutation_op: Callable = None,
                 selection_op: Callable = None,
                 crossover_params: Dict = None,
                 mutation_params: Dict = None,
                 selection_params: Dict = None):
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        
        self.population = Population(size=pop_size, problems=[problem])
        
        if problem.encoding == 'permutation':
            self.crossover = crossover_op or order_crossover
            self.mutation = mutation_op or swap_mutation
        else:
            self.crossover = crossover_op or arithmetic_crossover
            self.mutation = mutation_op or gaussian_mutation
            
        self.selection = selection_op or tournament_selection
        
        self.crossover_params = crossover_params or {}
        self.mutation_params = mutation_params or {'prob': mutation_prob}
        self.selection_params = selection_params or {}
        
        self.history = {
            'best_fitness': [],
            'best_genome': []
        }
    
    def initialize(self):
        self.population.initialize(self.problem.generate_random_genome)
    
    def run(self, verbose: bool = True):
        for gen in range(self.generations):
            new_individuals = []
            
            if self.elitism > 0:
                sorted_pop = sorted(self.population.individuals, 
                                  key=lambda ind: ind.fitness, reverse=True)
                for i in range(min(self.elitism, len(sorted_pop))):
                    new_individuals.append(sorted_pop[i].copy())
            
            while len(new_individuals) < self.pop_size:
                parent1 = self.selection(self.population, **self.selection_params)
                parent2 = self.selection(self.population, **self.selection_params)
                
                if random.random() < self.crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2, **self.crossover_params)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                self.mutation(child1, **self.mutation_params)
                self.mutation(child2, **self.mutation_params)
                
                child1.evaluate(self.problem)
                child2.evaluate(self.problem)
                
                new_individuals.extend([child1, child2])
            
            self.population.individuals = new_individuals[:self.pop_size]
            
            best = self.population.get_best()
            
            self.history['best_fitness'].append(best.fitness)
            self.history['best_genome'].append(best.genome)
            
            if verbose and gen % 50 == 0:
                print(f"Generation {gen}: Best = {best.fitness:.2f}")
        
        if verbose:
            print(f"Final best fitness: {self.population.get_best().fitness:.2f}")
    
    def get_best_solution(self):
        best = self.population.get_best()
        return best.genome, best.fitness
    
class MFEA:
    def __init__(self,
                 problems: List[Problem],
                 seed: int,
                 pop_size: int = 100,
                 generations: int = 500,
                 rmp: float = 0.3,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 elitism: int = 2,
                 crossover_ops: Dict[int, Callable] = None,
                 mutation_ops: Dict[int, Callable] = None,
                 cross_crossover_op: Callable = None,
                 selection_op: Callable = None,
                 op_params: Dict = None):

        random.seed(seed)
        np.random.seed(seed)

        self.problems = problems
        self.K = len(problems)
        self.pop_size = pop_size
        self.generations = generations
        self.rmp = rmp
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism = elitism
        
        self.op_params = op_params or {}

        self.populations = [Population(size=pop_size) for _ in problems]

        self._setup_operators(crossover_ops, mutation_ops, cross_crossover_op, selection_op)

        self.history = {f'task_{i}': {'best_fitness': []} for i in range(self.K)}
        
        self.genome_info = {
            i: {'type': p.encoding, 'length': p.get_genome_length()}
            for i, p in enumerate(self.problems)
        }

    def initialize(self):
        for i, pop in enumerate(self.populations):
            for _ in range(self.pop_size):
                genome = self.problems[i].generate_random_genome()
                ind = Individual(genome, skill_factor=i)
                pop.individuals.append(ind)
                
            for ind in pop.individuals:
                ind.evaluate(self.problems[i])



    def _setup_operators(self, crossover_ops, mutation_ops, cross_crossover_op, selection_op):
        self.crossover_ops = crossover_ops or {}
        self.mutation_ops = mutation_ops or {}

        for i, problem in enumerate(self.problems):
            if i not in self.crossover_ops:
                self.crossover_ops[i] = (order_crossover if problem.encoding == 'permutation'
                                         else arithmetic_crossover)
            if i not in self.mutation_ops:
                self.mutation_ops[i] = (swap_mutation if problem.encoding == 'permutation'
                                        else gaussian_mutation)

        self.cross_crossover_op = cross_crossover_op or arithmetic_crossover
        self.selection = selection_op or tournament_selection
    
    def _unify_genomes(self, g1: List, t1_id: int, g2: List, t2_id: int) -> Tuple[List, List, Dict]:
        info1 = self.genome_info[t1_id]
        info2 = self.genome_info[t2_id]
        max_len = max(info1['length'], info2['length'])

        is_both_perm = info1['type'] == 'permutation' and info2['type'] == 'permutation'
        unified_type = 'permutation' if is_both_perm else 'real'
        unified_rep = {'type': unified_type, 'length': max_len}

        def convert(genome, from_info, to_rep):
            from_type, from_len = from_info['type'], len(genome)
            to_type, to_len = to_rep['type'], to_rep['length']
            
            temp_genome = list(genome)

            if from_type == 'permutation' and to_type == 'real':
                temp_genome = [temp_genome.index(i) / (from_len - 1) for i in range(from_len)]
            elif from_type == 'real' and to_type == 'permutation':
                 temp_genome = [x[0] for x in sorted(enumerate(temp_genome), key=lambda item: item[1])]

            if len(temp_genome) < to_len:
                if to_type == 'permutation':
                    used = set(temp_genome)
                    temp_genome.extend([i for i in range(to_len) if i not in used])
                else: # real
                    temp_genome.extend(np.random.uniform(0, 1, to_len - len(temp_genome)))
            
            return temp_genome[:to_len]

        unified1 = convert(g1, info1, unified_rep)
        unified2 = convert(g2, info2, unified_rep)
        return unified1, unified2, unified_rep

    def _deunify_genome(self, genome: List, to_task_id: int, unified_rep: Dict) -> List:
        to_info = self.genome_info[to_task_id]
        to_type, to_len = to_info['type'], to_info['length']
        from_type = unified_rep['type']
        
        temp_genome = list(genome)

        if from_type == 'real' and to_type == 'permutation':
            mapped_indices = [x[0] for x in sorted(enumerate(temp_genome), key=lambda item: item[1])]
            perm = []
            for i in mapped_indices:
                if len(perm) < to_len:
                    perm.append(i % to_len)
            
            unique_perm = []
            seen = set()
            for val in perm:
                if val not in seen:
                    unique_perm.append(val)
                    seen.add(val)
            
            if len(unique_perm) < to_len:
                unique_perm.extend([i for i in range(to_len) if i not in seen])
                
            temp_genome = unique_perm[:to_len]
        
        if len(temp_genome) > to_len:
            temp_genome = temp_genome[:to_len]
        
        if len(temp_genome) < to_len:
            if to_type == 'permutation':
                seen = set(temp_genome)
                temp_genome.extend([i for i in range(to_len) if i not in seen])
            else: # real
                 temp_genome.extend(np.random.uniform(0, 1, to_len - len(temp_genome)))

        return temp_genome

    def _mate(self, parent1: Individual, parent2: Individual, t1_id: int, t2_id: int) -> Tuple[Individual, Individual]:
        if t1_id == t2_id: 
            op = self.crossover_ops[t1_id]
            mut_op = self.mutation_ops[t1_id]
            
            if random.random() < self.crossover_prob:
                child1, child2 = op(parent1, parent2, **self.op_params.get('crossover', {}))
            else:
                child1, child2 = parent1.copy(), parent2.copy() 
            
            mut_op(child1, **self.op_params.get('mutation', {}))
            mut_op(child2, **self.op_params.get('mutation', {}))
            
            return child1, child2
        else:
            uni1, uni2, uni_rep = self._unify_genomes(parent1.genome, t1_id, parent2.genome, t2_id)
            
            if random.random() < self.crossover_prob:
                temp_c1, temp_c2 = self.cross_crossover_op(Individual(uni1), Individual(uni2), **self.op_params.get('cross_crossover', {}))
                c1_uni, c2_uni = temp_c1.genome, temp_c2.genome
            else:
                c1_uni, c2_uni = uni1[:], uni2[:] 

            c1_gen = self._deunify_genome(c1_uni, t1_id, uni_rep)
            c2_gen = self._deunify_genome(c2_uni, t2_id, uni_rep)
            
            child1, child2 = Individual(c1_gen), Individual(c2_gen)
            
            self.mutation_ops[t1_id](child1, **self.op_params.get('mutation', {}))
            self.mutation_ops[t2_id](child2, **self.op_params.get('mutation', {}))
            
            return child1, child2

    def run(self, verbose: bool = True):
        for gen in range(self.generations):
            offspring_populations = [[] for _ in range(self.K)]

            for task_id in range(self.K):
                pop = self.populations[task_id]
                
                while len(offspring_populations[task_id]) < self.pop_size:
                    parent1 = self.selection(pop, **self.op_params.get('selection', {}))
                    
                    if not pop.individuals:
                        continue 

                    if random.random() < self.rmp and self.K > 1:
                        other_task = random.choice([t for t in range(self.K) if t != task_id])
                        pop2 = self.populations[other_task]
                        
                        if not pop2.individuals:
                             continue
                             
                        parent2 = self.selection(pop2, **self.op_params.get('selection', {}))
                        child1, child2 = self._mate(parent1, parent2, task_id, other_task)
                        
                        child1.evaluate(self.problems[task_id])
                        child2.evaluate(self.problems[other_task])
                        
                        offspring_populations[task_id].append(child1)
                        if len(offspring_populations[other_task]) < self.pop_size:
                            offspring_populations[other_task].append(child2)
                    else:
                        parent2 = self.selection(pop, **self.op_params.get('selection', {}))
                        child1, child2 = self._mate(parent1, parent2, task_id, task_id)

                        child1.evaluate(self.problems[task_id])
                        child2.evaluate(self.problems[task_id])
                        
                        offspring_populations[task_id].extend([child1, child2])
                        
                    if len(offspring_populations[task_id]) > self.pop_size:
                        offspring_populations[task_id] = offspring_populations[task_id][:self.pop_size]

            for i in range(self.K):
                for ind in offspring_populations[i]:
                    ind.skill_factor = i
                    
                combined_pop = self.populations[i].individuals + offspring_populations[i]
                combined_pop.sort(key=lambda x: x.fitness, reverse=True) 
                self.populations[i].individuals = combined_pop[:self.pop_size]

            if verbose and (gen % 50 == 0 or gen == self.generations - 1):
                if gen % 50 == 0:
                    print(f"\nThế hệ {gen}:")
                for task_id in range(self.K):
                    best = self.populations[task_id].get_best()
                    if gen % 50 == 0:
                       self.history[f'task_{task_id}']['best_fitness'].append(best.fitness)
                       print(f"  Task {task_id}: Fitness tốt nhất = {best.fitness:.4f}")
            
            if gen == self.generations - 1:
                 for task_id in range(self.K):
                    best = self.populations[task_id].get_best()
                    if not self.history[f'task_{task_id}']['best_fitness'] or self.history[f'task_{task_id}']['best_fitness'][-1] != best.fitness:
                        self.history[f'task_{task_id}']['best_fitness'].append(best.fitness)

        return self.get_best_solutions()

    def get_best_solutions(self) -> Dict:
        return {
            task_id: {
                'genome': (best := self.populations[task_id].get_best()).genome,
                'fitness': best.fitness,
                'decoded': self.problems[task_id].decode(best.genome)
            }
            for task_id in range(self.K)
        }