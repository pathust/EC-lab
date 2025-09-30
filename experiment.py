from typing import List, Callable
import numpy as np
import os
import json
from tqdm import tqdm
from problem import Problem, TSPProblem, KnapsackProblem
from algorithm import GeneticAlgorithm, MFEA

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class Experiment:
    def __init__(self, problem, seeds=range(30), **algo_kwargs):
        self.problem = problem
        self.seeds = seeds
        self.algo_kwargs = algo_kwargs
        self.results = []
        self.best_founds = []
        self.stats = None
    
    def get_result_dir(self):
        """Generate organized result directory path"""
        problem_name = self.problem.__class__.__name__.lower().replace('problem', '')
        
        def get_op_name(key, default_name):
            op = self.algo_kwargs.get(key)
            if isinstance(op, Callable) and hasattr(op, '__name__'):
                 return (op.__name__.
                     replace('_crossover', '').
                     replace('_mutation', '').
                     replace('_selection', '')
                 )
            elif isinstance(op, dict):
                 return '_'.join(v.__name__ for v in op.values())
            return default_name
        
        crossover_name = get_op_name('crossover_op', 'unknown')
        mutation_name = get_op_name('mutation_op', 'unknown')
        selection_name = get_op_name('selection_op', 'unknown')
        
         # Construct directory path
        result_dir = f"results/ga/{crossover_name}/{mutation_name}/{selection_name}/{problem_name}"
        return result_dir
    
    def run(self, verbose=True):
        self.results = []
        self.best_founds = []
        
        result_dir = self.get_result_dir()
        os.makedirs(result_dir, exist_ok=True)
        
        for i, seed in enumerate(tqdm(self.seeds, desc=f"Running {result_dir.split('/')[-1]} experiment")):
            algorithm = GeneticAlgorithm(
                problem=self.problem, 
                seed=seed, 
                **self.algo_kwargs
            )
            
            algorithm.initialize()
            algorithm.run(verbose=False)
            
            best = algorithm.population.get_best()
            self.results.append({
                'seed': seed,
                'best_fitness': best.fitness,
                'best_genome': best.genome,
                'history': algorithm.history
            })
            self.best_founds.append(best.fitness)
            
            if verbose and i % 10 == 0:
                print(f"  Completed {i+1}/{len(self.seeds)} runs")
        
        self.compute_statistics()
        
        self.save_results(result_dir)
                
        return self.results
    
    def compute_statistics(self):
        fitness_values = [r['best_fitness'] for r in self.results]
        
        self.stats = {
            'mean': np.mean(fitness_values),
            'min': np.min(fitness_values),
            'max': np.max(fitness_values),
            'median': np.median(fitness_values),
        }
        
        return self.stats
    
    def save_results(self, result_dir):
        # Save JSON results
        serializable_kwargs = {k: v.__name__ if hasattr(v, '__name__') else v 
                               for k, v in self.algo_kwargs.items()}
        
        result_data = {
            'experiment_info': {
                'problem': self.problem.__class__.__name__,
                'problem_size': getattr(self.problem, 'n_cities', getattr(self.problem, 'n_items', 'unknown')),
                'encoding': getattr(self.problem, 'encoding', 'unknown'),
                'seeds': list(self.seeds),
                'num_runs': len(self.seeds)
            },
            'algorithm_settings': serializable_kwargs,
            'statistics': self.stats,
            'detailed_results': [
                {
                    'seed': r['seed'],
                    'best_fitness': r['best_fitness'],
                    'best_genome': r['best_genome']
                } for r in self.results
            ],
            'best_run_history': max(self.results, key=lambda x: x['best_fitness'])['history'],
        }
        
        with open(f"{result_dir}/experiment_results.json", 'w') as f:
            json.dump(result_data, f, indent=4, cls=NumpyEncoder)
        
        print(f"✓ Results saved to {result_dir}")

class MFEAExperiment:
    def __init__(self, problems: List[Problem], seeds=range(30), **algo_kwargs):
        self.problems = problems
        self.K = len(problems)
        self.seeds = seeds
        self.algo_kwargs = algo_kwargs
        self.results = []
        self.stats = {}

    def get_result_dir(self):
        problem_names = "_".join([p.__class__.__name__.lower().replace('problem', '') 
                                 for p in self.problems])
        
        def get_op_name(key, default_name):
            op = self.algo_kwargs.get(key)
            if isinstance(op, Callable) and hasattr(op, '__name__'):
                 return (op.__name__.
                     replace('_crossover', '').
                     replace('_mutation', '').
                     replace('_selection', '')
                 )
            elif isinstance(op, dict):
                 return '_'.join(v.__name__ for v in op.values())
            return default_name

        crossover_name = get_op_name('cross_crossover_op', 'arithmetic')
        selection_name = get_op_name('selection_op', 'tournament')
        
        result_dir = f"results/mfea/{crossover_name}_{selection_name}_{problem_names}"
        return result_dir

    def run(self, verbose=True):
        self.results = []
        result_dir = self.get_result_dir()
        
        os.makedirs(result_dir, exist_ok=True)

        for seed in tqdm(self.seeds, desc=f"Running MFEA on {result_dir.split('/')[-1]}"):
            mfea = MFEA(problems=self.problems, seed=seed, **self.algo_kwargs)
            mfea.initialize()
            solutions = mfea.run(verbose=False) 
            
            run_result = {'seed': seed, 'solutions': solutions, 'history': mfea.history}
            for task_id, sol in solutions.items():
                run_result[f'task_{task_id}_fitness'] = sol['fitness']
            
            self.results.append(run_result)

        self.compute_statistics()
        self.save_results(result_dir)
        
        if verbose:
            print(f"\nHoàn thành thí nghiệm. Kết quả được lưu tại: {result_dir} (nếu có quyền ghi).")

        return self.results

    def compute_statistics(self):
        for task_id in range(self.K):
            fitness_key = f'task_{task_id}_fitness'
            fitness_values = [r.get(fitness_key, -float('inf')) for r in self.results] 
            
            if fitness_values:
                stats_data = {
                    'mean': np.mean(fitness_values),
                    'std': np.std(fitness_values),
                    'min': np.min(fitness_values),
                    'max': np.max(fitness_values),
                    'median': np.median(fitness_values),
                }
                if isinstance(self.problems[task_id], TSPProblem):
                    stats_data['best_tour_length'] = -stats_data['max'] 
                    stats_data['mean_tour_length'] = -stats_data['mean']
                self.stats[f'task_{task_id}'] = stats_data

    def save_results(self, result_dir):
        serializable_kwargs = {k: v.__name__ if hasattr(v, '__name__') else v 
                               for k, v in self.algo_kwargs.items()}
        
        result_data = {
            'experiment_info': {
                'algorithm': 'MFEA',
                'num_tasks': self.K,
                'problems': [p.__class__.__name__ for p in self.problems],
                'num_runs': len(self.seeds)
            },
            'algorithm_settings': serializable_kwargs,
            'statistics': self.stats,
            'detailed_results': self.results
        }
        
        with open(f"{result_dir}/mfea_results.json", 'w') as f:
            json.dump(result_data, f, indent=4, cls=NumpyEncoder)