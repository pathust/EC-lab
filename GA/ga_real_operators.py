import random
import math
import numpy as np

# ==================== TSP Problem ====================
class TSP:
    def __init__(self, file_name, opt_file_name):
        self.vertices = []
        self.dist_matrix = []
        self.file_name = file_name
        self.opt_file_name = opt_file_name
        self.size = 0
        self.best_individual = None
        self.read_file()
        self.compute_dist_matrix()
        self.get_best_individual_answer()

    def read_file(self):
        with open(self.file_name, 'r') as f:
            for line in f:
                line = line.strip()
                # Bỏ qua các dòng header và dòng trống
                if not line or ':' in line or line.startswith('NAME') or line.startswith('TYPE') or \
                line.startswith('COMMENT') or line.startswith('DIMENSION') or \
                line.startswith('EDGE_WEIGHT_TYPE') or line == 'NODE_COORD_SECTION' or line == 'EOF':
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # parts[0] là index, parts[1] là x, parts[2] là y
                        x, y = float(parts[1]), float(parts[2])
                        self.vertices.append((x, y))
                    except ValueError:
                        continue
    
    def compute_dist_matrix(self):
        n = len(self.vertices)
        self.size = n
        self.dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x1, y1 = self.vertices[i]
                x2, y2 = self.vertices[j]
                self.dist_matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    def calculate_fitness(self, individual):
        fitness = 0
        for i in range(self.size - 1):
            fitness += self.dist_matrix[individual.genome[i]][individual.genome[i + 1]]
        fitness += self.dist_matrix[individual.genome[-1]][individual.genome[0]]
        return fitness
    
    def get_best_individual_answer(self):
        self.best_individual = {"fitness": float('inf'), "genome": []}
        try:
            with open(self.opt_file_name, "r") as f:
                for line in f:
                    vertice = int(line.strip())
                    self.best_individual["genome"].append(vertice - 1)
            
            fitness = 0
            for i in range(len(self.best_individual["genome"]) - 1):
                fitness += self.dist_matrix[self.best_individual["genome"][i]][self.best_individual["genome"][i + 1]]
            fitness += self.dist_matrix[self.best_individual["genome"][-1]][self.best_individual["genome"][0]]
            self.best_individual["fitness"] = fitness
        except:
            pass

# ==================== Individual ====================
class Individual:
    def __init__(self, num):
        self.size = num
        self.genome = [random.random() for _ in range(num)]
        self.fitness = float('inf')
        self.tour = []

    def copy(self):
        new_individual = Individual(self.size)
        new_individual.genome = self.genome.copy()
        new_individual.fitness = self.fitness
        new_individual.tour = self.tour.copy()
        return new_individual

    def decode(self):
        """Decode real-valued genome to permutation"""
        return sorted(range(len(self.genome)), key=lambda i: self.genome[i])

# ==================== Crossover Operators ====================
class CrossoverOperators:
    @staticmethod
    def sbx_crossover(parent1, parent2, eta_c=20.0):
        """Simulated Binary Crossover"""
        size = len(parent1.genome)
        child1_genome = []
        child2_genome = []
        
        for i in range(size):
            if random.random() <= 0.5:
                p1 = parent1.genome[i]
                p2 = parent2.genome[i]
                
                if abs(p1 - p2) > 1e-14:
                    u = random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                    
                    c1 = 0.5 * ((p1 + p2) - beta * abs(p1 - p2))
                    c2 = 0.5 * ((p1 + p2) + beta * abs(p1 - p2))
                    
                    child1_genome.append(c1)
                    child2_genome.append(c2)
                else:
                    child1_genome.append(p1)
                    child2_genome.append(p2)
            else:
                child1_genome.append(parent1.genome[i])
                child2_genome.append(parent2.genome[i])
        
        c1 = Individual(size)
        c2 = Individual(size)
        c1.genome = child1_genome
        c2.genome = child2_genome
        return c1, c2
    
    @staticmethod
    def blend_crossover(parent1, parent2, alpha=0.5):
        """Blend Crossover (BLX-α)"""
        size = len(parent1.genome)
        child1_genome = []
        child2_genome = []
        
        for g1, g2 in zip(parent1.genome, parent2.genome):
            cmin, cmax = min(g1, g2), max(g1, g2)
            I = cmax - cmin
            lower = cmin - alpha * I
            upper = cmax + alpha * I
            child1_genome.append(random.uniform(lower, upper))
            child2_genome.append(random.uniform(lower, upper))
        
        c1 = Individual(size)
        c2 = Individual(size)
        c1.genome = child1_genome
        c2.genome = child2_genome
        return c1, c2
    
    @staticmethod
    def arithmetic_crossover(parent1, parent2, alpha=0.5):
        """Arithmetic Crossover"""
        size = len(parent1.genome)
        child1_genome = []
        child2_genome = []
        
        for g1, g2 in zip(parent1.genome, parent2.genome):
            child1_genome.append(alpha * g1 + (1 - alpha) * g2)
            child2_genome.append((1 - alpha) * g1 + alpha * g2)
        
        c1 = Individual(size)
        c2 = Individual(size)
        c1.genome = child1_genome
        c2.genome = child2_genome
        return c1, c2

# ==================== Mutation Operators ====================
class MutationOperators:
    @staticmethod
    def polynomial_mutation(individual, eta_m=20.0, pm=0.1):
        """Polynomial Mutation"""
        for i in range(len(individual.genome)):
            if random.random() < pm:
                y = individual.genome[i]
                u = random.random()
                
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                
                individual.genome[i] = y + delta
    
    @staticmethod
    def gaussian_mutation(individual, mutation_strength=0.1, pm=0.1):
        """Gaussian Mutation"""
        for i in range(len(individual.genome)):
            if random.random() < pm:
                individual.genome[i] += random.gauss(0, mutation_strength)
    
    @staticmethod
    def uniform_mutation(individual, low=-0.5, high=0.5, pm=0.1):
        """Uniform Mutation"""
        for i in range(len(individual.genome)):
            if random.random() < pm:
                individual.genome[i] += random.uniform(low, high)

# ==================== Selection Operators ====================
class SelectionOperators:
    @staticmethod
    def tournament_selection(population, tournament_size=3):
        """Tournament Selection"""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)
    
    @staticmethod
    def roulette_selection(population):
        """Roulette Wheel Selection (Fitness Proportionate)"""
        # Convert to maximization problem
        max_fitness = max(ind.fitness for ind in population)
        min_fitness = min(ind.fitness for ind in population)
        
        if max_fitness == min_fitness:
            return random.choice(population)
        
        # Inverse fitness for minimization
        inverse_fitness = [max_fitness - ind.fitness + 1 for ind in population]
        total_fitness = sum(inverse_fitness)
        
        if total_fitness == 0:
            return random.choice(population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind, fit in zip(population, inverse_fitness):
            current += fit
            if current >= pick:
                return ind
        
        return population[-1]
    
    @staticmethod
    def truncation_selection(population, truncation_rate=0.5):
        """Truncation Selection"""
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        truncation_size = max(1, int(len(population) * truncation_rate))
        selected_pool = sorted_pop[:truncation_size]
        return random.choice(selected_pool)

# ==================== GA Algorithm ====================
class GeneticAlgorithm:
    def __init__(self, problem, seed, crossover_op, mutation_op, selection_op,
                 pop_size=100, generations=500, pc=0.9, pm=0.1):
        random.seed(seed)
        np.random.seed(seed)
        
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        self.selection_op = selection_op
        
        self.population = []
        self.best_individual = None
        self.fitness_history = []
    
    def initialize(self):
        """Initialize population"""
        self.population = []
        for _ in range(self.pop_size):
            genome = [random.random() for _ in range(self.problem.size)]
            individual = Individual(self.problem.size)
            individual.genome = genome
            self.population.append(individual)
    
    def evaluate(self):
        """Evaluate population and update best"""
        for individual in self.population:
            tour = individual.decode()
            individual.tour = tour
            
            total_distance = 0
            for i in range(len(tour)):
                current_city = tour[i]
                next_city = tour[(i + 1) % len(tour)]
                total_distance += self.problem.dist_matrix[current_city][next_city]
            
            individual.fitness = total_distance
        
        current_best = min(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best.copy()
    
    def evolve(self):
        """Evolution loop"""
        self.initialize()
        self.evaluate()
        self.fitness_history.append(self.best_individual.fitness)
        
        for generation in range(self.generations):
            new_population = []
            
            # Elitism: keep best individual
            best = min(self.population, key=lambda ind: ind.fitness)
            new_population.append(best.copy())
            
            # Create offspring
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.selection_op(self.population)
                parent2 = self.selection_op(self.population)
                
                # Crossover
                if random.random() < self.pc:
                    child1, child2 = self.crossover_op(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                self.mutation_op(child1, pm=self.pm)
                self.mutation_op(child2, pm=self.pm)
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.pop_size]
            self.evaluate()
            
            # Track history
            self.fitness_history.append(self.best_individual.fitness)
        
        return self.best_individual.fitness