import random
import math
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class Individual:
    """Cá thể trong MFEA"""
    def __init__(self, genome: List[float]):
        self.genome = genome.copy()
        self.skill_factor = 0  # Task mà cá thể này giỏi nhất (0 hoặc 1)
        self.scalar_fitness = float('inf')  # Scalar fitness để so sánh
        
        # Fitness và tour cho từng task
        self.fitness = [float('inf'), float('inf')]  # [fitness_task1, fitness_task2]
        self.tour = [[], []]  # [tour_task1, tour_task2]
        self.factorial_rank = [float('inf'), float('inf')]  # Rank trong từng task
    
    def copy(self):
        new_ind = Individual(self.genome)
        new_ind.skill_factor = self.skill_factor
        new_ind.scalar_fitness = self.scalar_fitness
        new_ind.fitness = self.fitness.copy()
        new_ind.tour = [t.copy() for t in self.tour]
        new_ind.factorial_rank = self.factorial_rank.copy()
        return new_ind

class TSPProblem:
    """Bài toán TSP"""
    def __init__(self, cities: List[Tuple[float, float]]):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self):
        n = self.num_cities
        matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.cities[i]
                    x2, y2 = self.cities[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    matrix[i][j] = distance
        return matrix
    
    def decode(self, genome: List[float]) -> List[int]:
        """Giải mã genome thành tour"""
        # Chỉ lấy phần genome tương ứng với số thành phố của task này
        relevant_genome = genome[:self.num_cities]
        priority_pairs = [(relevant_genome[i], i) for i in range(len(relevant_genome))]
        priority_pairs.sort(key=lambda x: x[0])
        return [pair[1] for pair in priority_pairs]
    
    def evaluate(self, individual: Individual, task_id: int) -> float:
        """Tính fitness cho task cụ thể"""
        tour = self.decode(individual.genome)
        individual.tour[task_id] = tour
        
        total_distance = 0.0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[current_city][next_city]
        
        individual.fitness[task_id] = total_distance
        return total_distance

class GeneticOperators:
    """Các toán tử di truyền"""
    
    @staticmethod
    def sbx_crossover(parent1: Individual, parent2: Individual, eta_c: float = 20.0) -> Tuple[Individual, Individual]:
        """SBX Crossover"""
        n = len(parent1.genome)
        child1_genome = []
        child2_genome = []
        
        for i in range(n):
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
        
        return Individual(child1_genome), Individual(child2_genome)
    
    @staticmethod
    def polynomial_mutation(individual: Individual, eta_m: float = 20.0, pm: float = 0.1):
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
    def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
        """Tournament selection dựa trên scalar fitness"""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda ind: ind.scalar_fitness)

class MFEA:
    """Multifactorial Evolutionary Algorithm cho TSP"""
    
    def __init__(self, problems: List[TSPProblem], seed: int, pop_size: int = 100, 
                 generations: int = 500, rmp: float = 0.3, pc: float = 0.9, pm: float = 0.1):
        random.seed(seed)
        np.random.seed(seed)
        
        self.problems = problems
        self.num_tasks = len(problems)
        self.pop_size = pop_size
        self.generations = generations
        self.rmp = rmp  # Random mating probability
        self.pc = pc
        self.pm = pm
        
        # Genome size = max số thành phố của các task
        self.genome_size = max(p.num_cities for p in problems)
        
        self.population = []
        self.best_individuals = [None] * self.num_tasks
        self.fitness_history = [[] for _ in range(self.num_tasks)]
    
    def initialize(self):
        """Khởi tạo quần thể"""
        self.population = []
        for _ in range(self.pop_size):
            genome = [random.random() for _ in range(self.genome_size)]
            individual = Individual(genome)
            self.population.append(individual)
    
    def evaluate(self):
        """Đánh giá toàn bộ quần thể"""
        # Đánh giá từng cá thể trên tất cả các task
        for individual in self.population:
            for task_id, problem in enumerate(self.problems):
                problem.evaluate(individual, task_id)
        
        # Tính factorial rank cho từng task
        for task_id in range(self.num_tasks):
            # Sắp xếp theo fitness của task này
            sorted_pop = sorted(self.population, key=lambda ind: ind.fitness[task_id])
            for rank, individual in enumerate(sorted_pop):
                individual.factorial_rank[task_id] = rank + 1
        
        # Tính skill factor và scalar fitness
        for individual in self.population:
            # Skill factor = task có rank tốt nhất
            individual.skill_factor = min(range(self.num_tasks), 
                                        key=lambda t: individual.factorial_rank[t])
            # Scalar fitness = rank tốt nhất
            individual.scalar_fitness = min(individual.factorial_rank)
        
        # Cập nhật best individuals
        for task_id in range(self.num_tasks):
            task_best = min(self.population, key=lambda ind: ind.fitness[task_id])
            if (self.best_individuals[task_id] is None or 
                task_best.fitness[task_id] < self.best_individuals[task_id].fitness[task_id]):
                self.best_individuals[task_id] = task_best.copy()
    
    def can_mate(self, parent1: Individual, parent2: Individual) -> bool:
        """Kiểm tra xem 2 cá thể có thể lai ghép không"""
        if parent1.skill_factor == parent2.skill_factor:
            return True  # Cùng skill factor
        else:
            return random.random() < self.rmp  # Khác skill factor, xác suất rmp
    
    def evolve(self):
        """Tiến hóa qua các thế hệ"""
        for generation in range(self.generations):
            new_population = []
            
            # Elitism: giữ lại cá thể tốt nhất của mỗi task
            for task_id in range(self.num_tasks):
                task_best = min(self.population, key=lambda ind: ind.fitness[task_id])
                new_population.append(task_best.copy())
            
            # Tạo các cá thể con
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = GeneticOperators.tournament_selection(self.population)
                parent2 = GeneticOperators.tournament_selection(self.population)
                
                # Kiểm tra xem có thể lai ghép không
                if self.can_mate(parent1, parent2):
                    # Crossover
                    if random.random() < self.pc:
                        child1, child2 = GeneticOperators.sbx_crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    GeneticOperators.polynomial_mutation(child1, pm=self.pm)
                    GeneticOperators.polynomial_mutation(child2, pm=self.pm)
                    
                    new_population.extend([child1, child2])
                else:
                    # Không lai ghép, chỉ sao chép
                    new_population.extend([parent1.copy(), parent2.copy()])
            
            # Giới hạn kích thước quần thể
            self.population = new_population[:self.pop_size]
            
            # Đánh giá quần thể mới
            self.evaluate()
            
            # Lưu lịch sử fitness
            for task_id in range(self.num_tasks):
                best_fitness = min(ind.fitness[task_id] for ind in self.population)
                self.fitness_history[task_id].append(best_fitness)
            
            if generation % 50 == 0:
                print(f"Generation {generation}:")
                for task_id in range(self.num_tasks):
                    best_fitness = min(ind.fitness[task_id] for ind in self.population)
                    print(f"  Task {task_id}: Best fitness = {best_fitness:.2f}")
    
    def get_best_results(self):
        """Lấy kết quả tốt nhất cho tất cả các task"""
        results = []
        for task_id in range(self.num_tasks):
            if self.best_individuals[task_id]:
                tour = self.best_individuals[task_id].tour[task_id]
                fitness = self.best_individuals[task_id].fitness[task_id]
                results.append((tour, fitness))
            else:
                results.append(([], float('inf')))
        return results

def load_tsp_data(filename: str) -> List[Tuple[float, float]]:
    """Đọc dữ liệu TSP từ file"""
    cities = []
    reading_coords = False
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line == "NODE_COORD_SECTION":
                    reading_coords = True
                    continue
                elif line == "EOF":
                    break
                elif reading_coords and line:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y = float(parts[1]), float(parts[2])
                        cities.append((x, y))
        
        print(f"Đã đọc thành công {len(cities)} thành phố từ {filename}")
        return cities
    
    except FileNotFoundError:
        print(f"Không tìm thấy file {filename}")
        return []
    except Exception as e:
        print(f"Lỗi khi đọc file {filename}: {e}")
        return []

def create_sample_cities(n=101, seed=42):
    """Tạo dữ liệu mẫu cho n thành phố (dự phòng)"""
    random.seed(seed)
    cities = []
    for i in range(n):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append((x, y))
    return cities

class Experiment:
    """Class thực nghiệm MFEA"""
    def __init__(self, algo, problems, seed_list=range(30), **algo_kwargs):
        self.algo = algo
        self.problems = problems
        self.seeds = seed_list
        self.algo_kwargs = algo_kwargs
    
    def run(self):
        all_results = [[] for _ in range(len(self.problems))]  # Kết quả cho từng task
        all_fitness = [[] for _ in range(len(self.problems))]
        
        for i, seed in enumerate(self.seeds):
            algorithm = self.algo(self.problems, seed, **self.algo_kwargs)
            
            algorithm.initialize()
            algorithm.evaluate()
            algorithm.evolve()
            
            # Lấy kết quả tốt nhất cho từng task
            results = algorithm.get_best_results()
            
            print(f"Seed {i}:")
            for task_id, (tour, fitness) in enumerate(results):
                print(f"  Task {task_id}: best = {fitness:.2f}")
                all_results[task_id].append((tour, fitness))
                all_fitness[task_id].append(fitness)
        
        # Tính toán thống kê cho từng task
        print(f"\n=== KẾT QUẢ THỰC NGHIỆM MFEA ===")
        for task_id in range(len(self.problems)):
            fitness_values = all_fitness[task_id]
            best_fitness = min(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)
            
            print(f"Task {task_id} ({len(self.problems[task_id].cities)} cities):")
            print(f"  Best Found: {best_fitness:.2f}")
            print(f"  Average: {avg_fitness:.2f}")
            print(f"  Std Dev: {np.std(fitness_values):.2f}")
        
        return all_results, all_fitness