import random
import math
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class Individual:
    """Cá thể trong GA với mã hóa số thực"""
    def __init__(self, genome: List[float]):
        self.genome = genome.copy()  # Vector số thực biểu diễn độ ưu tiên
        self.fitness = float('inf')  # Khoảng cách tour (càng nhỏ càng tốt)
        self.tour = []  # Tour được giải mã từ genome
    
    def copy(self):
        """Tạo bản sao của cá thể"""
        new_ind = Individual(self.genome)
        new_ind.fitness = self.fitness
        new_ind.tour = self.tour.copy()
        return new_ind

class TSPProblem:
    """Bài toán TSP"""
    def __init__(self, cities: List[Tuple[float, float]]):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self):
        """Tính ma trận khoảng cách giữa các thành phố"""
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
        """Giải mã genome thành tour bằng cách sắp xếp theo độ ưu tiên"""
        # Tạo list các cặp (độ ưu tiên, index thành phố)
        priority_pairs = [(genome[i], i) for i in range(len(genome))]
        # Sắp xếp theo độ ưu tiên
        priority_pairs.sort(key=lambda x: x[0])
        # Trả về tour (list các index thành phố)
        return [pair[1] for pair in priority_pairs]
    
    def evaluate(self, individual: Individual) -> float:
        """Tính fitness (tổng khoảng cách tour)"""
        tour = self.decode(individual.genome)
        individual.tour = tour
        
        total_distance = 0.0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[current_city][next_city]
        
        individual.fitness = total_distance
        return total_distance

class GeneticOperators:
    """Các toán tử di truyền cho mã hóa số thực"""
    
    @staticmethod
    def sbx_crossover(parent1: Individual, parent2: Individual, eta_c: float = 20.0) -> Tuple[Individual, Individual]:
        """Simulated Binary Crossover (SBX)"""
        n = len(parent1.genome)
        child1_genome = []
        child2_genome = []
        
        for i in range(n):
            if random.random() <= 0.5:  # Probability of crossover for each gene
                # Thực hiện SBX cho gene thứ i
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
                # Không lai ghép, giữ nguyên
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
        """Tournament selection"""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda ind: ind.fitness)

class GeneticAlgorithmReal:
    """Genetic Algorithm với mã hóa số thực cho TSP"""
    
    def __init__(self, problem: TSPProblem, seed: int, pop_size: int = 100, 
                 generations: int = 500, pc: float = 0.9, pm: float = 0.1):
        random.seed(seed)
        np.random.seed(seed)
        
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc  # Crossover probability
        self.pm = pm  # Mutation probability
        self.population = []
        self.best_individual = None
        self.fitness_history = []
    
    def initialize(self):
        """Khởi tạo quần thể ngẫu nhiên"""
        self.population = []
        for _ in range(self.pop_size):
            # Tạo genome ngẫu nhiên với các giá trị trong [0, 1]
            genome = [random.random() for _ in range(self.problem.num_cities)]
            individual = Individual(genome)
            self.population.append(individual)
    
    def evaluate(self):
        """Đánh giá fitness cho toàn bộ quần thể"""
        for individual in self.population:
            self.problem.evaluate(individual)
        
        # Cập nhật best individual
        current_best = min(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
            self.best_individual = current_best.copy()
    
    def evolve(self):
        """Tiến hóa qua các thế hệ"""
        for generation in range(self.generations):
            # Tạo thế hệ mới
            new_population = []
            
            # Elitism: giữ lại cá thể tốt nhất
            best = min(self.population, key=lambda ind: ind.fitness)
            new_population.append(best.copy())
            
            # Tạo các cá thể con
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = GeneticOperators.tournament_selection(self.population)
                parent2 = GeneticOperators.tournament_selection(self.population)
                
                # Crossover
                if random.random() < self.pc:
                    child1, child2 = GeneticOperators.sbx_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                GeneticOperators.polynomial_mutation(child1, pm=self.pm)
                GeneticOperators.polynomial_mutation(child2, pm=self.pm)
                
                new_population.extend([child1, child2])
            
            # Giới hạn kích thước quần thể
            self.population = new_population[:self.pop_size]
            
            # Đánh giá quần thể mới
            self.evaluate()
            
            # Lưu lịch sử fitness
            current_best_fitness = min(ind.fitness for ind in self.population)
            self.fitness_history.append(current_best_fitness)
            
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {current_best_fitness:.2f}")
    
    def get_best_tour(self):
        """Lấy tour tốt nhất"""
        if self.best_individual:
            return self.best_individual.tour, self.best_individual.fitness
        return None, float('inf')

def create_sample_cities(n=51, seed=42):
    """Tạo dữ liệu mẫu cho n thành phố"""
    random.seed(seed)
    cities = []
    for i in range(n):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append((x, y))
    return cities

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
                        # Index, X, Y
                        x, y = float(parts[1]), float(parts[2])
                        cities.append((x, y))
        
        print(f"Đã đọc thành công {len(cities)} thành phố từ {filename}")
        return cities
    
    except FileNotFoundError:
        print(f"Không tìm thấy file {filename}")
        print("Sẽ tạo dữ liệu mẫu để demo...")
        return create_sample_cities(51, seed=42)
    except Exception as e:
        print(f"Lỗi khi đọc file {filename}: {e}")
        print("Sẽ tạo dữ liệu mẫu để demo...")
        return create_sample_cities(51, seed=42)

# Hàm tiện ích để vẽ tour
def plot_tour(cities: List[Tuple[float, float]], tour: List[int], title: str = "TSP Tour"):
    """Vẽ tour TSP"""
    plt.figure(figsize=(10, 8))
    
    # Vẽ các thành phố
    x_coords = [cities[i][0] for i in range(len(cities))]
    y_coords = [cities[i][1] for i in range(len(cities))]
    plt.scatter(x_coords, y_coords, c='red', s=50, zorder=2)
    
    # Vẽ tour
    tour_x = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    tour_y = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
    plt.plot(tour_x, tour_y, 'b-', linewidth=1, zorder=1)
    
    # Đánh số các thành phố
    for i, (x, y) in enumerate(cities):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.show()

class Experiment:
    """Class thực nghiệm như yêu cầu"""
    def __init__(self, algo, problem, seed_list=range(30), **algo_kwargs):
        self.algo = algo
        self.problem = problem
        self.seeds = seed_list
        self.algo_kwargs = algo_kwargs
    
    def run(self):
        results = []
        fitness_values = []
        
        for i, seed in enumerate(self.seeds):
            algorithm = self.algo(self.problem, seed, **self.algo_kwargs)
            
            # Khởi tạo quần thể, tính toán, lai tạo...
            algorithm.initialize()
            algorithm.evaluate()
            algorithm.evolve()
            
            # Lưu kết quả tốt nhất tìm được
            best = min(algorithm.population, key=lambda ind: ind.fitness)
            print(f"Seed {i}: best = {best.fitness:.2f}")
            
            results.append(best)
            fitness_values.append(best.fitness)
        
        # Tính toán thống kê
        best_fitness = min(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        
        print(f"\n=== KẾT QUẢ THỰC NGHIỆM ===")
        print(f"Best Found: {best_fitness:.2f}")
        print(f"Average: {avg_fitness:.2f}")
        print(f"Std Dev: {np.std(fitness_values):.2f}")
        return results, fitness_values