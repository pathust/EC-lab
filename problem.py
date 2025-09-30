import random
import numpy as np
from typing import List
import os
from abc import ABC, abstractmethod

from individual import Individual

class Problem(ABC):
    @abstractmethod
    def evaluate(self, individual: Individual) -> float:
        pass
    
    @abstractmethod
    def decode(self, genome):
        pass
    
    @abstractmethod
    def generate_random_genome(self):
        pass
    
    @abstractmethod
    def get_genome_length(self) -> int:
        pass


class TSPProblem(Problem):
    def __init__(self, 
                 n_cities: int = None, 
                 distance_matrix: np.ndarray = None, 
                 file_path: str = None, 
                 encoding: str = 'permutation'
    ):
        self.n_cities = n_cities
        self.distance_matrix = distance_matrix
        self.coordinates = None
        self.encoding = encoding
        
        if file_path:
            self.load_from_file(file_path)
        elif self.distance_matrix is None and self.n_cities is not None:
            self.generate_random_distances()

        if self.encoding not in ['real', 'permutation']:
            print(f"Warning: Unknown encoding '{self.encoding}'. Defaulting to 'permutation'.")
            self.encoding = 'permutation'
    
    def generate_random_distances(self):
        self.distance_matrix = np.random.randint(10, 100, (self.n_cities, self.n_cities))
        np.fill_diagonal(self.distance_matrix, 0)
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
    
    def load_from_file(self, file_path: str):
        self.coordinates = []
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        self.coordinates.append((x, y))
                    except (ValueError, IndexError):
                        continue
        
        self.n_cities = len(self.coordinates)
        self.distance_matrix = np.zeros((self.n_cities, self.n_cities))
        
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                dx = self.coordinates[i][0] - self.coordinates[j][0]
                dy = self.coordinates[i][1] - self.coordinates[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist
        
        print(f"Loaded TSP problem with {self.n_cities} cities from {file_path}")
    
    def evaluate(self, individual: Individual) -> float:
        path = self.decode(individual.genome)
        if len(path) < 2:
            return -float('inf')
        
        dist = 0
        for i in range(len(path) - 1):
            dist += self.distance_matrix[path[i]][path[i + 1]]
        dist += self.distance_matrix[path[-1]][path[0]]
        return -dist
    
    def decode(self, genome):
        if self.encoding == 'permutation':
            valid_cities = []
            seen = set()
            
            for city in genome:
                if isinstance(city, (int, np.integer)) and 0 <= city < self.n_cities and city not in seen:
                    valid_cities.append(int(city))
                    seen.add(city)
            
            if len(valid_cities) < self.n_cities:
                for i in range(self.n_cities):
                    if i not in seen:
                        valid_cities.append(i)
                        if len(valid_cities) >= self.n_cities:
                            break
            
            return valid_cities[:self.n_cities]
        else:
            if len(genome) < self.n_cities:
                genome_padded = list(genome) + [random.random() for _ in range(self.n_cities - len(genome))]
            else:
                genome_padded = genome[:self.n_cities]
            
            indices = list(range(self.n_cities))
            sorted_indices = sorted(indices, key=lambda i: genome_padded[i])
            return sorted_indices
    
    def generate_random_genome(self):
        if self.encoding == 'permutation':
            cities = list(range(self.n_cities))
            random.shuffle(cities)
            return cities
        else:
            return [random.random() for _ in range(self.n_cities)]
    
    def get_genome_length(self) -> int:
        return self.n_cities


class KnapsackProblem(Problem):
    def __init__(self, 
                 weights: List = None, 
                 values: List = None, 
                 capacity: int = None, 
                 file_prefix: str = None, 
                 encoding: str = 'binary'
    ):
        self.weights = weights if weights else []
        self.values = values if values else []
        self.capacity = capacity if capacity else 0
        self.encoding = encoding
        self.n_items = len(self.weights) if weights else 0
        self.optimal_solution = None
        
        if file_prefix:
            self.load_from_files(file_prefix)

        if self.encoding not in ['binary', 'real', 'permutation']:
            print(f"Warning: Unknown encoding '{self.encoding}'. Defaulting to 'binary'.")
            self.encoding = 'binary'
    
    def load_from_files(self, file_prefix: str):
        try:
            capacity_file = f"{file_prefix}_c.txt"
            if os.path.exists(capacity_file):
                with open(capacity_file, 'r') as f:
                    self.capacity = int(f.read().strip())
            
            values_file = f"{file_prefix}_p.txt"
            if os.path.exists(values_file):
                with open(values_file, 'r') as f:
                    self.values = [int(line.strip()) for line in f if line.strip()]
            
            weights_file = f"{file_prefix}_w.txt"
            if os.path.exists(weights_file):
                with open(weights_file, 'r') as f:
                    self.weights = [int(line.strip()) for line in f if line.strip()]
            
            solution_file = f"{file_prefix}_s.txt"
            if os.path.exists(solution_file):
                with open(solution_file, 'r') as f:
                    self.optimal_solution = [int(line.strip()) for line in f if line.strip()]
            
            self.n_items = len(self.weights)
            
            if len(self.weights) != len(self.values):
                raise ValueError(f"Mismatch: {len(self.weights)} weights vs {len(self.values)} values")
            
            print(f"Loaded Knapsack: {self.n_items} items, capacity={self.capacity}")
            
        except Exception as e:
            print(f"Error loading Knapsack data: {e}")
            raise
    
    def evaluate(self, individual: Individual) -> float:
        genome = self.decode(individual.genome)
        total_w, total_v = 0, 0
        
        for i in range(min(len(genome), self.n_items)):
            if genome[i] == 1:
                total_w += self.weights[i]
                total_v += self.values[i]
        
        if total_w > self.capacity:
            max_value = max(self.values)
            penalty = (total_w - self.capacity) * max_value
            return max(0, total_v - penalty)
        
        return total_v
    
    def decode(self, genome):
        if self.encoding == 'permutation':
            return self._decode_permutation_to_binary(genome)
        
        elif self.encoding == 'real':
            return self._decode_real_to_binary(genome)

        else:
            if len(genome) < self.n_items:
                return [int(round(x)) if isinstance(x, (int, float)) else 0 for x in genome] + [0] * (self.n_items - len(genome))
            return [int(round(x)) if isinstance(x, (int, float)) else 0 for x in genome[:self.n_items]]

    def _decode_permutation_to_binary(self, genome):
        binary_genome = [0] * self.n_items
        remaining_capacity = self.capacity
        
        valid_indices = [idx for idx in genome if isinstance(idx, (int, np.integer)) and 0 <= idx < self.n_items]
        
        ratios = []
        for idx in valid_indices:
            ratios.append((self.values[idx] / self.weights[idx], idx))
        
        ratios.sort(reverse=True)
        
        for _, idx in ratios:
            if self.weights[idx] <= remaining_capacity:
                binary_genome[idx] = 1
                remaining_capacity -= self.weights[idx]
        
        return binary_genome

    def _decode_real_to_binary(self, genome):
        binary_genome = [0] * self.n_items
        remaining_capacity = self.capacity
        
        items_with_priority = []
        for i in range(min(len(genome), self.n_items)):
            items_with_priority.append((genome[i], i))
        
        items_with_priority.sort(key=lambda x: x[0], reverse=True)
        
        for _, idx in items_with_priority:
            if self.weights[idx] <= remaining_capacity:
                binary_genome[idx] = 1
                remaining_capacity -= self.weights[idx]
        
        return binary_genome[:self.n_items]

    
    def generate_random_genome(self):
        if self.encoding == 'permutation':
            perm = list(range(self.n_items))
            random.shuffle(perm)
            return perm
        elif self.encoding == 'real':
            return [random.random() for _ in range(self.n_items)]
        else:
            return [random.randint(0, 1) for _ in range(self.n_items)]
    
    def get_genome_length(self) -> int:
        return self.n_items
