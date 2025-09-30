import random
import math
import json

# class định nghĩa cá thể
# khởi tạo ở dạng đã mã hoá nếu có
class Individual:
    def __init__(self, num, max_num = 101):
        self.size = num
        self.max_size = max_num
        self.genome = None
        self.fitness = float('inf')

    def copy(self):
        new_individual = Individual(self.size, self.max_size)
        new_individual.genome = self.genome.copy()
        new_individual.fitness = self.fitness
        return new_individual
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness}, genome={self.genome})"
    
    def to_dict(self):
        return {
            "fitness": self.fitness,
            "genome": self.genome
        }

# class lai tạo, đột biến...
class PermutationIndividual(Individual):
    def __init__(self, num = 0, max_num = 101):
        super().__init__(num, max_num)
        self.genome = list(range(max_num))
        random.shuffle(self.genome)

    def decode(self, indi):
        new_ind = indi.copy()
        new_ind.genome = [i for i in new_ind.genome if i < self.size]
        return new_ind
    
    def copy(self):
        child = PermutationIndividual(self.size, self.max_size)
        child.genome = self.genome.copy()
        child.fitness = self.fitness
        return child
    
    # lai
    def crossover(i1, i2):
        size = len(i1.genome)
        a, b = sorted(random.sample(range(size), 2))

        child1_genome = [None] * size
        child2_genome = [None] * size

        child1_genome[a:b] = i1.genome[a:b]
        child2_genome[a:b] = i2.genome[a:b]

        def fill(child, parent):
            pos = b % size
            for gene in parent.genome:
                if gene not in child:
                    child[pos] = gene
                    pos = (pos + 1) % size

        fill(child1_genome, i2)
        fill(child2_genome, i1)

        c1 = i1.copy()
        c2 = i2.copy()
        c1.genome = child1_genome
        c2.genome = child2_genome
        return c1, c2
    
    # đột biến
    def mutate(i1):
        child = i1.copy()
        a, b = random.sample(range(len(child.genome)), 2)
        child.genome[a], child.genome[b] = child.genome[b], child.genome[a]
        return child

class RealCodedIndividual(Individual):
    def __init__(self, num=0, max_num = 101, low=0.0, high=1.0):
        super().__init__(num, max_num)
        # khởi tạo genome là list các số thực trong [low, high]
        self.genome = [random.uniform(low, high) for _ in range(max_num)]
        self.low = low
        self.high = high

    # với real-coded thì không cần decode riêng
    def decode(self, indi):
        # sắp xếp chỉ số theo giá trị genome
        # print("genome: ", indi.genome)
        order = sorted(range(len(indi.genome)), key=lambda i: indi.genome[i])
        order = [i for i in order if i < indi.size]
        # print("order:", order)
        child = PermutationIndividual(indi.size)
        child.genome = order
        child.fitness = indi.fitness
        # print(child)
        return child

    def copy(self):
        child = RealCodedIndividual(self.size, self.max_size)
        child.genome = self.genome.copy()
        child.fitness = self.fitness
        return child
    
    # Crossover: dùng blend crossover
    @staticmethod
    def crossover(i1, i2, alpha=0.5):
        size = len(i1.genome)
        child1_genome = []
        child2_genome = []
        for g1, g2 in zip(i1.genome, i2.genome):
            cmin, cmax = min(g1, g2), max(g1, g2)
            I = cmax - cmin
            lower = cmin - alpha * I
            upper = cmax + alpha * I
            child1_genome.append(random.uniform(lower, upper))
            child2_genome.append(random.uniform(lower, upper))

        c1 = i1.copy()
        c2 = i2.copy()
        c1.genome = child1_genome
        c2.genome = child2_genome
        return c1, c2
    
    # Mutation: thay đổi nhỏ một gene
    @staticmethod
    def mutate(i1, mutation_strength=0.1):
        child = i1.copy()
        idx = random.randrange(len(child.genome))
        # thêm nhiễu Gaussian
        child.genome[idx] += random.gauss(0, mutation_strength)
        return child

# helper: crossover 2 individual có kích thước khác nhau
class CrossoverDifferentTypeIndividual:
    @staticmethod
    def crossover(i1, i2, alpha=0.5):
        size = len(i1.genome)
        child1_genome = []
        child2_genome = []
        for g1, g2 in zip(i1.genome, i2.genome):
            cmin, cmax = min(g1, g2), max(g1, g2)
            I = cmax - cmin
            lower = cmin - alpha * I
            upper = cmax + alpha * I
            child1_genome.append(random.uniform(lower, upper))
            child2_genome.append(random.uniform(lower, upper))

        c1 = i1.copy()
        c2 = i2.copy()
        c1.genome = child1_genome
        c2.genome = child2_genome
        return c1, c2

# class bài toán (TSP/Knapsack)
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

    # đọc file dữ liệu
    def read_file(self):
        with open(self.file_name, 'r') as f:
            for line in f:
                parts = line.split()
                index, x, y = float(parts[0]), float(parts[1]), float(parts[2])
                self.vertices.append((x, y))
    
    # ma trận khoảng cách
    def compute_dist_matrix(self):
        n = len(self.vertices)
        self.size = n
        self.dist_matrix = [[0.0] * n for _ in range (n)]
        for i in range(n):
            for j in range(n):
                x1, y1 = self.vertices[i]
                x2, y2 = self.vertices[j]
                self.dist_matrix[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    # fitness
    def calculate_fitness(self, individual, num = None):
        if (num == 0 or num is None): num = self.size
        fitness = 0
        for i in range(num - 1):
            # tmp_genome = individual.decode(individual)
            # print(individual)
            fitness += self.dist_matrix[individual.genome[i]][individual.genome[i + 1]]
        fitness += self.dist_matrix[individual.genome[-1]][individual.genome[0]]
        return fitness
    
    def get_best_individual_answer(self):
        self.best_individual = {
            "fitness": float('inf'),
            "genome": []
        }
        with open(self.opt_file_name, "r") as f:
            for line in f:
                vertice = int(line.strip())
                self.best_individual["genome"].append(vertice - 1)
        # print(self.best_individual["genome"])
        fitness = 0
        for i in range (len(self.best_individual["genome"]) - 1):
            fitness += self.dist_matrix[self.best_individual["genome"][i]][self.best_individual["genome"][i + 1]]
        fitness += self.dist_matrix[self.best_individual["genome"][-1]][self.best_individual["genome"][0]]
        self.best_individual["fitness"] = fitness

# class thuật toán (GA/MFEA) (khởi tạo quần thể, tính toán...)
class GA:
    def __init__(self, problem, seed, ops, max_pop = 100, num_generation = 500, mutate_rate = 0.1, crossover_rate = 0.9, **kwargs):
        self.problem = problem(**kwargs)
        self.population = list()
        self.max_pop = max_pop # số cá thể trong quần thể
        self.seed = seed
        self.num_generation = num_generation # số thế hệ
        self.mutate_rate = mutate_rate # tỉ lệ đột biến
        self.crossover_rate = crossover_rate # tỉ lệ lai
        self.ops = ops # kiểu individual
        self.best_individual = ops() # individual tốt nhất tìm được
        self.answer = self.problem.best_individual # đáp án của bài toán
        print(f"Answer fitness: {self.answer["fitness"]}")
    
    def initialize(self):
        random.seed(self.seed)
        for _ in range(self.max_pop):
            new_ind = self.ops(self.problem.size) # new individual chưa mã hoá
            # new_ind.fitness = self.problem.calculate_fitness(new_ind, self.problem.size)
            self.population.append(new_ind)
    
    def evaluate(self):
        for ind in self.population:
            # giải mã trước khi đưa vào tính toán
            ind.fitness = self.problem.calculate_fitness(self.ops().decode(ind), self.problem.size)

    def tournament_selection(self, k=3):
        candidates = random.sample(self.population, k)
        return min(candidates, key=lambda ind: ind.fitness)  # TSP là minimization

    def run(self):
        self.initialize()
        self.evaluate()

        for _ in range(self.num_generation):
            new_population = []

            while len(new_population) < self.max_pop:
                # Selection
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()

                # Crossover
                if random.random() < self.crossover_rate:
                    c1, c2 = self.ops.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                if random.random() < self.mutate_rate:
                    c1 = self.ops.mutate(c1)
                if random.random() < self.mutate_rate:
                    c2 = self.ops.mutate(c2)

                new_population.extend([c1, c2])

            # cập nhật quần thể
            self.population = new_population[:self.max_pop]
            self.evaluate()
            best = min(self.population, key=lambda ind: ind.fitness)
            # print(f"Generation {_}: best fitness = {best.fitness}")
            if best.fitness < self.best_individual.fitness:
                self.best_individual = best.copy()

class MFEA_2_Problem:
    def __init__(self, problem, seed, ops, crossover_helper, 
                 max_pop = 100, num_generation = 500, k_tournament = 3,
                 mutate_rate = 0.1, crossover_rate = 0.9, rmp = 0.3, problem_kwargs = None):
        if problem_kwargs is None:
            problem_kwargs = ({}, {})
        self.problem_1 = problem[0](**problem_kwargs[0])
        self.problem_2 = problem[1](**problem_kwargs[1])
        self.population_1 = list()
        self.population_2 = list()
        self.seed = seed
        self.ops = ops
        self.crossover_helper = crossover_helper
        self.k_tournament = k_tournament
        self.max_pop = max_pop
        self.num_generation = num_generation
        self.mutate_rate = mutate_rate
        self.crossover_rate = crossover_rate
        self.rmp = rmp
        self.best_individual_1 = ops()
        self.best_individual_2 = ops()
        self.answer_1 = self.problem_1.best_individual
        self.answer_2 = self.problem_2.best_individual
        print(f"Problem 1 answer fitness: {self.answer_1["fitness"]}")
        print(f"Problem 2 answer fitness: {self.answer_2["fitness"]}")

    def initialize(self):
        random.seed(self.seed)
        for _ in range(self.max_pop):
            new_ind_1 = self.ops(self.problem_1.size)
            new_ind_2 = self.ops(self.problem_2.size)
            self.population_1.append(new_ind_1)
            self.population_2.append(new_ind_2)
            #print(new_ind_1)
            #print(new_ind_2)
        pass

    def evaluate(self):
        for ind in self.population_1:
            ind.fitness = self.problem_1.calculate_fitness(ind.decode(ind), self.problem_1.size)
        for ind in self.population_2:
            # print(ind)
            # print(ind.decode(ind))
            # print(self.problem_2.size)
            ind.fitness = self.problem_2.calculate_fitness(ind.decode(ind), self.problem_2.size)

    def tournament_selection(self, pop = 0):
        k = self.k_tournament
        if (pop == 0):
            candidates = random.sample(self.population_1, k)
            return min(candidates, key=lambda ind: ind.fitness)  # TSP là minimization
        if (pop == 1):
            candidates = random.sample(self.population_2, k)
            return min(candidates, key=lambda ind: ind.fitness)  # TSP là minimization

    def run(self):
        self.initialize()
        self.evaluate()

        for _ in range(self.num_generation):
            new_pop1, new_pop2 = [], []

            while len(new_pop1) < self.max_pop:
                if random.random() < self.rmp:
                    # crossover giữa 2 population
                    p1 = self.tournament_selection(pop=0)
                    p2 = self.tournament_selection(pop=1)
                    # print("before cross: ", len(p1.genome), len(p2.genome))
                    c1, c2 = self.crossover_helper.crossover(p1, p2)
                    if random.random() < self.mutate_rate:
                        c1 = self.ops.mutate(c1)
                    if random.random() < self.mutate_rate:
                        c2 = self.ops.mutate(c2)
                    # print("after cross: ", len(c1.genome), len(c2.genome))
                    new_pop1.append(c1)
                    if len(new_pop2) < self.max_pop:
                        new_pop2.append(c2)
                else:
                    # crossover trong cùng population 1
                    p1 = self.tournament_selection(pop=0)
                    p2 = self.tournament_selection(pop=0)
                    c1, c2 = self.crossover_helper.crossover(p1, p2)
                    if random.random() < self.mutate_rate:
                        c1 = self.ops.mutate(c1)
                    if random.random() < self.mutate_rate:
                        c2 = self.ops.mutate(c2)
                    new_pop1.extend([c1, c2])

            while len(new_pop2) < self.max_pop:
                # crossover trong cùng population 2
                p1 = self.tournament_selection(pop=1)
                p2 = self.tournament_selection(pop=1)
                c1, c2 = self.crossover_helper.crossover(p1, p2)
                if random.random() < self.mutate_rate:
                    c1 = self.ops.mutate(c1)
                if random.random() < self.mutate_rate:
                    c2 = self.ops.mutate(c2)
                new_pop2.extend([c1, c2])

            # cắt nếu vượt quá max_pop
            self.population_1 = new_pop1[:self.max_pop]
            self.population_2 = new_pop2[:self.max_pop]

            self.evaluate()

            best_1 = min(self.population_1, key=lambda ind: ind.fitness)
            if best_1.fitness < self.best_individual_1.fitness:
                self.best_individual_1 = best_1.copy()

            best_2 = min(self.population_2, key=lambda ind: ind.fitness)
            if best_2.fitness < self.best_individual_2.fitness:
                self.best_individual_2 = best_2.copy()

# có thể dùng hàm, không nhất thiết class
class Experiment:
    def __init__(self, algo, problem, seed_list = range(30), **algo_kwards):
        self.algo = algo
        self.problem = problem
        self.seeds = seed_list
        self.algo_kwards = algo_kwards
    
    def run(self):
        result = []
        for (i, seed) in enumerate(self.seeds):
            algorithm = self.algo(self.problem, seed, **self.algo_kwards)
            # if (len(result) == 0): result.append(algorithm.answer)

            # khởi tạo quần thể, tính toán, lai tạo...
            algorithm.run()
            # lưu kết quả tốt nhất tìm được
            best_1 = min(algorithm.population_1, key=lambda ind: ind.fitness) 
            best_2 = min(algorithm.population_2, key=lambda ind: ind.fitness) 
            """
                population: là 1 list chứa các cá thể.
                Cá thể nên có các thuộc tính genome (biểu diễn giá trị), fitness
                Kiến nghị: nên tạo hàm copy() cho cá thể.
            """
            # print(f"Seed {i}, best = {best.fitness}") # giá trị tìm được
            print(f"Seed {i}, best = {algorithm.best_individual_1.fitness}") 
            print(f"Seed {i}, best = {algorithm.best_individual_2.fitness}") 
            best_individual_decoded_1 = algorithm.best_individual_1.decode(algorithm.best_individual_1)
            best_individual_decoded_2 = algorithm.best_individual_2.decode(algorithm.best_individual_2)
            result.append([best_individual_decoded_1, best_individual_decoded_2])
        return result

# danh sách thí nghiệm
experiments = [
    {
        "name": "eil51-eil101-permutation",
        "exp": Experiment(
            MFEA_2_Problem, [TSP, TSP], range(32),
            ops=PermutationIndividual,
            crossover_helper = PermutationIndividual,
            k_tournament = 7,
            problem_kwargs=(
                {"file_name": "GA/eil51.dat", "opt_file_name": "GA/eil51opt.dat"},
                {"file_name": "GA/eil101.dat", "opt_file_name": "GA/eil101opt.dat"}
            )
        )
    },
    {
        "name": "eil51-eil101-real-coded",
        "exp": Experiment(
            MFEA_2_Problem, [TSP, TSP], range(32),
            ops=RealCodedIndividual,
            crossover_helper = RealCodedIndividual,
            k_tournament = 3,
            problem_kwargs=(
                {"file_name": "GA/eil51.dat", "opt_file_name": "GA/eil51opt.dat"},
                {"file_name": "GA/eil101.dat", "opt_file_name": "GA/eil101opt.dat"}
            )
        )
    }
]

# chạy và lưu kết quả
import json

for e in experiments:
    print(f"Running experiment: {e['name']}")
    result = e["exp"].run()   # result: list of [ind1_decoded, ind2_decoded] per seed

    # --- tính thống kê riêng cho mỗi problem ---
    fitness_list_1 = [pair[0].fitness for pair in result]
    fitness_list_2 = [pair[1].fitness for pair in result]

    avg_fitness_1 = sum(fitness_list_1) / len(fitness_list_1)
    avg_fitness_2 = sum(fitness_list_2) / len(fitness_list_2)

    best_found_1 = min(fitness_list_1)
    best_found_2 = min(fitness_list_2)

    # (tuỳ chọn) seed index của best
    best_idx_1 = fitness_list_1.index(best_found_1)
    best_idx_2 = fitness_list_2.index(best_found_2)

    # --- đóng gói kết quả theo seed ---
    results_data = []
    for seed_idx, (ind1, ind2) in enumerate(result):
        results_data.append({
            "seed": seed_idx,
            "problem1": {
                "fitness": ind1.fitness,
                "genome": ", ".join(map(str, ind1.genome))
            },
            "problem2": {
                "fitness": ind2.fitness,
                "genome": ", ".join(map(str, ind2.genome))
            }
        })

    output = {
        "summary": {
            "problem1": {
                "average_fitness": avg_fitness_1,
                "best_found": best_found_1,
                "best_seed_index": best_idx_1
            },
            "problem2": {
                "average_fitness": avg_fitness_2,
                "best_found": best_found_2,
                "best_seed_index": best_idx_2
            }
        },
        "results": results_data
    }

    out_file = f"MFEA/result-{e['name']}_Vinh.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved: {out_file}")
