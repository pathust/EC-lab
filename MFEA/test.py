from mfea_tsp import *

# Đọc dữ liệu từ 2 file TSP
cities_51 = load_tsp_data('../eil51.tsp')
cities_101 = load_tsp_data('../eil101.tsp')

# Tạo dữ liệu mẫu nếu không đọc được file
if not cities_51:
    print("Không đọc được eil51.tsp, tạo dữ liệu mẫu 51 thành phố")
    cities_51 = create_sample_cities(51, seed=51)

if not cities_101:
    print("Không đọc được eil101.tsp, tạo dữ liệu mẫu 101 thành phố")
    cities_101 = create_sample_cities(101, seed=101)

# Tạo các bài toán TSP
problem_51 = TSPProblem(cities_51)
problem_101 = TSPProblem(cities_101)
problems = [problem_51, problem_101]

print(f"Task 0: TSP với {len(cities_51)} thành phố")
print(f"Task 1: TSP với {len(cities_101)} thành phố")

# Chạy experiment MFEA
experiment = Experiment(
    algo=MFEA,
    problems=problems,
    seed_list=range(30),
    pop_size=100,
    generations=500,
    rmp=0.3,  # Random mating probability
    pc=0.9,
    pm=0.1
)

results, fitness_values = experiment.run()