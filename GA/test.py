from ga_real_tsp import *

# Đọc dữ liệu
cities = load_tsp_data('eil51.tsp')
problem = TSPProblem(cities)

# Chạy experiment
experiment = Experiment(
    algo=GeneticAlgorithmReal,
    problem=problem,
    seed_list=range(30),
    pop_size=100,
    generations=500,
    pc=0.9,
    pm=0.1
)

results, fitness_values = experiment.run()