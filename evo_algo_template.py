import random

# class thuật toán (GA/MFEA) (khởi tạo quần thể, tính toán...)
# class bài toán (TSP/Knapsack)
# class định nghĩa cá thể
# class lai tạo, đột biến...
# class mã hoá/giải mã hoá cá thể 
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

            # khởi tạo quần thể, tính toán, lai tạo...
            algorithm.initialize()
            algorithm.evaluate()

            # lưu kết quả tốt nhất tìm được
            best = min(algorithm.population, key=lambda ind: ind.fitness) 
            """
                population: là 1 list chứa các cá thể.
                Cá thể nên có các thuộc tính Genome (biểu diễn giá trị), fitness
                Kiến nghị: nên tạo hàm copy() cho cá thể.
            """
            print(f"Seed {i}, best = {best.fitness}") # giá trị tìm được
            result.append(best)
        return result


exp = Experiment()
exp.run()