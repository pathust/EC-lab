# class thuật toán (GA/MFEA) (khởi tạo quần thể, tính toán...)
# class bài toán (TSP/Knapsack)
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
            algorithm = self.algo(self.problem, seed, self.algo_kwards)
            # khởi tạo quần thể, tính toán, lai tạo...
            best = min() # lưu kết quả tốt nhất tìm được
            print(f"Seed {i}, best = {best.fitness}") # giá trị tìm được
            result.append(best)
        return result


exp = Experiment()
exp.run()