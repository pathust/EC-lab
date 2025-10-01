import random
import numpy as np
from ga_real_operators import (
    TSP, 
    CrossoverOperators, 
    MutationOperators, 
    SelectionOperators, 
    GeneticAlgorithm
)

def run_top_configuration():
    """Chạy cấu hình tốt nhất: Blend-Gaussian-Truncation với 30 seeds"""
    
    # Đọc dữ liệu TSP
    problem = TSP('../eil51.tsp', '')
    
    # ===== SỬA ĐÂY =====
    # Cấu hình toán tử - ĐÚNG với top 1: Blend-Gaussian-Truncation
    crossover_op = CrossoverOperators.blend_crossover
    mutation_op = MutationOperators.gaussian_mutation      # ← ĐÃ SỬA: gaussian thay vì polynomial
    selection_op = SelectionOperators.truncation_selection
    
    # Tham số GA
    NUM_SEEDS = 30
    POP_SIZE = 100
    GENERATIONS = 500
    PC = 0.9
    PM = 0.1
    
    print("\n" + "="*70)
    print("=== CHẠY CẤU HÌNH TỐT NHẤT: BLEND-GAUSSIAN-TRUNCATION ===")
    print("="*70)
    print(f"Quần thể: {POP_SIZE}")
    print(f"Số thế hệ: {GENERATIONS}")
    print(f"Tỷ lệ lai ghép (Pc): {PC}")
    print(f"Tỷ lệ đột biến (Pm): {PM}")
    print(f"Số lần chạy (seeds): {NUM_SEEDS} (0-{NUM_SEEDS-1})")
    print("="*70 + "\n")
    
    # Chạy với 30 seeds
    seed_results = []
    
    for seed in range(NUM_SEEDS):
        ga = GeneticAlgorithm(
            problem=problem,
            seed=seed,
            crossover_op=crossover_op,
            mutation_op=mutation_op,
            selection_op=selection_op,
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            pc=PC,
            pm=PM
        )
        
        final_fitness = ga.evolve()
        seed_results.append(final_fitness)
        
        print(f"Seed {seed:2d}: Best Found = {final_fitness:.2f}")
    
    # Tính toán thống kê
    best_found = min(seed_results)
    average = np.mean(seed_results)
    std_dev = np.std(seed_results)
    best_seed_index = seed_results.index(best_found)
    worst_found = max(seed_results)
    worst_seed_index = seed_results.index(worst_found)
    
    # In tổng kết
    print("\n" + "="*70)
    print("=== TỔNG KẾT KẾT QUẢ ===")
    print("="*70)
    print(f"Best Found:    {best_found:.2f} (Seed {best_seed_index})")
    print(f"Worst Found:   {worst_found:.2f} (Seed {worst_seed_index})")
    print(f"Average:       {average:.2f}")
    print(f"Std Dev:       {std_dev:.2f}")
    print(f"Range:         [{best_found:.2f} - {worst_found:.2f}]")
    print("="*70 + "\n")
    
    return seed_results, best_found, average, std_dev

if __name__ == "__main__":
    seed_results, best_found, average, std_dev = run_top_configuration()
    print(f"✓ Hoàn thành! Kết quả tốt nhất: {best_found:.2f}")
    print(f"✓ Trung bình: {average:.2f} ± {std_dev:.2f}")