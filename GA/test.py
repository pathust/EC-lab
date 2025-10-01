import random
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Import từ ga_real_operators.py
from ga_real_operators import (
    Individual, 
    TSP, 
    CrossoverOperators, 
    MutationOperators, 
    SelectionOperators, 
    GeneticAlgorithm
)

def run_experiment():
    """Chạy thực nghiệm so sánh các toán tử với 30 seeds"""
    
    # Tạo thư mục cần thiết
    os.makedirs('images', exist_ok=True)
    
    # Đọc dữ liệu TSP
    problem = TSP('../eil51.tsp', '')
    
    # Định nghĩa các toán tử
    crossover_ops = {
        'SBX': CrossoverOperators.sbx_crossover,
        'Blend': CrossoverOperators.blend_crossover,
        'Arithmetic': CrossoverOperators.arithmetic_crossover
    }
    
    mutation_ops = {
        'Gaussian': MutationOperators.gaussian_mutation,
        'Polynomial': MutationOperators.polynomial_mutation,
        'Uniform': MutationOperators.uniform_mutation
    }
    
    selection_ops = {
        'Tournament': SelectionOperators.tournament_selection,
        'Roulette': SelectionOperators.roulette_selection,
        'Truncation': SelectionOperators.truncation_selection
    }
    
    # Tham số GA theo đề bài
    NUM_SEEDS = 30
    POP_SIZE = 100
    GENERATIONS = 500
    PC = 0.9
    PM = 0.1
    
    print("\n=== CẤU HÌNH THỰC NGHIỆM ===")
    print(f"Quần thể: {POP_SIZE}")
    print(f"Số thế hệ: {GENERATIONS}")
    print(f"Tỷ lệ lai ghép (Pc): {PC}")
    print(f"Tỷ lệ đột biến (Pm): {PM}")
    print(f"Số lần chạy (seeds): {NUM_SEEDS} (0-{NUM_SEEDS-1})")
    print("=" * 50)
    
    # Tạo tất cả các kết hợp
    results = {}
    
    print("\n=== BẮT ĐẦU THỰC NGHIỆM ===\n")
    
    config_count = 0
    total_configs = len(crossover_ops) * len(mutation_ops) * len(selection_ops)
    
    for cross_name, cross_op in crossover_ops.items():
        for mut_name, mut_op in mutation_ops.items():
            for sel_name, sel_op in selection_ops.items():
                config_count += 1
                config_name = f"{cross_name}-{mut_name}-{sel_name}"
                print(f"\n[{config_count}/{total_configs}] Đang chạy: {config_name}")
                print("-" * 60)
                
                # Chạy với 30 seeds
                seed_results = []
                all_fitness_histories = []
                
                for seed in range(NUM_SEEDS):
                    ga = GeneticAlgorithm(
                        problem=problem,
                        seed=seed,
                        crossover_op=cross_op,
                        mutation_op=mut_op,
                        selection_op=sel_op,
                        pop_size=POP_SIZE,
                        generations=GENERATIONS,
                        pc=PC,
                        pm=PM
                    )
                    
                    final_fitness = ga.evolve()
                    seed_results.append(final_fitness)
                    all_fitness_histories.append(ga.fitness_history)
                    
                    if (seed + 1) % 10 == 0:
                        print(f"  Hoàn thành seed {seed}: {final_fitness:.2f}")
                
                # Tính toán thống kê
                best_found = min(seed_results)
                average = np.mean(seed_results)
                std_dev = np.std(seed_results)
                best_seed_index = seed_results.index(best_found)
                
                # Tính trung bình convergence curve
                avg_history = np.mean(all_fitness_histories, axis=0)
                
                results[config_name] = {
                    'seed_results': seed_results,
                    'best_found': best_found,
                    'average': average,
                    'std_dev': std_dev,
                    'best_seed_index': best_seed_index,
                    'avg_convergence': avg_history.tolist()
                }
                
                print(f"  ✓ Best Found: {best_found:.2f} (seed {best_seed_index})")
                print(f"  ✓ Average: {average:.2f} ± {std_dev:.2f}")
    
    # Lưu kết quả chi tiết
    results_file = 'operator_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Đã lưu kết quả chi tiết vào: {results_file}")
    
    # Visualize
    visualize_results(results)
    
    return results

def visualize_results(results):
    """Visualize kết quả so sánh"""
    
    os.makedirs('images', exist_ok=True)
    
    # Sắp xếp theo Best Found
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_found'])
    
    # 1. So sánh Best Found của tất cả cấu hình
    plt.figure(figsize=(16, 8))
    configs = [item[0] for item in sorted_results]
    best_founds = [item[1]['best_found'] for item in sorted_results]
    averages = [item[1]['average'] for item in sorted_results]
    
    x = np.arange(len(configs))
    width = 0.35
    
    plt.bar(x - width/2, best_founds, width, label='Best Found', alpha=0.8, color='green')
    plt.bar(x + width/2, averages, width, label='Average', alpha=0.8, color='blue')
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Best Found vs Average - All Configurations', fontsize=14)
    plt.xticks(x, configs, rotation=90, ha='right', fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('images/best_found_comparison.png', dpi=150, bbox_inches='tight')
    print("Đã lưu: images/best_found_comparison.png")
    
    # 2. Top 5 configurations - Average convergence
    top_5 = sorted_results[:5]
    
    plt.figure(figsize=(14, 8))
    for config_name, data in top_5:
        plt.plot(data['avg_convergence'], 
                label=f"{config_name}\n(Best: {data['best_found']:.2f}, Avg: {data['average']:.2f})", 
                linewidth=2)
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Average Best Fitness', fontsize=12)
    plt.title('Top 5 Configurations - Average Convergence (30 seeds)', fontsize=14)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/top5_convergence.png', dpi=150)
    print("Đã lưu: images/top5_convergence.png")
    
    # 3. Box plot cho top 10
    top_10 = sorted_results[:10]
    
    plt.figure(figsize=(14, 8))
    box_data = [item[1]['seed_results'] for item in top_10]
    box_labels = [item[0] for item in top_10]
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Distribution of Results (30 seeds) - Top 10 Configurations', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('images/top10_boxplot.png', dpi=150)
    print("Đã lưu: images/top10_boxplot.png")
    
    # 4-6. So sánh theo từng loại operator
    # Crossover comparison
    crossover_results = {}
    for config, data in results.items():
        cross = config.split('-')[0]
        if cross not in crossover_results:
            crossover_results[cross] = {'best_founds': [], 'averages': []}
        crossover_results[cross]['best_founds'].append(data['best_found'])
        crossover_results[cross]['averages'].append(data['average'])
    
    plt.figure(figsize=(10, 6))
    cross_names = list(crossover_results.keys())
    cross_best_avgs = [np.mean(crossover_results[c]['best_founds']) for c in cross_names]
    cross_avg_avgs = [np.mean(crossover_results[c]['averages']) for c in cross_names]
    
    x = np.arange(len(cross_names))
    width = 0.35
    
    plt.bar(x - width/2, cross_best_avgs, width, label='Avg of Best Found', alpha=0.8, color='green')
    plt.bar(x + width/2, cross_avg_avgs, width, label='Avg of Average', alpha=0.8, color='blue')
    
    plt.xlabel('Crossover Operator', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Crossover Operator Comparison', fontsize=14)
    plt.xticks(x, cross_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('images/crossover_comparison.png', dpi=150)
    print("Đã lưu: images/crossover_comparison.png")
    
    # Mutation comparison
    mutation_results = {}
    for config, data in results.items():
        mut = config.split('-')[1]
        if mut not in mutation_results:
            mutation_results[mut] = {'best_founds': [], 'averages': []}
        mutation_results[mut]['best_founds'].append(data['best_found'])
        mutation_results[mut]['averages'].append(data['average'])
    
    plt.figure(figsize=(10, 6))
    mut_names = list(mutation_results.keys())
    mut_best_avgs = [np.mean(mutation_results[m]['best_founds']) for m in mut_names]
    mut_avg_avgs = [np.mean(mutation_results[m]['averages']) for m in mut_names]
    
    x = np.arange(len(mut_names))
    
    plt.bar(x - width/2, mut_best_avgs, width, label='Avg of Best Found', alpha=0.8, color='green')
    plt.bar(x + width/2, mut_avg_avgs, width, label='Avg of Average', alpha=0.8, color='blue')
    
    plt.xlabel('Mutation Operator', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Mutation Operator Comparison', fontsize=14)
    plt.xticks(x, mut_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('images/mutation_comparison.png', dpi=150)
    print("Đã lưu: images/mutation_comparison.png")
    
    # Selection comparison
    selection_results = {}
    for config, data in results.items():
        sel = config.split('-')[2]
        if sel not in selection_results:
            selection_results[sel] = {'best_founds': [], 'averages': []}
        selection_results[sel]['best_founds'].append(data['best_found'])
        selection_results[sel]['averages'].append(data['average'])
    
    plt.figure(figsize=(10, 6))
    sel_names = list(selection_results.keys())
    sel_best_avgs = [np.mean(selection_results[s]['best_founds']) for s in sel_names]
    sel_avg_avgs = [np.mean(selection_results[s]['averages']) for s in sel_names]
    
    x = np.arange(len(sel_names))
    
    plt.bar(x - width/2, sel_best_avgs, width, label='Avg of Best Found', alpha=0.8, color='green')
    plt.bar(x + width/2, sel_avg_avgs, width, label='Avg of Average', alpha=0.8, color='blue')
    
    plt.xlabel('Selection Operator', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Selection Operator Comparison', fontsize=14)
    plt.xticks(x, sel_names)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('images/selection_comparison.png', dpi=150)
    print("Đã lưu: images/selection_comparison.png")
    
    # In tổng kết
    print("\n" + "="*80)
    print("=== TỔNG KẾT KẾT QUẢ ===")
    print("="*80)
    
    print("\nTOP 10 CONFIGURATIONS (theo Best Found):")
    print("-" * 80)
    print(f"{'Rank':<5} {'Configuration':<35} {'Best Found':<12} {'Average':<12} {'Std Dev':<10}")
    print("-" * 80)
    for i, (config, data) in enumerate(sorted_results[:10], 1):
        print(f"{i:<5} {config:<35} {data['best_found']:<12.2f} {data['average']:<12.2f} {data['std_dev']:<10.2f}")
    
    print("\n" + "="*80)
    print("SO SÁNH THEO LOẠI TOÁN TỬ:")
    print("="*80)
    
    print("\nCrossover Operators:")
    print("-" * 60)
    for cross in cross_names:
        avg_best = np.mean(crossover_results[cross]['best_founds'])
        avg_avg = np.mean(crossover_results[cross]['averages'])
        print(f"  {cross:<15} Best Found Avg: {avg_best:.2f}  |  Average Avg: {avg_avg:.2f}")
    
    print("\nMutation Operators:")
    print("-" * 60)
    for mut in mut_names:
        avg_best = np.mean(mutation_results[mut]['best_founds'])
        avg_avg = np.mean(mutation_results[mut]['averages'])
        print(f"  {mut:<15} Best Found Avg: {avg_best:.2f}  |  Average Avg: {avg_avg:.2f}")
    
    print("\nSelection Operators:")
    print("-" * 60)
    for sel in sel_names:
        avg_best = np.mean(selection_results[sel]['best_founds'])
        avg_avg = np.mean(selection_results[sel]['averages'])
        print(f"  {sel:<15} Best Found Avg: {avg_best:.2f}  |  Average Avg: {avg_avg:.2f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    results = run_experiment()
    print("\n✓ Hoàn thành! Kiểm tra thư mục 'images/' để xem các biểu đồ.")
    print("✓ File kết quả chi tiết: operator_comparison_results.json")