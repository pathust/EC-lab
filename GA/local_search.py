import random
import math

# ========================
# Đọc dữ liệu & chuẩn bị
# ========================
def read_coords(filename):
    coords = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                idx, x, y = map(int, parts)
                coords[idx] = (x, y)
    return coords

def calc_distance_matrix(coords):
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    nodes = sorted(coords.keys())
    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = coords[nodes[i]]
                xj, yj = coords[nodes[j]]
                dist[i][j] = math.hypot(xi - xj, yi - yj)
    return dist, nodes

def route_length(route, dist):
    total = 0.0
    n = len(route)
    for i in range(n):
        total += dist[route[i]][route[(i + 1) % n]]
    return total

# ========================
# 3 phương pháp tối ưu
# ========================
def swap_local_search(route, dist, max_iter=1000):
    n = len(route)
    best_route = route[:]
    best_len = route_length(best_route, dist)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = best_route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                new_len = route_length(new_route, dist)
                if new_len < best_len:
                    best_len = new_len
                    best_route = new_route
                    improved = True
        if not improved:
            break
    return best_route, best_len

def two_opt(route, dist, max_iter=1000):
    n = len(route)
    best_route = route[:]
    best_len = route_length(best_route, dist)

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                new_len = route_length(new_route, dist)
                if new_len < best_len:
                    best_len = new_len
                    best_route = new_route
                    improved = True
        if not improved:
            break
    return best_route, best_len

def hybrid_local_search(route, dist, max_iter=1000):
    n = len(route)
    best_route = route[:]
    best_len = route_length(best_route, dist)

    for _ in range(max_iter):
        improved = False
        # Swap
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = best_route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                new_len = route_length(new_route, dist)
                if new_len < best_len:
                    best_len = new_len
                    best_route = new_route
                    improved = True
        # 2-opt
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                new_len = route_length(new_route, dist)
                if new_len < best_len:
                    best_len = new_len
                    best_route = new_route
                    improved = True
        if not improved:
            break
    return best_route, best_len

# ========================
# Main
# ========================
def main():
    coords = read_coords("GA/eil101.dat")
    dist, nodes = calc_distance_matrix(coords)

    # set seed để so sánh công bằng
    random.seed(31)
    init_route = list(range(len(nodes)))
    random.shuffle(init_route)

    print("Chiều dài ban đầu:", route_length(init_route, dist))

    # Swap only
    r1, l1 = swap_local_search(init_route, dist)
    print("\n[Swap]   Độ dài:", l1)
    print("Lộ trình:", [nodes[i] for i in r1])

    # 2-opt only
    r2, l2 = two_opt(init_route, dist)
    print("\n[2-opt]  Độ dài:", l2)
    print("Lộ trình:", [nodes[i] for i in r2])

    # Hybrid
    r3, l3 = hybrid_local_search(init_route, dist)
    print("\n[Hybrid] Độ dài:", l3)
    print("Lộ trình:", [nodes[i] for i in r3])

if __name__ == "__main__":
    main()
