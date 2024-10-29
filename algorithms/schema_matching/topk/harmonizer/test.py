import numpy as np
from scipy.optimize import linear_sum_assignment
from munkres import Munkres
import time

def maximize_assignment(profit_matrix, method='scipy'):
    """
    Solve the maximization assignment problem using different methods.
    profit_matrix: numpy array where profit_matrix[i,j] is the profit of assigning i to j
    method: 'scipy' or 'munkres'
    """
    if method == 'scipy':
        # For scipy, directly use maximize=True parameter
        row_ind, col_ind = linear_sum_assignment(profit_matrix, maximize=True)
        assignment = list(zip(row_ind, col_ind))
        total_profit = profit_matrix[row_ind, col_ind].sum()
        
    elif method == 'munkres':
        # For Munkres, subtract from a large number to convert max to min
        max_value = np.max(profit_matrix)
        cost_matrix = max_value - profit_matrix
        m = Munkres()
        indices = m.compute(cost_matrix.tolist())
        assignment = indices
        total_profit = sum(profit_matrix[i][j] for i, j in indices)
    
    return assignment, total_profit

def demo_maximization():
    # Create a rectangular profit matrix (3 workers, 4 tasks)
    profit_matrix = np.array([
        [7, 5, 9, 8],
        [6, 4, 3, 7],
        [2, 8, 1, 6]
    ])
    
    print("Profit Matrix (3 workers, 4 tasks):")
    print(profit_matrix)
    print("\nShape:", profit_matrix.shape)
    print("\nTesting different methods:")
    
    for method in ['scipy', 'munkres']:
        assignment, total_profit = maximize_assignment(profit_matrix, method)
        print(f"\n{method.upper()} Method:")
        print(f"Assignment: {assignment}")
        print(f"Total Profit: {total_profit}")

# Benchmark different methods
def benchmark_maximization(rows=100, cols=150, num_runs=5):
    """Compare performance of different maximization methods."""
    profit_matrix = np.random.rand(rows, cols) * 100  # Random profits between 0 and 100
    results = {}
    
    for method in ['scipy', 'munkres']:
        start = time.time()
        for _ in range(num_runs):
            maximize_assignment(profit_matrix, method)
        avg_time = (time.time() - start) / num_runs
        results[method] = avg_time
    
    print(f"\nBenchmark results for {rows}x{cols} matrix:")
    for method, time_taken in results.items():
        print(f"{method:10s}: {time_taken:.4f} seconds")
    
    return results

if __name__ == "__main__":
    # Run demo with rectangular matrix
    demo_maximization()
    
    # Run benchmarks with rectangular matrix
    benchmark_maximization(rows=100, cols=150)