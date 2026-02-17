"""
Sorting Assignment Starter Code
Implement five sorting algorithms and benchmark their performance.
"""

import json
import time
import random
import tracemalloc


# ============================================================================
# PART 1: SORTING IMPLEMENTATIONS
# ============================================================================

def bubble_sort(arr):
    """
    Sort array using bubble sort algorithm.
    
    Bubble sort repeatedly steps through the list, compares adjacent elements,
    and swaps them if they're in the wrong order.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        bubble_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def selection_sort(arr):
    """
    Sort array using selection sort algorithm.
    
    Selection sort divides the list into sorted and unsorted regions, repeatedly
    selecting the minimum element from unsorted region and moving it to sorted region.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        selection_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr):
    """
    Sort array using insertion sort algorithm.
    
    Insertion sort builds the final sorted array one item at a time, inserting
    each element into its proper position in the already-sorted portion.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        insertion_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr):
    """
    Sort array using merge sort algorithm.
    
    Merge sort is a divide-and-conquer algorithm that divides the array into halves,
    recursively sorts them, and then merges the sorted halves.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        merge_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return _merge(left, right)


def _merge(left, right):
    """
    Helper function to merge two sorted arrays.
    
    Args:
        left (list): First sorted array
        right (list): Second sorted array
    
    Returns:
        list: Merged sorted array
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# ============================================================================
# PART 2: STABILITY DEMONSTRATION
# ============================================================================

def demonstrate_stability():
    """
    Demonstrate which sorting algorithms are stable by sorting products by price.
    
    Creates a list of product dictionaries with prices and original order.
    Sorts by price and checks if products with same price maintain original order.
    
    Returns:
        dict: Results showing which algorithms preserved order for equal elements
    """
    # Sample products with duplicate prices
    products = [
        {"name": "Widget A", "price": 1999, "original_position": 0},
        {"name": "Gadget B", "price": 999, "original_position": 1},
        {"name": "Widget C", "price": 1999, "original_position": 2},
        {"name": "Tool D", "price": 999, "original_position": 3},
        {"name": "Widget E", "price": 1999, "original_position": 4},
    ]
    
    # Helper class that compares products by price only
    # This allows us to test if algorithms preserve original order for equal prices
    class PriceComparable:
        def __init__(self, price, original_position, name):
            self.price = price
            self.original_position = original_position
            self.name = name
        
        def __lt__(self, other):
            return self.price < other.price
        
        def __le__(self, other):
            return self.price <= other.price
        
        def __gt__(self, other):
            return self.price > other.price
        
        def __ge__(self, other):
            return self.price >= other.price
        
        def __eq__(self, other):
            return self.price == other.price
    
    # Create comparable objects for testing
    comparable_products = [
        PriceComparable(p["price"], p["original_position"], p["name"])
        for p in products
    ]
    
    # What a stable sort should produce (using Python's guaranteed stable sort)
    expected_stable = sorted(comparable_products, key=lambda x: (x.price, x.original_position))
    
    algorithms = {
        "bubble_sort": bubble_sort,
        "selection_sort": selection_sort,
        "insertion_sort": insertion_sort,
        "merge_sort": merge_sort
    }
    
    results = {}
    for algo_name, algo_func in algorithms.items():
        # Create a fresh copy of comparable products for each algorithm
        test_products = [
            PriceComparable(p["price"], p["original_position"], p["name"])
            for p in products
        ]
        
        # Sort using the algorithm
        sorted_result = algo_func(test_products)
        
        # Check if stable by comparing original positions with expected stable sort
        is_stable = all(
            result.original_position == expected.original_position
            for result, expected in zip(sorted_result, expected_stable)
        )
        
        results[algo_name] = "Stable" if is_stable else "Unstable"
    
    return results


# ============================================================================
# PART 3: PERFORMANCE BENCHMARKING
# ============================================================================

def load_dataset(filename):
    """Load a dataset from JSON file."""
    with open(f"datasets/{filename}", "r") as f:
        return json.load(f)


def load_test_cases():
    """Load test cases for validation."""
    with open("datasets/test_cases.json", "r") as f:
        return json.load(f)


def test_sorting_correctness():
    """Test that sorting functions work correctly on small test cases."""
    print("="*70)
    print("TESTING SORTING CORRECTNESS")
    print("="*70 + "\n")
    
    test_cases = load_test_cases()
    
    test_names = ["small_random", "small_sorted", "small_reverse", "small_duplicates"]
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort
    }
    
    for test_name in test_names:
        print(f"Test: {test_name}")
        print(f"  Input:    {test_cases[test_name]}")
        print(f"  Expected: {test_cases['expected_sorted'][test_name]}")
        print()
        
        for algo_name, algo_func in algorithms.items():
            try:
                result = algo_func(test_cases[test_name].copy())
                expected = test_cases['expected_sorted'][test_name]
                status = "✓ PASS" if result == expected else "✗ FAIL"
                print(f"    {algo_name:20s}: {result} {status}")
            except Exception as e:
                print(f"    {algo_name:20s}: ERROR - {str(e)}")
        
        print()


def benchmark_algorithm(sort_func, data):
    """
    Benchmark a sorting algorithm on given data.
    
    Args:
        sort_func: The sorting function to test
        data: The dataset to sort (will be copied so original isn't modified)
    
    Returns:
        tuple: (execution_time_ms, peak_memory_kb)
    """
    # Copy data so we don't modify original
    data_copy = data.copy()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Measure execution time
    start_time = time.perf_counter()
    sort_func(data_copy)
    end_time = time.perf_counter()
    
    # Get peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time_ms = (end_time - start_time) * 1000
    peak_memory_kb = peak / 1024
    
    return execution_time_ms, peak_memory_kb


def benchmark_all_datasets():
    """Benchmark all sorting algorithms on all datasets."""
    print("\n" + "="*70)
    print("BENCHMARKING SORTING ALGORITHMS")
    print("="*70 + "\n")
    
    datasets = {
        "orders.json": ("Order Processing Queue", 50000, 5000),
        "products.json": ("Product Catalog", 100000, 5000),
        "inventory.json": ("Inventory Reconciliation", 25000, 5000),
        "activity_log.json": ("Customer Activity Log", 75000, 5000)
    }
    
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort
    }
    
    for filename, (description, full_size, sample_size) in datasets.items():
        print(f"Dataset: {description} ({sample_size:,} element sample)")
        print("-" * 70)
        
        data = load_dataset(filename)
        # Use first sample_size elements for fair comparison
        data_sample = data[:sample_size]
        
        for algo_name, algo_func in algorithms.items():
            try:
                exec_time, memory = benchmark_algorithm(algo_func, data_sample)
                print(f"  {algo_name:20s}: {exec_time:8.2f} ms | {memory:8.2f} KB")
            except Exception as e:
                print(f"  {algo_name:20s}: ERROR - {str(e)}")
        
        print()


def analyze_stability():
    """Test and display which algorithms are stable."""
    print("="*70)
    print("STABILITY ANALYSIS")
    print("="*70 + "\n")
    
    print("Testing which algorithms preserve order of equal elements...\n")
    
    results = demonstrate_stability()
    
    for algo_name, stability in results.items():
        print(f"  {algo_name:20s}: {stability}")
    
    print()


if __name__ == "__main__":
    print("SORTING ASSIGNMENT - STARTER CODE")
    print("Implement the sorting functions above, then run tests.\n")
    
    # Uncomment these as you complete each part:
    
    test_sorting_correctness()
    analyze_stability()
    benchmark_all_datasets()