#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time

# ==============================
# File path configuration
# ==============================
INPUT_NPY_PATH = r"processed_data_1.npy"               # Input data file path
OUTPUT_DTW_NPY = "./data_010/fast_test_sensor_1.npy"  # DTW distance matrix output path
OUTPUT_ADJ_CSV = "./data_010/adj_010.csv"             # Weighted adjacency matrix CSV output path
OUTPUT_ADJ_NPY = "./data_010/adj_010.npy"             # Weighted adjacency matrix NPY output path

# Key parameter configuration
tr_day_ratio = 0.1  # Ratio of training data to total days (for tr_day calculation)
adj_percent = 0.10  # Top ratio for adjacency matrix (to filter most similar nodes)


def gen_data(data, ntr, N):
    """Reshape data into required format"""
    data = np.reshape(data, [-1, 360, N])
    print(f"gen_data: data reshaped to {data.shape}")
    return data[0:ntr]


def normalize(a):
    """Normalize the input array"""
    mu = np.mean(a, axis=1, keepdims=True)
    std = np.std(a, axis=1, keepdims=True)
    return (a - mu) / std


def compute_dtw(a, b, o=1, T=15):
    """Compute DTW distance between two time series"""
    a = normalize(a)
    b = normalize(b)
    # Reshape and calculate difference
    d = np.reshape(a, [-1, 1, T0]) - np.reshape(b, [-1, T0, 1])
    d = np.linalg.norm(d, axis=0, ord=o)
    
    # Initialize DTW matrix
    D = np.zeros([T0, T0])
    for i in range(T0):
        for j in range(max(0, i - T), min(T0, i + T + 1)):
            if (i == 0) and (j == 0):
                D[i, j] = d[i, j] ** o
                continue
            if i == 0:
                D[i, j] = d[i, j] ** o + D[i, j - 1]
                continue
            if j == 0:
                D[i, j] = d[i, j] ** o + D[i - 1, j]
                continue
            if j == i - T:
                D[i, j] = d[i, j] ** o + min(D[i - 1, j - 1], D[i - 1, j])
                continue
            if j == i + T:
                D[i, j] = d[i, j] ** o + min(D[i - 1, j - 1], D[i, j - 1])
                continue
            D[i, j] = d[i, j] ** o + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[-1, -1] ** (1.0 / o)


def csv_to_npy(csv_path, npy_path):
    """Convert CSV adjacency matrix to NPY format"""
    # Read CSV file (no header and index as saved previously)
    df = pd.read_csv(csv_path, header=None)
    
    # Convert to numpy array
    adj_matrix = df.to_numpy()
    
    # Save as NPY file
    np.save(npy_path, adj_matrix)
    
    # Print conversion information
    print(f"Successfully converted CSV to NPY file!")
    print(f"Input CSV path: {csv_path}")
    print(f"Output NPY path: {npy_path}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")


if __name__ == "__main__":
    # Load data and display dimensions
    data = np.load(INPUT_NPY_PATH)
    print(f"Loaded data shape: {data.shape}")
    
    total_day = data.shape[0] / 360
    print(f"Total number of days (total_day): {total_day}")
    tr_day = int(total_day * tr_day_ratio)
    n_route = data.shape[1]
    print(f"Number of routes (n_route): {n_route}")
    
    # Generate training data
    xtr = gen_data(data, tr_day, n_route)
    print(f"xtr shape after gen_data: {xtr.shape}")
    
    # DTW calculation parameters
    T0 = 360
    T = 15
    N = n_route
    d = np.zeros([N, N])
    print(f"Distance matrix d shape: {d.shape}")
    
    # Compute DTW distance matrix
    for i in range(N):
        t1 = time.time()
        for j in range(i + 1, N):
            d[i, j] = compute_dtw(xtr[:, :, i], xtr[:, :, j])
        t2 = time.time()
        print(f"Time for row {i}: {t2 - t1} seconds")
        print("=======================")
    
    # Save DTW distance matrix
    np.save(OUTPUT_DTW_NPY, d)
    print(f"Saved distance matrix to {OUTPUT_DTW_NPY}, shape: {d.shape}")
    print("Time series calculation completed!")
    
    # Process adjacency matrix
    adj = np.load(OUTPUT_DTW_NPY)
    adj = adj + adj.T  # Symmetrize the matrix
    print(f"Adjacency matrix after symmetrization shape: {adj.shape}")
    
    n = adj.shape[0]
    w_adj = np.zeros([n, n])
    print(f"Weighted adjacency matrix w_adj shape before processing: {w_adj.shape}")
    
    # Calculate top similar nodes
    top = int(n * adj_percent)
    print(f"Top percent: {adj_percent}, Top routes: {top}")
    
    # Build weighted adjacency matrix
    for i in range(adj.shape[0]):
        a = adj[i, :].argsort()[0:top]
        for j in range(top):
            w_adj[i, a[j]] = 1
    
    # Ensure symmetry and self-connections
    for i in range(n):
        for j in range(n):
            if w_adj[i][j] != w_adj[j][i] and w_adj[i][j] == 0:
                w_adj[i][j] = 1
            if i == j:
                w_adj[i][j] = 1
    
    # Print adjacency matrix information
    print(f"Final weighted adjacency matrix w_adj shape: {w_adj.shape}")
    print(f"Total route number: {n}")
    print(f"Density of adjacency matrix: {len(w_adj.nonzero()[0]) / (n * n)}")
    
    # Save as CSV
    ww = pd.DataFrame(w_adj)
    ww.to_csv(OUTPUT_ADJ_CSV, index=False, header=None)
    print(f"Temporal graph weighted matrix generated and saved to {OUTPUT_ADJ_CSV}!")
    
    # Convert CSV to NPY
    csv_to_npy(OUTPUT_ADJ_CSV, OUTPUT_ADJ_NPY)