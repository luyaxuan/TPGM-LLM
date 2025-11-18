import numpy as np

# File paths
adj_path = '/root/TPGM-LLM-main/data/adjacency_matrix.npy'
matrix_path = '/root/TPGM-LLM-main/data/matrix.npy'

# Load data
adj = np.load(adj_path)
matrix = np.load(matrix_path)

# Print shape and type
print("✅ adjacency_matrix.npy shape:", adj.shape, "dtype:", adj.dtype)
print("✅ matrix.npy shape:", matrix.shape, "dtype:", matrix.dtype)

# Check if they are square matrices if needed
if adj.shape[0] == adj.shape[1]:
    print("adjacency_matrix.npy is a square matrix ✅")
else:
    print("⚠️ adjacency_matrix.npy is not a square matrix")

if matrix.shape[0] == matrix.shape[1]:
    print("matrix.npy is a square matrix ✅")
else:
    print("⚠️ matrix.npy is not a square matrix")

# Check size relationship
if matrix.shape[0] < adj.shape[0]:
    print(f"matrix.npy ({matrix.shape[0]}×{matrix.shape[1]}) is smaller than adjacency_matrix.npy ({adj.shape[0]}×{adj.shape[1]}), zero-padding is needed.")
elif matrix.shape[0] == adj.shape[0]:
    print("The two matrices have the same size.")
else:
    print("⚠️ matrix.npy is larger than adjacency_matrix.npy, please check the data.")