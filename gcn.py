import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = th.device('cuda:0')

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix.to(device)
    matrix = matrix + th.eye(matrix.size(0)).to(device)
    row_sum = matrix.sum(1)
    d_inv_sqrt = th.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[th.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = th.diag(d_inv_sqrt)
    normalized_laplacian = th.matmul(matrix, d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian

class GCN(nn.Module):
    def __init__(self, adj=np.load('/root/TPGM-LLM-main/data/adjacency_matrix.npy'), input_dim=15, hidden_dim=64, output_dim=256):
        super(GCN, self).__init__()
        # Compute the Laplacian matrix only once to avoid recalculating it in each forward pass
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(th.FloatTensor(adj).to(device=device))
        )
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim  # seq_len for prediction
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim  # output_dim for prediction
        self.weights1 = nn.Parameter(th.FloatTensor(self._input_dim, self._hidden_dim))
        self.weights2 = nn.Parameter(th.FloatTensor(self._hidden_dim, self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain("sigmoid"))

    def forward(self, inputs):
        # Step 1: Transpose and select the 0th dimension (batch_size, 15, 1296, 3) -> (batch_size, 15, 1296)
        inputs = inputs.transpose(1, 3)[..., 0]  # (batch_size, 15, 1296)

        # Step 2: Transpose and reshape (batch_size, 15, 1296) -> (1296, batch_size, 15)
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(0, 2).transpose(1, 2)  # (1296, batch_size, 15)

        # Step 3: Reshape to (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape(self._num_nodes, batch_size * self._input_dim)  # (num_nodes, batch_size * seq_len)

        # Step 4: AX (num_nodes, batch_size * seq_len)
        ax = th.matmul(self.laplacian, inputs)  # (num_nodes, batch_size * seq_len)

        # Step 5: Reshape to (num_nodes, batch_size, seq_len)
        ax = ax.view(self._num_nodes, batch_size, self._input_dim)  # (num_nodes, batch_size, seq_len)

        # Step 6: Activate and perform matrix multiplication
        ax_reshaped = ax.view(self._num_nodes * batch_size, self._input_dim)
        outputs = th.relu(th.matmul(ax_reshaped, self.weights1))  # (num_nodes * batch_size, hidden_dim)

        # Step 7: Reshape to (num_nodes, batch_size * hidden_dim)
        outputs = outputs.view(self._num_nodes, batch_size * self._hidden_dim)  # (num_nodes, batch_size * hidden_dim)

        # Step 8: AX (num_nodes, batch_size * hidden_dim)
        ax = th.matmul(self.laplacian, outputs)  # (num_nodes, batch_size * hidden_dim)

        # Step 9: Reshape to (num_nodes, batch_size, hidden_dim)
        ax = ax.view(self._num_nodes, batch_size, self._hidden_dim)  # (num_nodes, batch_size, hidden_dim)

        # Step 10: Activate and perform matrix multiplication
        ax_reshaped = ax.view(self._num_nodes * batch_size, self._hidden_dim)
        outputs = th.sigmoid(th.matmul(ax_reshaped, self.weights2))  # (num_nodes * batch_size, output_dim)

        # Step 11: Reshape to (num_nodes, batch_size, output_dim)
        outputs = outputs.view(self._num_nodes, batch_size, self._output_dim)  # (num_nodes, batch_size, output_dim)

        # Step 12: Final transpose (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1).transpose(1, 2)  # (batch_size, num_nodes, output_dim)

        return outputs