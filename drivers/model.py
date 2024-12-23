# Import necessary libraries
import sys
sys.path.insert(0, "../")
from core.__importList__ import *
from core.config import *

# Initialize weights with zeros and bias with 0.01
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.01)


# Custom linear layer with positive weights and no bias
class ConvexLinear(torch.nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = torch.nn.Parameter(torch.Tensor(size_out, size_in))
        
        # Initialize weights with Kaiming uniform distribution
        torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

    def forward(self, x):
        return torch.mm(x, torch.nn.functional.softplus(self.weights.t()))


# Feed-Forward Neural Network with skip connections
class FFNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.layers = torch.nn.ModuleDict()
        self.skip_layers = torch.nn.ModuleDict()
        self.depth = len(n_hidden)

        # Input layer
        self.layers['0'] = torch.nn.Linear(n_input, n_hidden[0]).float()

        # Hidden layers with skip connections
        for i in range(1, self.depth):
            self.layers[str(i)] = torch.nn.Linear(n_hidden[i - 1], n_hidden[i]).float()
            self.skip_layers[str(i)] = torch.nn.Linear(n_input, n_hidden[i]).float()

        # Output layer
        self.layers[str(self.depth)] = torch.nn.Linear(n_hidden[-1], n_output).float()
        self.skip_layers[str(self.depth)] = torch.nn.Linear(n_input, n_output).float()

    def forward(self, x):
        z = self.layers['0'](x.clone())
        for i in range(1, self.depth):
            skip = self.skip_layers[str(i)](x)
            z = self.layers[str(i)](z) + skip
            z = torch.nn.functional.softplus(z)

        y = self.layers[str(self.depth)](z) + self.skip_layers[str(self.depth)](x)
        return y


# Partial Input Convex Neural Network (PICNN)
class PICNN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        bias_opt = True
        self.depth = len(n_hidden)
        self.n_input = n_input
        self.y_size = 1
        self.sftpSq_scaling = 1 / n_hidden[-1]

        # Initialize dictionaries for different layer types
        self.u1_layers = torch.nn.ModuleDict()
        self.zu_layers = torch.nn.ModuleDict()
        self.zz_layers = torch.nn.ModuleDict()
        self.yu_layers = torch.nn.ModuleDict()
        self.yy_layers = torch.nn.ModuleDict()
        self.u2_layers = torch.nn.ModuleDict()

        # Initial layers
        self.u1_layers['0'] = torch.nn.Linear(n_input - self.y_size, n_hidden[0], bias=bias_opt).float()
        self.zu_layers['0'] = torch.nn.Linear(n_input - self.y_size, self.y_size, bias=bias_opt).float()
        self.zz_layers['0'] = ConvexLinear(self.y_size, n_hidden[0]).float()
        self.u2_layers['0'] = torch.nn.Linear(n_input - self.y_size, n_hidden[0], bias=bias_opt).float()

        # Hidden layers
        for i in range(1, self.depth):
            self.u1_layers[str(i)] = torch.nn.Linear(n_hidden[i - 1], n_hidden[i], bias=bias_opt).float()
            self.zu_layers[str(i)] = torch.nn.Linear(n_hidden[i - 1], n_hidden[i], bias=bias_opt).float()
            self.zz_layers[str(i)] = ConvexLinear(n_hidden[i - 1], n_hidden[i]).float()
            self.yu_layers[str(i)] = torch.nn.Linear(n_hidden[i - 1], self.y_size, bias=bias_opt).float()
            self.yy_layers[str(i)] = torch.nn.Linear(self.y_size, n_hidden[i], bias=bias_opt).float()
            self.u2_layers[str(i)] = torch.nn.Linear(n_hidden[i - 1], n_hidden[i], bias=bias_opt).float()

        # Final layer
        self.zu_layers[str(self.depth)] = torch.nn.Linear(n_hidden[-1], n_hidden[-1], bias=bias_opt).float()
        self.zz_layers[str(self.depth)] = ConvexLinear(n_hidden[-1], n_output).float()
        self.yu_layers[str(self.depth)] = torch.nn.Linear(n_hidden[-1], self.y_size, bias=bias_opt).float()
        self.yy_layers[str(self.depth)] = torch.nn.Linear(self.y_size, n_output).float()
        self.u2_layers[str(self.depth)] = torch.nn.Linear(n_hidden[-1], n_output, bias=bias_opt).float()

    # Forward methods for different layer types
    def forward_u1(self, u, layer):
        return torch.nn.functional.relu(self.u1_layers[str(layer)](u))

    def forward_zu(self, u, layer):
        return torch.nn.functional.relu(self.zu_layers[str(layer)](u))

    def forward_zz(self, z, zu, layer):
        return self.zz_layers[str(layer)](z * zu)

    def forward_yu(self, u, layer):
        return self.yu_layers[str(layer)](u)

    def forward_yy(self, y, yu, layer):
        return self.yy_layers[str(layer)](y * yu)

    def forward_u2(self, u, layer):
        return self.u2_layers[str(layer)](u)

    def forward(self, y, x):
        # First layer computations
        u1 = self.forward_u1(x, layer=0)
        zu = self.forward_zu(x, layer=0)
        zz = self.forward_zz(y, zu, layer=0)
        u2 = self.forward_u2(x, layer=0)
        z = self.sftpSq_scaling * torch.square(torch.nn.functional.softplus(zz + u2))

        # Iterate over hidden layers
        for i in range(1, self.depth + 1):
            u_ = u1.clone()
            if i < self.depth:
                u1 = self.forward_u1(u_, layer=i)
            zu = self.forward_zu(u_, layer=i)
            zz = self.forward_zz(z, zu, layer=i)
            yu = self.forward_yu(u_, layer=i)
            yy = self.forward_yy(y, yu, layer=i)
            u2 = self.forward_u2(u_, layer=i)

            if i < self.depth:
                z = self.sftpSq_scaling * torch.square(torch.nn.functional.softplus(zz + yy + u2))
            else:
                z = self.sftpSq_scaling * torch.square(torch.nn.functional.softplus(zz + yy))

        return z
