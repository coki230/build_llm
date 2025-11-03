from typing import Self

import torch


class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / 1.41421)) * (x + 0.044715 * torch.pow(x, 3.0))

# x = torch.arange(-3.0, 3.0, 0.1)
# y = GELU()(x)
# y2 = torch.nn.functional.gelu(x)
#
# import matplotlib.pyplot as plt
# plt.plot(x, y)
# plt.plot(x, y2)
# plt.show()

class ExampleDeepNeuralNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, use_shortcut=False):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[0], layer_sizes[1], GELU())),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[1], layer_sizes[2], GELU())),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[2], layer_sizes[3], GELU())),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[3], layer_sizes[4], GELU())),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[4], layer_sizes[5], GELU())),
        ])

    def forward(self, x):
        for layer in self.layers:
            output = layer(x)
            if self.use_shortcut and output.shape == x.shape:
                x = x + output
            else:
                x = output

        return x

layer_sizes = [3, 3, 3, 3, 3, 1]
input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
model = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)

def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = torch.nn.MSELoss()(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        # check name contains "weight"
        if "weight" in name:
            print(name, "-->", param.grad.mean())
    print(loss)


print_gradients(model, input)