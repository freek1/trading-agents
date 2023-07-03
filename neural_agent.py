import torch
import torch.nn as nn
import torch.nn.functional as F

# Input size = amt of resources 
# (resA, resB, [5 closest neighbors pos])
input_size = 7
# Output size = coordinates of goal position
output_size = 2

class NeuralAgent(nn.Module):

    def __init__(self):
        super(NeuralAgent, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, 40) 
        self.fc2 = nn.Linear(40, 20)
        self.fc3 = nn.Linear(20, 40)
        self.fc4 = nn.Linear(40, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = NeuralAgent()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1, input_size)
input = torch.tensor([[1., 1., .1, .3, .4, .2, .1]])
out = net(input)
print(out)


