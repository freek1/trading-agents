import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Input size = amt of resources 
# (resA, resB, [3 closest neighbors pos x, y])
input_size = 8

[resA, resB, x1, y1, x2, y2, x3, y3] = [1., 1., 0, 0, 4, 11, 25, 28]

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


net = NeuralAgent().to(device)
print(net)

input = torch.tensor([[resA, resB, x1, y1, x2, y2, x3, y3]])
out = net(input)
print(out)




train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

def train_loop(dataloader, net, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    net.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = net(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, net, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    net.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = net(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
