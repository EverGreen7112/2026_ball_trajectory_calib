import torch
import torch.nn as nn
import torch.optim as optim


# 1. ARCHITECTURE: 6 Inputs -> 2 Hidden Layers -> 12 Outputs
class MyNeuralNet(nn.Module):
    def __init__(self):
        super(MyNeuralNet, self).__init__()
        # PyTorch automatically initializes weights/biases upon layer creation
        self.hidden1 = nn.Linear(6, 32)  # Hidden Layer 1 (6 in, 32 out)
        self.hidden2 = nn.Linear(32, 16)  # Hidden Layer 2 (32 in, 16 out)
        self.output = nn.Linear(16, 12)  # Output Layer (16 in, 12 out)
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)  # Final 12 outputs
        return x


# 2. SETUP MODEL & OPTIMIZER
model = MyNeuralNet()
criterion = nn.MSELoss()  # Standard for training toward specific values (regression)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. SETTING TARGETS & DATA
# Inputs must have 6 features; targets must have 12 values to match output
inputs = torch.randn(10, 6)  # 10 samples of 6 features
targets = torch.randn(10, 12)  # 10 samples of 12 "correct" target values

# 4. TRAINING LOOP WITH BACKPROPAGATION
for epoch in range(100):
    model.train()

    # Forward Pass: Compute current predictions
    predictions = model(inputs)

    # Calculate Loss: How far are we from the targets?
    loss = criterion(predictions, targets)

    # Backpropagation Steps:
    optimizer.zero_grad()  # 1. Clear old gradients from previous step
    loss.backward()  # 2. Calculate gradients using backpropagation
    optimizer.step()  # 3. Update weights toward the targets

    if (epoch + 1) < 100:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")