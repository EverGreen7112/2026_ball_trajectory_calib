import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ==========================================
# 1. THE DATA GENERATOR (Simulating 50 Shots)
# ==========================================
def generate_robot_data(num_samples=50):
    data = []
    for _ in range(num_samples):
        # Inputs: Vx, Vy, RPM, Pitch, Feed, Launch_V
        robot_vx = np.random.uniform(-1.0, 1.0)
        robot_vy = np.random.uniform(-1.0, 1.0)
        rpm = np.random.uniform(500, 3000)
        pitch = np.random.uniform(10, 60)
        feed = np.random.uniform(0.5, 2.0)
        v_launch = np.random.uniform(10, 30)

        # Simple physics simulation to create "Target" polynomials
        t = np.linspace(0, 2, 100)
        p_rad = np.radians(pitch)

        # Calculate X, Y, Z paths with slight non-linear drag/lift
        # This gives the NN something "physical" to learn
        x_p = (v_launch * np.cos(p_rad) + robot_vx) * t - (0.01 * v_launch) * t ** 2
        y_p = (v_launch * 0.1) * t + (rpm / 3000) * 0.2 * t ** 3
        z_p = (v_launch * np.sin(p_rad)) * t - 4.9 * t ** 2 - (rpm / 5000) * t ** 3

        # Extract 12 coefficients (3 axes * 4 params for 3rd degree)
        px = np.polyfit(t, x_p, 3)
        py = np.polyfit(t, y_p, 3)
        pz = np.polyfit(t, z_p, 3)

        inputs = [robot_vx, robot_vy, rpm, pitch, feed, v_launch]
        targets = np.concatenate([px, py, pz])
        data.append(list(inputs) + list(targets))

    return np.array(data, dtype=np.float32)


# ==========================================
# 2. THE NANO-SHOOTER NETWORK
# ==========================================
class NanoShooterNet(nn.Module):
    def __init__(self):
        super(NanoShooterNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 16),
            nn.Tanh(),  # Smooth curves for physics
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 12)  # 12 Linear outputs (Coefficients)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 3. EXECUTION & TRAINING
# ==========================================

# A. Generate & Prepare Data
raw_data = generate_robot_data(50)
X_raw = raw_data[:, :6]
Y_raw = raw_data[:, 6:]

# B. Normalization (Critical for 50 samples)
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_scaled = (X_raw - X_mean) / (X_std + 1e-6)

inputs = torch.from_numpy(X_scaled)
targets = torch.from_numpy(Y_raw)

# C. Initialize Model
model = NanoShooterNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)

print("Starting Training on 50 samples...\n")
print(f"{'Epoch':<10} | {'Cost (MSE Loss)':<20}")
print("-" * 35)

# D. Training Loop
for epoch in range(2001):
    model.train()

    # Forward Pass
    predictions = model(inputs)
    loss = criterion(predictions, targets)

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 100 == 0:
        print(f"{epoch:<10} | {loss.item():<20.8f}")


# ==========================================
# 4. TEST PREDICTION
# ==========================================
def predict_shot(test_settings):
    model.eval()
    with torch.no_grad():
        # Scale the new input exactly like the training data
        scaled = (test_settings - X_mean) / (X_std + 1e-6)
        scaled_tensor = torch.from_numpy(scaled.astype(np.float32))
        return model(scaled_tensor).numpy()


# Test with a random robot setting
sample_robot_input = np.array([0.5, -0.2, 2500, 35, 1.0, 20.0])
result = predict_shot(sample_robot_input)

print("\nTraining Complete!")
print("-" * 35)
print("Example Prediction (12 Coefficients):")
print(result)


def save_model_to_json(model, x_mean, x_std, filename="shooter_model.json"):
    # Extract the weights/biases from the state_dict
    state_dict = model.state_dict()

    # Convert tensors to lists so JSON can handle them
    model_data = {
        "metadata": {
            "mean": x_mean.tolist(),
            "std": x_std.tolist()
        },
        "weights": {key: val.cpu().numpy().tolist() for key, val in state_dict.items()}
    }

    with open(filename, "w") as f:
        json.dump(model_data, f)

    print(f"\nModel and scaling data saved to {filename}")


# Call the function after training
save_model_to_json(model, X_mean, X_std)