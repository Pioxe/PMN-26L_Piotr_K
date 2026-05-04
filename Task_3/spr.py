import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# =====================================================================
# 1. OPTYMALIZACJA GRADIENTOWA (Wizualizacja)
# =====================================================================
def loss_function(w1, w2):
    return w1**2 + w2**2 + 5

def gradient_descent(start_w1, start_w2, lr, steps):
    w1, w2 = start_w1, start_w2
    history = [(w1, w2, loss_function(w1, w2))]
    
    for _ in range(steps):
        dw1 = 2 * w1
        dw2 = 2 * w2
        
        w1 = w1 - lr * dw1
        w2 = w2 - lr * dw2
        history.append((w1, w2, loss_function(w1, w2)))
        
    return np.array(history)

lr = 0.1
steps = 25
history = gradient_descent(start_w1=8, start_w2=8, lr=lr, steps=steps)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

W1 = np.arange(-10, 10, 0.5)
W2 = np.arange(-10, 10, 0.5)
W1, W2 = np.meshgrid(W1, W2)
Z = loss_function(W1, W2)

surf = ax.plot_surface(W1, W2, Z, cmap=cm.coolwarm, alpha=0.6, antialiased=True)

ax.plot(history[:, 0], history[:, 1], history[:, 2], 'r-o', markersize=5, label='Kroki optymalizatora', zorder=10)
ax.scatter(0, 0, 5, color='green', s=100, label='Globalne Optimum')

ax.set_xlabel('Waga w1')
ax.set_ylabel('Waga w2')
ax.set_zlabel('Loss (Błąd)')
ax.set_title(f'Optymalizacja Gradientowa (Learning Rate = {lr})')
ax.legend()

plt.show()

# =====================================================================
# 2. PRZYGOTOWANIE DANYCH (FashionMNIST)
# =====================================================================
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

print(f"Rozmiar zbioru treningowego: {len(train_dataset)}")
print(f"Rozmiar zbioru walidacyjnego: {len(val_dataset)}")
print(f"Rozmiar zbioru testowego: {len(test_dataset)}\n")

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
    plt.title(f"Etykieta: {example_targets[i]}")
    plt.axis('off')
plt.show()

# =====================================================================
# 3. DEFINICJA MODELU SIECI NEURONOWEJ
# =====================================================================
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),  
            nn.Linear(256, 128),
            nn.ReLU(),  
            nn.Linear(128, 10)
        )
    def forward(self, x): 
        return self.net(x)

# =====================================================================
# 4. TEST PRĘDKOŚCI CPU vs GPU (Na wycinku danych)
# =====================================================================
def train_and_time(device_name):
    device = torch.device(device_name)
    model = SimpleMLP().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        
        if batch_idx > 100: break 

    end_time = time.time()
    return end_time - start_time

print("--- Rozpoczynam test prędkości ---")
time_cpu = train_and_time("cpu")
print(f"Czas na CPU: {time_cpu:.2f} sekund")

if torch.cuda.is_available():
    time_gpu = train_and_time("cuda")
    print(f"Czas na GPU (RTX 4050): {time_gpu:.2f} sekund")
    print(f"Przyspieszenie GPU wzgledem CPU: {time_cpu / time_gpu:.1f}x\n")
else:
    print("GPU niedostępne dla testu.\n")

