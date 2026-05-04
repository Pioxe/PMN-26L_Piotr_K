import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class CIFAR10_MLP(nn.Module):
    #spłaszczenie danych wejściowych (Flatten) 
    # dynamiczne tworzenie warstw ukrytych (liniowych, aktywacji ReLU i Dropout) na podstawie listy hidden_sizes
    def __init__(self, input_size=3072, num_classes=10, hidden_sizes=[512, 256, 128], dropout_rate=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        current_dim = input_size
        
        # Automatyczne tworzenie warstw na podstawie listy hidden_sizes
        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(current_dim, h_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            current_dim = h_size
            
        self.out = nn.Linear(current_dim, num_classes)
    #przepływ danych przez sieć (przejście w przód). Przetwarza wejściowy obraz przez wszystkie zdefiniowane w konstruktorze warstwy
    def forward(self, x):
        x = self.flatten(x) # 
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

def initialize_model(device, hidden_sizes, dropout_rate):
    return CIFAR10_MLP(hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)

def setup_optimization(model, learning_rate=0.001, optimizer_type='adam'):
    criterion = nn.CrossEntropyLoss()
    opt_class = optim.Adam if optimizer_type.lower() == 'adam' else optim.SGD
    optimizer = opt_class(model.parameters(), lr=learning_rate)
    return criterion, optimizer
#Przygotowuje dane do treningu. 
# Pobiera zbiór CIFAR-10 (jeśli nie istnieje lokalnie), nakłada transformacje (zamiana na tensor i normalizacja) oraz opakowuje dane w obiekty DataLoader
def get_cifar10_loaders(batch_size=64):
    import os
    data_dir = './data'
    download = not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))
    
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=t)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=t)
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), \
           DataLoader(test_set, batch_size=batch_size, shuffle=False)
#Zwraca obiekt torch.device
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    status = f"Wykorzystywane urządzenie: {device.type.upper()}"
    if device.type == 'cuda':
        status += f" ({torch.cuda.get_device_name(0)})"
    print("\n" + "="*len(status))
    print(status)
    print("="*len(status) + "\n")
    return device