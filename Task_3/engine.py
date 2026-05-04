import torch
from sklearn.metrics import classification_report
import time, torch

from class_cifar10 import get_cifar10_loaders
#przebieg przez epokę
# fazę trenowania (z aktualizacją wag)
# walidacji (bez obliczania gradientów)
#  Zwraca średnią wartość straty oraz procentową dokładność dla danej fazy.
def run_step(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss, correct, total = 0.0, 0, 0
    
    with torch.set_grad_enabled(is_train):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if is_train: optimizer.zero_grad()
            outputs = model(inputs)
            current_loss = criterion(outputs, labels)
            if is_train:
                current_loss.backward()
                optimizer.step()
            
            loss += current_loss.item() * inputs.size(0)
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
    return loss / len(loader.dataset), 100. * correct / total
#Koordynuje pełny cykl treningowy modelu dla określonej liczby epok
def run_experiment(device, config, train_loader, test_loader):
    from class_cifar10 import initialize_model, setup_optimization
    model = initialize_model(device, config["hidden_layers"], config["dropout"])
    criterion, optimizer = setup_optimization(model, config["lr"], config["optimizer_type"])
    
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    for epoch in range(config["epochs"]):
        t_loss, t_acc = run_step(model, train_loader, criterion, device, optimizer)
        v_loss, v_acc = run_step(model, test_loader, criterion, device)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        print(f"E{epoch+1} Val Acc: {v_acc:.2f}% | Val Loss: {v_loss:.4f}")
    
    # Zwracamy również model, aby móc go później przetestować
    return history, model
#ocena wytrenowanego modelu
def evaluate_best_model(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Zapisywanie błędnie sklasyfikowanych obrazów
            for i in range(len(preds)):
                if preds[i] != labels[i] and len(misclassified) < 10:
                    misclassified.append((inputs[i].cpu(), labels[i].item(), preds[i].item()))

    print("\n" + "="*60)
    print("RAPORT KLASYFIKACJI (Classification Report):")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return all_labels, all_preds, misclassified

#test wydajnościowy porównujący czas trenowania jednej epoki
def compare_hardware_speed(device_gpu, base_cfg):
    print("\n" + "="*40)
    print("TEST PORÓWNAWCZY: CPU vs CUDA (1 Epoki)")
    print("="*40)

    # Przygotowanie danych
    tr, te = get_cifar10_loaders(batch_size=base_cfg["batch_size"])
    
    # --- TEST NA CUDA ---
    print(f"Rozpoczynam test na: {device_gpu}")
    start_gpu = time.time()
    run_experiment(device_gpu, {**base_cfg, "epochs": 1}, tr, te)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    # --- TEST NA CPU ---
    device_cpu = torch.device("cpu")
    print(f"\nRozpoczynam test na: CPU")
    start_cpu = time.time()
    # Ważne: model musi zostać stworzony od nowa na CPU
    run_experiment(device_cpu, {**base_cfg, "epochs": 1}, tr, te)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # Podsumowanie
    print("\n" + "="*30)
    print(f"CZAS CUDA: {gpu_time:.2f} sekundy")
    print(f"CZAS CPU : {cpu_time:.2f} sekundy")
    print(f"CUDA jest szybsze o: {cpu_time/gpu_time:.1f}x")
    print("="*30)




#testowanie wpływu jednego konkretnego parametru
def execute_series(device, base_cfg, param_name, values):
    results = {}
    tr, te = get_cifar10_loaders(batch_size=base_cfg["batch_size"])
    
    for val in values:
        print(f"\n>>> TEST SERII: {param_name} = {val}")
        cfg = base_cfg.copy()
        cfg[param_name] = val
        
        # Jeśli val jest listą, zamień na string, żeby mogła być kluczem w słowniku
        key = str(val) if isinstance(val, list) else val
        
        # 'key' zamiast 'val' jako klucza słownika
        results[key] = run_experiment(device, cfg, tr, te)
        
    return results