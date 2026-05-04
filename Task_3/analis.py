from collections import Counter
import torch, torchvision, numpy as np, random
from graph import visualize_random_samples

#inf techniczne o zbiorze danych, 
# łączna liczba obrazów w zestawie treningowym i testowym, 
# wymiary pojedynczego zdjęcia 
#  zakres wartości pikseli (zazwyczaj 0–255 przed normalizacją).
def print_dataset_stats(train_dataset, test_dataset):
    train_data = train_dataset.data
    
    print("--- Dataset Statistics ---")
    print(f"Total training images: {len(train_dataset)}")
    print(f"Total test images: {len(test_dataset)}")
    
    # Image dimensions
    sample_image = train_data[0]
    print(f"Single image dimensions: {sample_image.shape} (H x W x C)")
    
    # Pixel value range
    print(f"Pixel value range: min = {train_data.min()}, max = {train_data.max()}")
#Zlicza i wyświetla liczbę przykładów dla każdej z 10 klas w całym zbiorze (łącznie dla treningu i testu).
#czy zbiór jest zbalansowany (czy każda klasa ma tyle samo zdjęć).
def check_class_distribution(train_dataset, test_dataset):
    classes = train_dataset.classes
    all_targets = train_dataset.targets + test_dataset.targets
    class_counts = Counter(all_targets)
    
    print("\n--- Class Distribution ---")
    print("Total images per class (Train + Test):")
    for class_idx, count in sorted(class_counts.items()):
        print(f" - Class '{classes[class_idx]}': {count} images")
#Konfiguruje ziarno losowości (seed) dla wszystkich używanych bibliotek (Python, NumPy, PyTorch)
def set_seed(seed=47):
    #Ustawia ziarno losowości dla wszystkich bibliotek.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Ustawiono ziarno losowości: {seed}")
#Odpowiada za bezpośrednie pobranie surowego zbioru CIFAR-10 z serwerów
def dow():
    ts = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    vs = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    return ts, vs

#proces wstępnej analizy danych. 
# Pobiera zbiór 
# uruchamia statystyki,
# sprawdza rozkład klas 
#  wywołuje wizualizację losowych próbek obrazów.
def run_eda(set_seed1=47):
    ts,vs = dow()
    print_dataset_stats(ts, vs)
    check_class_distribution(ts, vs)
    visualize_random_samples(ts, seed=set_seed1)
    
#podsumowuje wyniki serii testów w tabeli 
def print_summary_table(results_dict, param_name):
    """Wypisuje elegancką tabelę z wynikami danej serii."""
    print(f"\nTABELA WYNIKÓW: Wpływ parametru '{param_name}'")
    print(f"{param_name:<15} | Końcowe Accuracy | Końcowy Loss")
    print("-" * 50)
    for val, (history, _) in results_dict.items():
        final_acc = history['val_acc'][-1]
        final_loss = history['val_loss'][-1]
        print(f"{str(val):<15} | {final_acc:>15.2f}% | {final_loss:>12.4f}")
    print("-" * 50)