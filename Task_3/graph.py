from matplotlib.pylab import seed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def imshow(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
#losowy zestaw obrazów z podanego zbioru danych i wyświetla je w formie siatki
def visualize_random_samples(dataset, num_samples=10, seed=47):
    classes = dataset.classes
    fig = plt.figure(figsize=(12, 5))
    
    np.random.seed(seed)
    random_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    print(f"\n--- Visualizing {num_samples} Sample Images ---")
    for i, idx in enumerate(random_indices):
        img, label_idx = dataset[idx]
        fig.add_subplot(2, 5, i + 1)
        imshow(img, classes[label_idx])
        
    plt.tight_layout()
    plt.show()
    
#wykresy porównawcze: 
#  dla dokładności walidacyjnej (Accuracy), 
#  dla funkcji straty (Loss)
def plot_series_results(results_dict, param_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Zestawienie wyników: Wpływ parametru {param_name}")

    for val, (history, _) in results_dict.items():
        ax1.plot(history['val_acc'], marker='o', label=f'{param_name}={val}')
        ax2.plot(history['val_loss'], marker='o', label=f'{param_name}={val}')

    ax1.set_title("Dokładność walidacyjna (Accuracy)")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title("Strata walidacyjna (Loss)")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

#heatmap macierzy pomyłek
def plot_confusion_matrix_heatmap(y_true, y_pred, classes):
    """Rysuje macierz pomyłek."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Macierz Pomyłek (Confusion Matrix)")
    plt.xlabel("Przewidywana klasa (Predykcja)")
    plt.ylabel("Rzeczywista klasa (Prawda)")
    plt.tight_layout()
    plt.show()

#galeria zdjęć, na których model popełnił błąd
def show_misclassified_images(misclassified, classes):
    """Wyświetla przykłady błędnie sklasyfikowanych obrazów."""
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle("Analiza Błędów: Przykłady fałszywych predykcji sieci", fontsize=16)
    
    for i in range(len(misclassified)):
        img, true_label, pred_label = misclassified[i]
        
        # Odwracanie normalizacji do wyświetlenia obrazka
        img = img.numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5 
        img = np.clip(img, 0, 1)
        
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(img)
        ax.set_title(f"Jest: {classes[true_label]}\nSieć myśli: {classes[pred_label]}", color="red", fontsize=11)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()