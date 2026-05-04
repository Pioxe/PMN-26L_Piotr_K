
from analis import print_dataset_stats, check_class_distribution, print_summary_table, set_seed, run_eda
from graph import plot_confusion_matrix_heatmap, plot_series_results, show_misclassified_images, visualize_random_samples
from class_cifar10 import get_cifar10_loaders, get_device
from engine import evaluate_best_model, run_experiment, compare_hardware_speed, execute_series








def main():
    set_seed1=47 #taki sam seed by wyniki były powtarzalne
    set_seed(seed=set_seed1) 
    device = get_device()
    run_eda(set_seed1)  # EDA z ustalonym seedem dla powtarzalności
    
    #Konfiguracja bazowa do testów (krótkie serie, by nie czekać wiecznie)
    base_cfg = {
        "batch_size": 128, "lr": 0.001, "epochs": 5,
        "hidden_layers": [256, 128], "dropout": 0.0, "optimizer_type": "adam"
    }

    # --- TEST PRĘDKOŚCI ---
    compare_hardware_speed(device, base_cfg)

    # --- SERIE EKSPERYMENTÓW (Dokumentacja punkt 4) ---
    res_lr = execute_series(device, base_cfg, "lr", [0.0001, 0.001, 0.01])
    print_summary_table(res_lr, "Learning Rate")
    plot_series_results(res_lr, "Learning Rate")

    res_bs = execute_series(device, base_cfg, "batch_size", [32, 64, 128])
    print_summary_table(res_bs, "Batch Size")
    plot_series_results(res_bs, "Batch Size")

    res_dr = execute_series(device, base_cfg, "dropout", [0.0, 0.2, 0.5])
    print_summary_table(res_dr, "Dropout")
    plot_series_results(res_dr, "Dropout")

    res_opt = execute_series(device, base_cfg, "optimizer_type", ["adam", "sgd"])
    print_summary_table(res_opt, "Optimizer")
    plot_series_results(res_opt, "Optimizer")
#  Wpływ liczby neuronów (Architektura)
    print("\nEksperyment: Architektura sieci (liczba neuronów)")
    res_layers = execute_series(device, base_cfg, "hidden_layers", [
        [64, 64],                   # Bardzo mała sieć
        [256, 128],                 # Średnia sieć bazowa
        [512, 256, 128],            # Duża sieć
        [1024, 512, 256]            # Bardzo duża sieć
    ])
    print_summary_table(res_layers, "Hidden Layers")
    plot_series_results(res_layers, "Hidden Layers")
    print("\n" + "#"*60)
    print("TRENOWANIE NAJLEPSZEGO MODELU (Parametry wybrane z testów)")
    print("#"*60)
    best_cfg = {
        "batch_size": 64, "lr": 0.001, "epochs": 10,
        "hidden_layers": [512, 256, 128], "dropout": 0.2, "optimizer_type": "adam"
    }

    tr, te = get_cifar10_loaders(batch_size=best_cfg["batch_size"])
    history, best_model = run_experiment(device, best_cfg, tr, te)
    
    # Klasy i metryki końcowe
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    y_true, y_pred, misclassified = evaluate_best_model(best_model, te, device, classes)
    
    plot_confusion_matrix_heatmap(y_true, y_pred, classes)
    show_misclassified_images(misclassified, classes)

if __name__ == "__main__":
    main()