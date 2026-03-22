from ucimlrepo import fetch_ucirepo 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#wczytanie danych
def wczytaj_iris():
    # x-cechy y-nazwy
    iris = fetch_ucirepo(id=53) 
    X = iris.data.features
    y = iris.data.targets 
    return X, y
#marktery by zawsze miały ten sam znaczek na wykresie
MOJE_MARKERY = {
    'Iris-setosa': 'o',      # kółko
    'Iris-versicolor': 's',  # kwadrat 
    'Iris-virginica': 'X'    # duży krzyżyk
}

#marktery by zawsze miały ten sam kolor na wykresie
MOJE_KOLORY = {
    'Iris-setosa': '#3498db',      #  niebieski
    'Iris-versicolor': '#f1c40f',  #  żółty
    'Iris-virginica': '#9b59b6'    #  fioletowy
}

def wykres1(X, y, tytul, x_label, ylabel, legend):
    X_test_plot = X.copy()
    X_test_plot['gatunek'] = y 

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=X_test_plot, 
        x='petal length', 
        y='petal width', 
        hue='gatunek', 
        style='gatunek',  
        markers=MOJE_MARKERY, # <--- TO GWARANTUJE STAŁE KSZTAŁTY
        s=60,            
        palette=MOJE_KOLORY
    )

    plt.title(tytul, fontsize=15)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title=legend, loc='upper left')
    plt.show()


#cześć własciwa kodu
X, y = wczytaj_iris()
wykres1(X, y.values.ravel(), 'Wszystkie dane z bazy', 'Długość płatka (cm)', 'Szerokość płatka (cm)', 'Prawdziwy gatunek')
# 0,2 - 20 % idzie do testu 80 do uczenia
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
#K-Najbliższych Sąsiadów - jesli sasiedzi sa gatunkiem x to ty tez jestes gatunkiem x
knn = KNeighborsClassifier(n_neighbors=3)
#zapamietanie  położenia  pkt z treningu, .values.ravel() - sformatowanie do prostej tabeli i spłaszczenie zwykłego ciągu nazw
knn.fit(X_train, y_train.values.ravel())
#jak przewidzi dane testowe
y_pred = knn.predict(X_test)

# WYKRES
wykres1(X_train, y_train.values.ravel(), 'Dane TRENINGOWE (120 kwiatów)', 'Długość płatka (cm)', 'Szerokość płatka (cm)', 'Prawdziwy gatunek')
wykres1(X_test, y_pred, 'Wyniki klasyfikacji KNN na danych testowych (30 kwiatów)','Długość płatka (cm)','Szerokość płatka (cm)','Zgadnięty gatunek')
print(f"Skuteczność algorytmu KNN: {accuracy_score(y_test, y_pred) * 100:.2f}%")

#macro->Liczy średnią "gatunek po gatunku".
print(f"Precyzja (Precision):   {precision_score(y_test, y_pred, average='macro') * 100:.2f}%")
print(f"Czułość (Recall):       {recall_score(y_test, y_pred, average='macro') * 100:.2f}%")
print(f"F1-Score:               {f1_score(y_test, y_pred, average='macro') * 100:.2f}%")
print("\nPełny raport klasyfikacji")
print(classification_report(y_test, y_pred))


# WYKRES Wizualizacja trafień z prostą legendą 
#Przygotowanie danych
wyniki = X_test.copy()
wyniki['Prawda'] = y_test.values.ravel()
wyniki['Predykcja'] = y_pred

#kolumna ze statusem Poprawne/BŁĄD
wyniki['Status'] = np.where(wyniki['Prawda'] == wyniki['Predykcja'], 'Poprawne', 'BŁĄD')

#jedna połączona kolumna dla legendy, typu 'Poprawne - Iris-setosa', 'BŁĄD - Iris-versicolor' 
wyniki['Legenda'] = wyniki['Status'] + " - " + wyniki['Prawda']


Combined_palette = {}
Combined_markers = {}
#dla wszystkich 6 możliwych kombinacji (3 gatunki * 2 statusy)
for species, marker in MOJE_MARKERY.items():
    # Etykieta Poprawne -> kolor zielony
    label_ok = f"Poprawne - {species}"
    Combined_palette[label_ok] = '#2ecc71' # zielony
    Combined_markers[label_ok] = marker    # Prawdziwy kształt gatunku
    
    # Etykieta BŁĄD -> kolor czerwony
    label_error = f"BŁĄD - {species}"
    Combined_palette[label_error] = '#e74c3c' # czerwony
    Combined_markers[label_error] = marker    # Prawdziwy kształt gatunku

#wykres
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")


ax = sns.scatterplot(
    data=wyniki, 
    x='petal length', 
    y='petal width', 
    hue='Legenda',       
    style='Legenda',     
    markers=Combined_markers, 
    palette=Combined_palette, 
    s=70, 
    alpha=0.8,
    edgecolor='black'
)

#etykiety tekstowe TYLKO dla błędów
for i in range(len(wyniki)):
    if wyniki['Status'].iloc[i] == 'BŁĄD':
        plt.text(
            wyniki['petal length'].iloc[i] + 0.1, # lekkie przesunięcie tekstu w prawo
            wyniki['petal width'].iloc[i], 
            f"Zgadł: {wyniki['Predykcja'].iloc[i]}\nByło: {wyniki['Prawda'].iloc[i]}",
            fontsize=9, color='red', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='red', boxstyle='round')
        )

# Ustawienia estetyczne
plt.title("Analiza skuteczności KNN", fontsize=16)
plt.xlabel("Długość płatka (cm)", fontsize=12)
plt.ylabel("Szerokość płatka (cm)", fontsize=12)    
plt.legend(title="Legenda", loc='upper left')
plt.tight_layout()
plt.show()



# WYKRES TSNE
#Przygotowanie danych
X_all = X.copy()
y_pred_train = knn.predict(X_train)

y_pred_combined = pd.Series(index=X_all.index, dtype='object')
y_pred_combined.loc[X_train.index] = y_pred_train
y_pred_combined.loc[X_test.index] = y_pred

#Uruchomienie t-SNE
tsne = TSNE(n_components=2, random_state=67, perplexity=30)
X_tsne = tsne.fit_transform(X_all)

#Wykres
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=X_tsne[:, 0], 
    y=X_tsne[:, 1], 
    hue=y_pred_combined,
    style=y_pred_combined,
    markers=MOJE_MARKERY,
    palette=MOJE_KOLORY,
    s=60,
    alpha=0.8,
    edgecolor='black'
)

plt.title("Wizualizacja t-SNE", fontsize=16)
plt.xlabel("Wymiar t-SNE 1")
plt.ylabel("Wymiar t-SNE 2")
plt.legend(title="Gatunek")
plt.tight_layout()
plt.show()