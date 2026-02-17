import numpy as np
from nbc import NBCDiscrete
from sklearn.model_selection import train_test_split

def wine_data(path):
    data = np.loadtxt(path, delimiter=",")
    x = data[:, 1:]
    y = data[:, 0].astype(int)

    return x, y

import numpy as np

def car_data(path):
    data = np.genfromtxt(path, delimiter=",", dtype='str')
    X_str = data[:, :-1]
    y_str = data[:, -1]
    
    X = np.zeros(X_str.shape, dtype=int)
    domain_sizes = []
    for j in range(X_str.shape[1]):
        unique_values = np.unique(X_str[:, j])
        mapping = {val: i for i, val in enumerate(unique_values)}
        for i in range(X_str.shape[0]):
            X[i, j] = mapping[X_str[i, j]]
            
        domain_sizes.append(len(unique_values))
        
    unique_classes = np.unique(y_str)
    y_mapping = {val: i for i, val in enumerate(unique_classes)}
    y = np.array([y_mapping[val] for val in y_str], dtype=int)
    
    return X, y, np.array(domain_sizes)

def discretize(X, bins=5, mins_ref=None, maxes_ref=None):
    if mins_ref is None:
        mins_ref = np.min(X, axis=0)
    if maxes_ref is None:
        maxes_ref = np.max(X, axis=0)

    X_discrete = np.floor((X - mins_ref) / (maxes_ref - mins_ref) * bins)
    X_discrete = np.clip(X_discrete, 0, bins - 1).astype(int)

    return X_discrete, mins_ref, maxes_ref



if __name__ == "__main__":
    DATASET_NAME = "MUSHROOMS"
    bins = 5
    

    if DATASET_NAME == "WINE":
        file_path = "C:/Users/vv55624/Desktop/SI/wine/wine.data" 
        
        try:
            X, y = wine_data(file_path)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
            X_train_disc, min_vals, max_vals = discretize(X_train, bins=bins)
            X_test_disc, _, _ = discretize(
                X_test, bins=bins, mins_ref=min_vals, maxes_ref=max_vals
            )
            n_features = X.shape[1]
            domain_sizes = np.array([bins] * n_features)

        except OSError:
            print(f"Nie znaleziono pliku z danymi '{file_path}'. Upewnij się, że ścieżka jest poprawna.")
            exit()

    elif DATASET_NAME == "MUSHROOMS":
        file_path = "C:/Users/vv55624/Desktop/SI/mushroom/agaricus-lepiota.data"
        
        try:
            X, y, domain_sizes = car_data(file_path)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_train_disc = X_train
            X_test_disc = X_test
            n_features = X.shape[1]

        except OSError:
            print(f"Nie znaleziono pliku z danymi '{file_path}'. Upewnij się, że ścieżka jest poprawna.")
            exit()
            
    else:
        print("Niepoprawna nazwa zbioru danych. Ustaw DATASET_NAME na 'WINE' lub 'MUSHROOMS'.")
        exit()
    
    print(f"--- Klasyfikacja NBC dla {DATASET_NAME} ---")
    print(f"Rozmiar zbioru uczącego: {X_train_disc.shape}")
    print(f"Rozmiar zbioru testowego: {X_test_disc.shape}")
    print(f"Rozmiary domen: {domain_sizes}")

    # 2. Uruchomienie z LAPLACE'A
    nbc = NBCDiscrete(domain_sizes=domain_sizes, laplace=True)
    
    nbc.fit(X_train_disc, y_train)
    
    # 3. Pomiary dokładności
    train_acc = np.mean(nbc.predict(X_train_disc) == y_train)
    test_acc = np.mean(nbc.predict(X_test_disc) == y_test)

    print(f"\nDokładność (Train): {train_acc:.2%}")
    print(f"Dokładność (Test):  {test_acc:.2%}")



