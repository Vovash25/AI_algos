import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from template_mlp import MLPApproximator
from template_mlp_main import fake_data, loss_during_fit, r2_during_fit


SEED = 55624
DOMAIN = 1.5 * np.pi
N_EPOCHS = 1000
BATCH_SIZE = 256

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "eksperymenty_wyniki")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


lrs = [1e-2, 1e-3, 1e-4]
activations = ["sigmoid", "relu"]
algos = ["sgd_simple", "sgd_momentum", "rmsprop"]
structures = [
    [128, 64, 32],
    [128, 128, 64, 64, 32, 32],
    [64]*5 + [32]*5 + [16]*5 + [8]*5
]

#Dane
X_train, y_train = fake_data(1000, DOMAIN, 0.1)
X_test, y_test = fake_data(1000, DOMAIN, 0.1)

summary_log = []
print(f"Rozpoczynam eksperymenty (Seed: {SEED}, Domain: {DOMAIN:.2f})...")

for struct in structures:
    for act in activations:
        for algo in algos:
            for lr in lrs:
                exp_name = f"S{len(struct)}_{act}_{algo}_LR{lr}"
                print(f"Przetwarzanie: {exp_name}")
                #Inicjalizacja i trening
                approx = MLPApproximator(
                    structure=struct, activation_name=act, algo_name=algo,
                    learning_rate=lr, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                    seed=SEED, verbosity_e=100
                )
                
                try:
                    approx.fit(X_train, y_train)
                    #Serializacja obiektu (Pickle)
                    with open(os.path.join(LOG_DIR, f"{exp_name}.pkl"), "wb") as f:
                        pickle.dump(approx, f)
                    
                    # Obliczenie wynikow
                    mse_test = np.mean((approx.predict(X_test) - y_test)**2)
                    r2_test = approx.score(X_test, y_test)
                    summary_log.append(f"{exp_name}; MSE: {mse_test:.6f}; R2: {r2_test:.4f}")
                
                except Exception as e:
                    print(f"BŁĄD w {exp_name}: {e}")
                    summary_log.append(f"{exp_name}; FAILED; {e}")

with open(os.path.join(LOG_DIR, "summary_log.txt"), "w") as f:
    f.write("\n".join(summary_log))

print("\nGenerowanie wykresu...")

target_struct = [128, 128, 64, 64, 32, 32]
target_lr = 1e-3
target_act = "relu"

fig, axes = plt.subplots(1, 3, figsize=(15, 10))
fig.suptitle(f"Porównanie algorytmów (Struct={target_struct}, LR={target_lr}, Act={target_act})", fontsize=14)
axes = axes.flatten()

for i, algo in enumerate(algos):
    filename = f"S{len(target_struct)}_{target_act}_{algo}_LR{target_lr}.pkl"
    path = os.path.join(LOG_DIR, filename)
    
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        epochs, l_train, l_test = loss_during_fit(model, X_train, y_train, X_test, y_test)
        
        axes[i].plot(epochs, l_train, color="blue", label="Train Loss")
        axes[i].plot(epochs, l_test, color="red", linestyle="--", label="Test Loss")
        axes[i].set_title(f"Algorytm: {algo}")
        axes[i].set_xlabel("Epoka")
        axes[i].set_ylabel("MSE")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(LOG_DIR, "porownanie.png"))
plt.show()
print("Ready")