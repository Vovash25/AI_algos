import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

class NieLiniowyPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, m=20, sigma=1.0, eta=1.0, k_max=1000):
        self.m = m
        self.sigma = sigma
        self.eta = eta
        self.k_max = k_max
        
        self.centers_ = None
        self.weights_ = None
        self.classes_ = None

    def _rbf_transform(self, X, centers):
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Gaussa: exp(-dist^2 / 2*sigma^2)
        return np.exp(-dist_sq / (2 * self.sigma ** 2))

    def fit(self, X, y):
        #Uczenie modelu
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        self.centers_ = np.random.uniform(min_vals, max_vals, (self.m, n_features))
        Z_rbf = self._rbf_transform(X, self.centers_)
        Z = np.hstack([np.ones((n_samples, 1)), Z_rbf])
        self.weights_ = np.zeros(self.m + 1)
        for k in range(self.k_max):
            activations = Z.dot(self.weights_)
            predictions = np.where(activations >= 0, 1, -1)
            error_indices = np.where(y != predictions)[0]
            if len(error_indices) == 0:
                break
            idx = np.random.choice(error_indices)           
            # Reguła Rosenblatta: w = w + eta * (y - y_pred) * x
            self.weights_ += self.eta * (y[idx] - predictions[idx]) * Z[idx]
            
        return self

    def predict(self, X):
        X = np.array(X)
        Z_rbf = self._rbf_transform(X, self.centers_)
        Z = np.hstack([np.ones((X.shape[0], 1)), Z_rbf])
        activations = Z.dot(self.weights_)
        return np.where(activations >= 0, 1, -1)
    
class PerceptronProsty(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=1.0, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w_ = None
        self.steps_ = 0

    def fit(self, X, y):
        rgen = np.random.RandomState(1)
        n_samples, n_features = X.shape
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + n_features)
        self.steps_ = 0
        
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                prediction = self.predict(xi.reshape(1, -1))[0]
                update = self.learning_rate * (target - prediction)
                if update != 0.0:
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    self.steps_ += 1        
        return self
    
    def decision_function(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

def data(n, gamma):
    X = np.random.rand(n, 2) * 2 - 1
    y = np.where(X[:, 0] < 0, -1, 1)
    X[y == -1, 0] -= 0.5 * gamma
    X[y == 1, 0] += 0.5 * gamma
    alpha = np.random.random() * 2 * np.pi
    c, s = np.cos(alpha), np.sin(alpha)
    rot = np.array([[c, -s], [s, c]])
    tr = np.random.random(2) * 5 - 5
    return X @ rot + tr, y

def ndata(n_samples=1000):
    x1 = np.random.uniform(0, 2 * np.pi, n_samples)
    x2 = np.random.uniform(-1, 1, n_samples)
    X = np.column_stack((x1, x2))
    y = np.ones(n_samples)
    warunek = np.abs(np.sin(x1)) > np.abs(x2)
    y[warunek] = -1
    
    return X, y


def Perceptron():
    X, y = data(200, 0.2)
    ppn = PerceptronProsty(learning_rate=0.1, n_iter=1000)
    ppn.fit(X, y)
    
    # 3. Wykres
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', marker='o', label='Klasa -1')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='x', label='Klasa 1')

    # Rysowanie granicy tylko jeśli wagi zostały wyznaczone
    if ppn.w_ is not None and ppn.w_[2] != 0:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        xx = np.array([x_min, x_max])
        # x2 = -(w1*x1 + w0) / w2
        yy = -(ppn.w_[1] * xx + ppn.w_[0]) / ppn.w_[2]
        plt.plot(xx, yy, 'k-', linewidth=2, label='Granica decyzyjna')

    plt.title(f'Perceptron - Wynik (Kroki: {ppn.steps_})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

def PerceptronNieLiniowy():
    X_raw, y = ndata(4000)
    X_norm = np.copy(X_raw)
    X_norm[:, 0] = (X_raw[:, 0] / np.pi) - 1 

    clf = NieLiniowyPerceptron(m=50, sigma=0.3, eta=1.0, k_max=2000)
    clf.fit(X_norm, y)

    y_pred = clf.predict(X_norm)
    accuracy = np.mean(y == y_pred) * 100
    
    print(f"--------------------------------------------------")
    print(f"Liczba punktów: {len(y)}")
    print(f"Poprawnie sklasyfikowane: {np.sum(y == y_pred)}")
    print(f"Dokładność modelu: {accuracy:.2f}%")
    print(f"--------------------------------------------------")

    resolution = 100
    xx_norm, yy_norm = np.meshgrid(np.linspace(-1, 1, resolution),
                                   np.linspace(-1, 1, resolution))
    grid_points_norm = np.c_[xx_norm.ravel(), yy_norm.ravel()]
    xx_raw = (xx_norm + 1) * np.pi
    yy_raw = yy_norm


    Z_rbf = clf._rbf_transform(grid_points_norm, clf.centers_)
    Z_bias = np.hstack([np.ones((grid_points_norm.shape[0], 1)), Z_rbf])
    Z_vals = Z_bias.dot(clf.weights_)
    Z_plot_weighted = Z_vals.reshape(xx_norm.shape)
    Z_class = np.where(Z_plot_weighted >= 0, 1, -1)

    # WYKRES 1
    plt.figure(figsize=(10, 6))

    plt.contourf(xx_raw, yy_raw, Z_class, levels=[-1.1, 0, 1.1], 
                 colors=['#bfd3ff', '#fddbc7'], alpha=1.0) 
    plt.contour(xx_raw, yy_raw, Z_class, levels=[0], colors='black', linewidths=2.5)
    plt.scatter(X_raw[y == -1, 0][::5], X_raw[y == -1, 1][::5], 
                c='blue', marker='o', s=20, edgecolors='b', label='Klasa -1')
    plt.scatter(X_raw[y == 1, 0][::5], X_raw[y == 1, 1][::5], 
                c='lime', marker='x', s=20, linewidths=1.5, label='Klasa 1')
    
    plt.title(f'Perceptron Nieliniowy (Dokładność: {accuracy:.2f}%)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-1, 1)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.show()

    # WYKRES 2
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx_norm, yy_norm, Z_plot_weighted, cmap="jet", linewidth=0, antialiased=False)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Weighted Sum')
    ax.set_title('Powierzchnia sumy ważonej')
    ax.view_init(elev=45, azim=-45)
    plt.show()

    # WYKRES 3
    plt.figure(figsize=(10, 8))
    plt.contourf(xx_norm, yy_norm, Z_plot_weighted, levels=30, cmap='YlOrRd', alpha=0.9)
    cntr_lines = plt.contour(xx_norm, yy_norm, Z_plot_weighted, levels=15, colors='black', linewidths=1.0)
    plt.clabel(cntr_lines, inline=True, fontsize=8, fmt='%1.0f')
    plt.scatter(X_norm[y == 1, 0], X_norm[y == 1, 1], c='lime', s=10, alpha=0.6, label='Klasa 1')
    plt.scatter(X_norm[y == -1, 0], X_norm[y == -1, 1], c='blue', s=10, alpha=0.6, label='Klasa -1')
    if clf.centers_ is not None:
        plt.scatter(clf.centers_[:, 0], clf.centers_[:, 1], c='black', s=50, label='Centra')
    plt.title('Wykres warstwicowy sumy ważonej')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

if __name__ == "__main__":
    # Perceptron() 
    PerceptronNieLiniowy()