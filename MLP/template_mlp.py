import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import time
import copy

class MLPApproximator(BaseEstimator, RegressorMixin):

    ALGO_NAMES = ["sgd_simple", "sgd_momentum", "rmsprop", "adam"]

    def __init__(self, structure=[16, 8, 4], activation_name="relu", targets_activation_name="linear", initialization_name="uniform", 
                 algo_name="sgd_momentum", learning_rate=1e-2,  n_epochs=100, batch_size=10, seed=0,
                 verbosity_e=100, verbosity_b=10):
        self.structure = structure
        self.activation_name = activation_name
        self.targets_activation_name = targets_activation_name
        self.initialization_name = initialization_name
        self.algo_name = algo_name
        if self.algo_name not in self.ALGO_NAMES:
            self.algo_name = self.ALGO_NAMES[0]                            
        self.loss_name = "squared_loss"
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed        
        self.verbosity_e = verbosity_e 
        self.verbosity_b = verbosity_b
        self.history_weights = {}
        self.history_weights0 = {}
        self.n_params = None
        # params / constants for algorithms
        self.momentum_beta = 0.9
        self.rmsprop_beta = 0.9
        self.rmsprop_epsilon = 1e-7
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-7

                
    def __str__(self):
        txt = f"{self.__class__.__name__}(structure={self.structure},"
        txt += "\n" if len(self.structure) > 32 else " "              
        txt += f"activation_name={self.activation_name}, targets_activation_name={self.targets_activation_name}, initialization_name={self.initialization_name}, "
        txt += f"algo_name={self.algo_name}, learning_rate={self.learning_rate}, n_epochs={self.n_epochs}, batch_size={self.batch_size})"
        if self.n_params:
            txt += f" [n_params: {self.n_params}]"     
        return txt
    
    @staticmethod
    def he_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / n_in)
        return ((np.random.rand(n_out, n_in)  * 2.0 - 1.0) * scaler).astype(np.float32)
    
    @staticmethod
    def he_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / n_in)
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def glorot_uniform(n_in, n_out):
        scaler = np.sqrt(6.0 / (n_in + n_out))
        return ((np.random.rand(n_out, n_in)  * 2.0 - 1.0) * scaler).astype(np.float32)
    
    @staticmethod
    def glorot_normal(n_in, n_out):
        scaler = np.sqrt(2.0 / (n_in + n_out))
        return (np.random.randn(n_out, n_in) * scaler).astype(np.float32)

    @staticmethod
    def prepare_batch_ranges(m, batch_size):
        n_batches = int(np.ceil(m / batch_size))
        batch_ranges = batch_size * np.ones(n_batches, dtype=np.int32)
        remainder = m % batch_size
        if remainder > 0:        
            batch_ranges[-1] = remainder
        batch_ranges = np.r_[0, np.cumsum(batch_ranges)]                
        return n_batches, batch_ranges    

    @staticmethod
    def sigmoid(S):         
        # f(s) = 1 / (1 + exp(-s))
        return 1.0 / (1.0 + np.exp(-S))
    
    @staticmethod
    def sigmoid_d(phi_S):
        # f'(s) = f(s) * (1 - f(s))
        return phi_S * (1.0 - phi_S)
        
    @staticmethod
    def relu(S):
        #Zwraca 0 dla wartości ujemnych i s dla dodatnich.
        return np.maximum(0, S)

    @staticmethod
    def relu_d(phi_S):
        # f'(s) = 1 dla s > 0, w przeciwnym razie 0
        return (phi_S > 0).astype(np.float32)  

    @staticmethod
    def linear(S):
        # f(s) = s
        return S

    @staticmethod
    def linear_d(phi_S):
        # f'(s) = 1
        return np.ones_like(phi_S)
    
    @staticmethod
    def squared_loss(y_MLP, y_target):
        #f(s)= 1/2 * (y - y*)^2
        return 0.5 * (y_MLP - y_target)**2
        
    @staticmethod
    def squared_loss_d(y_MLP, y_target):
        #f'(s)= d(1/2 e^2) / dy = (y - y*)
        return (y_MLP - y_target)
    
    def pre_algo_sgd_simple(self):
        return # no special preparation needed for simple SGD
    
    def algo_sgd_simple(self, l):
        #Aktualizacja macierzy wag W_{t+1} = W_t - \eta \cdot abla E
        self.weights_[l] -= self.learning_rate * self.gradients[l]
        #Aktualizacja wag
        self.weights0_[l] -= self.learning_rate * self.gradients0[l]

    def pre_algo_sgd_momentum(self):
        self.v_w = [np.zeros_like(w) if w is not None else None for w in self.weights_]
        self.v_w0 = [np.zeros_like(w0) if w0 is not None else None for w0 in self.weights0_]
    
    def algo_sgd_momentum(self, l):
        #v_t = beta * v_{t-1} + (1 - beta) * grad_t
        self.v_w[l] = self.momentum_beta * self.v_w[l] + (1 - self.momentum_beta) * self.gradients[l]
        self.v_w0[l] = self.momentum_beta * self.v_w0[l] + (1 - self.momentum_beta) * self.gradients0[l]

        #w = w - eta * v_t
        self.weights_[l] -= self.learning_rate * self.v_w[l]
        self.weights0_[l] -= self.learning_rate * self.v_w0[l]

    def pre_algo_rmsprop(self):
        self.s_w = [np.zeros_like(w) if w is not None else None for w in self.weights_]
        self.s_w0 = [np.zeros_like(w0) if w0 is not None else None for w0 in self.weights0_]
    
    def algo_rmsprop(self, l):
        #v_t = beta * v_{t-1} + (1 - beta) * grad^2
        self.s_w[l] = self.rmsprop_beta * self.s_w[l] + (1 - self.rmsprop_beta) * (self.gradients[l]**2)
        self.s_w0[l] = self.rmsprop_beta * self.s_w0[l] + (1 - self.rmsprop_beta) * (self.gradients0[l]**2)

        #w = w - (eta / sqrt(v_t + eps)) * grad
        # eps zapobiega dzieleniu przez zero
        self.weights_[l] -= (self.learning_rate / (np.sqrt(self.s_w[l]) + self.rmsprop_epsilon)) * self.gradients[l]
        self.weights0_[l] -= (self.learning_rate / (np.sqrt(self.s_w0[l]) + self.rmsprop_epsilon)) * self.gradients0[l]                      
                
    def pre_algo_adam(self):
        # Bufory dla wag głównych (W)
        self.m_w = [np.zeros_like(w) if w is not None else None for w in self.weights_]
        self.v_w = [np.zeros_like(w) if w is not None else None for w in self.weights_]
        
        # Bufory dla wag wejść progowych (biasów - w0)
        self.m_w0 = [np.zeros_like(w0) if w0 is not None else None for w0 in self.weights0_]
        self.v_w0 = [np.zeros_like(w0) if w0 is not None else None for w0 in self.weights0_]
    
    def pre_algo_adam(self):
        pass # TODO (homework)
    
    def algo_adam(self, l):
        pass # TODO: self.weights_[l], self.weights0_[l] to be updated (l is a layer index)                        
            
    def fit(self, X, y):
        np.random.seed(self.seed)
        self.activation_ = getattr(MLPApproximator, self.activation_name)
        self.activation_d_ = getattr(MLPApproximator, self.activation_name + "_d")
        self.initialization_ = getattr(MLPApproximator, ("he_" if self.activation_name == "relu" else "glorot_") + self.initialization_name)
        self.targets_activation_ = getattr(MLPApproximator, self.targets_activation_name)
        self.targets_activation_d_ = getattr(MLPApproximator, self.targets_activation_name + "_d")        
        self.loss_ = getattr(MLPApproximator, self.loss_name)
        self.loss_d_ = getattr(MLPApproximator, self.loss_name + "_d")
        self.pre_algo_ = getattr(self, "pre_algo_" + self.algo_name)
        self.algo_ = getattr(self, "algo_" + self.algo_name)                
        self.weights_ = [None] # so that network inputs are considered layer 0, and actual layers of neurons are numbered 1, 2, ...  
        self.weights0_ = [None] # so that network inputs are considered layer 0, and actual layers of neurons are numbered 1, 2, ...
        m, n = X.shape
        if len(y.shape) == 1:
            y = np.array([y]).T
        self.n_ = n
        self.n_targets_ = 1 if len(y.shape) == 1 else y.shape[1]
        self.n_params = 0
        for l in range(len(self.structure) + 1):
            n_in = n if l == 0 else self.structure[l - 1]
            n_out = self.structure[l] if l < len(self.structure) else self.n_targets_ 
            w = self.initialization_(n_in, n_out)
            w0 = np.zeros((n_out, 1), dtype=np.float32)            
            self.weights_.append(w)
            self.weights0_.append(w0)
            self.n_params += w.size
            self.n_params += w0.size
        t1 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT [total of weights (params): {self.n_params}]")
        self.pre_algo_() # if some preparation needed         
        n_batches, batch_ranges = MLPApproximator.prepare_batch_ranges(m, self.batch_size)
        self.t = 0
        for e in range(self.n_epochs):
            t1_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                print("-" * 3)
                print(f"EPOCH {e + 1}/{self.n_epochs}:")
                self.forward(X)
                loss_e_before = np.mean(self.loss_(self.signals[-1], y))                
            p = np.random.permutation(m)          
            for b in range(n_batches):
                indexes = p[batch_ranges[b] : batch_ranges[b + 1]]
                X_b = X[indexes] 
                y_b = y[indexes]                
                self.forward(X_b)
                loss_b_before = np.mean(self.loss_(self.signals[-1], y_b))                
                self.backward(y_b)
                for l in range(1, len(self.structure) + 2):
                    self.algo_(l)                    
                if (e % self.verbosity_e == 0 or e == self.n_epochs - 1) and b % self.verbosity_b == 0:
                    self.forward(X_b)
                    loss_b_after = np.mean(self.loss_(self.signals[-1], y_b))                    
                    print(f"[epoch {e + 1}/{self.n_epochs}, batch {b + 1}/{n_batches} -> loss before: {loss_b_before}, loss after: {loss_b_after}]")                                                                        
                self.t += 1
            t2_e = time.time()
            if e % self.verbosity_e == 0 or e == self.n_epochs - 1:
                self.forward(X)
                loss_e_after = np.mean(self.loss_(self.signals[-1], y))
                self.history_weights[e] = copy.deepcopy(self.weights_)
                self.history_weights0[e] = copy.deepcopy(self.weights0_)
                print(f"ENDING EPOCH {e + 1}/{self.n_epochs} [loss before: {loss_e_before}, loss after: {loss_e_after}; epoch time: {t2_e - t1_e} s]")                  
        t2 = time.time()
        if self.verbosity_e > 0:
            print(f"FIT DONE. [time: {t2 - t1} s]")
                                                          
    def forward(self, X_b):
        self.signals = [None] * (len(self.structure) + 2)
        self.signals[0] = X_b
        for l in range(1, len(self.structure) + 2):
            #s = W * x + w0
            S = self.signals[l-1] @ self.weights_[l].T + self.weights0_[l].T
            
            #targets_activation dla ostatniej warstwy, activation dla reszty
            if l <= len(self.structure):
                #nakladamy funkcje aktywacji
                self.signals[l] = self.activation_(S)
            else:
                #y=s
                self.signals[l] = S
         
    def backward(self, y_b):        
        self.deltas = [None] * len(self.signals)        
        self.gradients = [None] * len(self.signals)
        self.gradients0 = [None] * len(self.signals)
        L = len(self.signals) - 1
        
        # delta = (y - y*) * f'(s)
        self.deltas[L] = self.loss_d_(self.signals[L], y_b)
        
        for l in range(L, 0, -1):
            m = y_b.shape[0]
            self.gradients[l] = (self.deltas[l].T @ self.signals[l-1]) / m
            self.gradients0[l] = np.mean(self.deltas[l], axis=0, keepdims=True).T
            
            #Propagacja bledu
            if l > 1:
                #delta_{l-1} = (delta_l @ W_l) * f'(s_{l-1})
                prop_error = self.deltas[l] @ self.weights_[l]
                self.deltas[l-1] = prop_error * self.activation_d_(self.signals[l-1])
                            
    def predict(self, X):
        self.forward(X)        
        y_pred = self.signals[-1] # TODO replace by: y_pred = self.signals[-1] (when self.forward(X) is ready)  
        if self.n_targets_ == 1:
            y_pred = y_pred[:, 0]
        return y_pred