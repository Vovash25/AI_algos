import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NBCDiscrete(BaseEstimator, ClassifierMixin):
    def __init__(self, domain_sizes, laplace=False, logs=False):
        self.domain_sizes = domain_sizes
        self.laplace = laplace
        self.logs = logs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        m, n = X.shape
        K = self.classes_.size

        # Struktury danych
        self.priors_ = np.zeros(K)
        max_domain = np.max(self.domain_sizes)
        self.cond_probs_ = np.zeros((K, n, max_domain))

        # 2. Pętla ucząca
        for k_idx, label in enumerate(self.classes_):
            X_subset = X[y == label]
            count_in_class = X_subset.shape[0]

            # P(Y)
            self.priors_[k_idx] = count_in_class / m

            # P(X|Y)
            for j in range(n):
                counts = np.bincount(X_subset[:, j].astype(int), minlength=self.domain_sizes[j])
                counts = counts[:self.domain_sizes[j]]

                if self.laplace:
                    numerator = counts + 1
                    denominator = count_in_class + self.domain_sizes[j]
                else:
                    numerator = counts
                    denominator = count_in_class if count_in_class > 0 else 1
                
                self.cond_probs_[k_idx, j, :len(numerator)] = numerator / denominator
        return self

    def predict_proba(self, X):
        m, n = X.shape
        K = len(self.classes_)
        scores = np.zeros((m, K))

        for i in range(m):
            sample = X[i].astype(int)
            for k_idx in range(K):
                # Start: P(Y)
                val = self.priors_[k_idx]
                for j in range(n):
                    # Iloczyn: * P(X_j | Y)
                    feat_val = sample[j]
                    val *= self.cond_probs_[k_idx, j, feat_val]
                scores[i, k_idx] = val

        row_sums = np.sum(scores, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return scores / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
