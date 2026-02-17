import numpy as np

class GeneticAlgorithmDKP:
    def __init__(self, n, v, c, C, pop_size=1000, T=100, p_m=0.001, p_c = 0.9):
        self.n = n #dlugosc chromosomy(liczba genow)
        self.v = v #wartosci
        self.c = c #wagi
        self.C = C #pojemnosc plecaka
        self.pop_size = pop_size #rozmiar populacji
  
        self.T = T #liczba iteracij
        self.p_m = p_m
        self.p_c = p_c
        
        self.population = None #zestaw chromosom(locus)
        self.fitness_scores = None
        self.best_individual = None #najlepszy osobnik
        self.best_fitness = -1 

    def initialize_population(self):
        self.population = np.random.randint(2, size=(self.pop_size, self.n))
        self.fitness_scores = np.zeros(self.pop_size)#ocena ilosciowa

    def fitness_function(self, chromosome): #genotyp => fenotyp
        total_weight = np.sum(chromosome * self.c)
        if total_weight <= self.C:
            return np.sum(chromosome * self.v)
        else:
            return 0
              
    def evaluate_population(self):
        for i in range(self.pop_size):
            score = self.fitness_function(self.population[i])
            self.fitness_scores[i] = score
            
            if score > self.best_fitness:
                self.best_fitness = score
                self.best_individual = self.population[i].copy()

    def mutation(self):
        mask = np.random.rand(self.pop_size, self.n) < self.p_m
        self.population[mask] = 1 - self.population[mask]

        
    #selekcja jako funkcja 
    #def selection(self):
        #7.1 strona 172
    def roulette_selection(self):
        total_fitness = np.sum(self.fitness_scores)
        probs = self.fitness_scores / total_fitness
        return  np.random.choice(self.pop_size, size = self.pop_size, p = probs)

    def crossover(self):
        for i in range(0, self.pop_size - 1, 2):
            if np.random.rand() < self.p_c:
                point = np.random.randint(1, self.n)
                temp = self.population[i, :point].copy()
                self.population[i, :point] = self.population[i + 1, :point]
                self.population[i + 1, :point] = temp

    def run(self):
        self.initialize_population()
        
        for t in range(self.T):
            self.evaluate_population()
            self.roulette_selection()
            self.crossover()
            self.mutation()
        
        #zwroc najlepszego osobnika oraz wartosc
        return self.best_individual, self.best_fitness

def generate_dkp(n, scale):
    # Generowanie wartości i wag (n przedmiotów, 2 cechy)
    items = np.ceil(scale * np.random.rand(n, 2)).astype("int32")
    
    # Pojemnosc plecaka
    C = int(np.ceil(0.5 * 0.5 * n * scale))
    
    v = items[:, 0] #Wartosci
    c = items[:, 1] #Wagi
    
    return v, c, C


def solve_dkp(v, c, C):
    n = len(v)
    # Inicjalizacja tablicy DP: (n+1) wierszy, (C+1) kolumn
    dp = np.zeros((n + 1, C + 1), dtype=int)

    # Indukcja
    for i in range(1, n + 1):
        for w in range(C + 1):
            if c[i-1] <= w:
                # Wybieramy max(vi) przy wagie w-c[i-1]
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - c[i-1]] + v[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    max_value = dp[n][C]

    #Uzyskanie formy bitowej oraz aktualizacja wag
    solution_bits = np.zeros(n, dtype=int)
    current_w = C
    for i in range(n, 0, -1):
        if dp[i][current_w] != dp[i-1][current_w]:
            solution_bits[i-1] = 1
            current_w -= c[i-1]

    return max_value, solution_bits

n = 15
scale = 20
v, c, C = generate_dkp(n, scale)
max_value, solution_bits = solve_dkp(v, c, C)

print(f"Pojemność: {C}")
print(f"Najlepsza wartość: {max_value}")
print(f"Chromosom optymalny: {solution_bits}")


ag = GeneticAlgorithmDKP(n, v, c, C, pop_size=1000, T=100)
max_value, solution_bits = ag.run()

print("\n\nALGORYTM")
print(f"Najlepsza wartość: {max_value}")
print(f"Chromosom najlepszy: {solution_bits}")