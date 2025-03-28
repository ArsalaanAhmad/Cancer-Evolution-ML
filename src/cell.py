import numpy as np

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dna = self.random_dna()
        self.drug_resistance = self.calculate_resistance()
        self.alive = True

    def random_dna(self):
        return np.random.rand(10)  # 10 "mutation" genes

    def calculate_resistance(self):
        return np.mean(self.dna)  # Higher avg = more resistant

    def mutate(self):
        mutation_chance = 0.05
        mutation = np.random.normal(0, mutation_chance, size=self.dna.shape)
        self.dna += mutation
        self.dna = np.clip(self.dna, 0, 1)  # Keep values between 0 and 1
        self.drug_resistance = self.calculate_resistance()

