import numpy as np
import matplotlib.pyplot as plt

with open("energies.txt") as file:
    energies = [float(line.strip()) for line in file]

burnin = 10000

energies = np.array(energies[burnin:])
unique, counts = np.unique(energies, return_counts=True)

plt.bar(x=unique, height=counts)
plt.title("Energies frequencies")
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.show()