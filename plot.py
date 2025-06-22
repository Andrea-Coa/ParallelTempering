import matplotlib.pyplot as plt
import os
import numpy as np

num_temperatures = 8
burn_in = 10000
max_samples = 50000
step = 100         # Sample every `step` entries to reduce autocorrelation
kB = 1.0

def stream_energy(file_path, burn_in, max_samples=None, step=1):
    with open(file_path, 'r') as f:
        # la primera línea del archivo es la temperatura
        T = float(next(f).strip())
        # las demás sí son energía
        for _ in range(burn_in):
            next(f)
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if i % step == 0:
                yield float(line.strip()), T

def plot_histograms():
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()

    for i in range(num_temperatures):
        filename = f"energies/energies_{i}.txt"
        if not os.path.exists(filename):
            print(f"{filename} not found.")
            continue

        energy_stream = list(stream_energy(filename, burn_in, max_samples, step))
        if not energy_stream:
            continue

        energies, T = zip(*energy_stream)
        energies = np.array(energies)
        T = T[0]

        ax = axes[i]
        counts, bins, _ = ax.hist(energies, bins=50, alpha=0.6, color='steelblue', density=True)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        boltz = np.exp(-bin_centers / (kB * T))
        boltz /= np.trapz(boltz, bin_centers)  # Normalize

        ax.plot(bin_centers, boltz, 'r--', label='Boltzmann shape')
        ax.set_title(f"T = {T:.2f}")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Probability density")
        ax.legend()

    plt.tight_layout()
    plt.savefig("energy_histograms_with_boltzmann.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_histograms()
