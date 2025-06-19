#include <iostream>
#include <random>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <chrono>


using namespace std;
using namespace std::chrono;


// Globals
constexpr int J = -1; // ferromagnetic
constexpr int L = 3;
constexpr int seed = 202; // hardcoded for reproductibility
constexpr int n_iter = 1000000;
constexpr double T = 2.0;
constexpr double kB = 1.0;
const string filename = "energies.txt";

void write_to_file(double val) {
    ofstream file;
    file.open(filename, ios::app);
    if (file.is_open()) {
        file << val;
        file << "\n";
        file.close();
    }
}

double energy(int config [L][L]) {
    double e = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int spin = config[i][j];
            if (i > 0) { // i > 0 implica estar en una fila que no es la primera
                e = e + spin * config[i-1][j];
            }
            if (j > 0) { // una columna que no es la primera
                e = e + spin * config[i][j-1];
            }
        }
    }
    return (- J * e);
}

void print_config(int config[L][L]) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            printf("%3d ", config[i][j]);  // Prints each value with padding
        }
        printf("\n");
    }
}

int main() {
    auto start = high_resolution_clock::now();

    int result = remove(filename.c_str());
    if (result == 0) {
        std::cout << "File deleted successfully." << std::endl;
    } else {
        std::cout << "Error deleting file." << std::endl;
    }
    std::mt19937 gen(seed);
    uniform_real_distribution<double>uniform_continuous(0.0, 1.0);
    uniform_int_distribution<int>uniform_discrete(0, L*L-1);

    int config[L][L];
    for (int i = 0; i < int(pow(L, 2)); i++) {
        double u = uniform_continuous(gen);
        config[i/L][i%L] = u <= 0.5 ? -1:1;
    }

    print_config(config);
    double prev_energy = energy(config);
    cout << "Initial energy: " << prev_energy << endl;
    write_to_file(prev_energy);

    for (int iter = 0; iter < n_iter; iter++) {
        int temp = uniform_discrete(gen);
        int i = temp / L;
        int j = temp % L;

        config[i][j] *= -1;
        double new_energy = energy(config);
        double delta_energy = new_energy - prev_energy;

        if (delta_energy < 0.0) {
            prev_energy = new_energy;
        }
        else {
            double u = uniform_continuous(gen);
            if (u < exp(-delta_energy / (kB * T)))
                prev_energy = new_energy;
            else
                config[i][j] *= -1;
        }
        write_to_file(prev_energy);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "DURATION: " << duration.count() << endl;
    return 0;
}