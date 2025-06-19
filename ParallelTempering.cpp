#include <iostream>
#include <string>
#include <mpi.h>
#include <random>
#include <cmath>
#include "buffer.cpp"

// Parámetros
constexpr int n_steps = 100000;
constexpr double T1 = 0.2;
constexpr double TM  = 3.0;
constexpr int period = 1000;
constexpr int L = 3;
constexpr double J = -1.0;
constexpr double kB = 1.0;

struct Message {
    int config[L][L];
    double energy;
    Message() = default;
    Message(int config_[L][L], double energy_) {
        energy = energy_;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                config[i][j] = config_[i][j];
            }
        }
    }
};

MPI_Datatype createMessageType() {
    MPI_Datatype message_type;

    // 2 fields: config and energy
    const int count = 2;
    int block_lengths[count] = { L * L, 1 };
    MPI_Datatype types[count] = { MPI_INT, MPI_DOUBLE };

    // Displacements
    MPI_Aint displacements[count];
    Message dummy;

    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.config, &displacements[0]);
    MPI_Get_address(&dummy.energy, &displacements[1]);

    for (int i = 0; i < count; i++) {
        displacements[i] -= base_address;
    }

    MPI_Type_create_struct(count, block_lengths, displacements, types, &message_type);
    MPI_Type_commit(&message_type);

    return message_type;
}

double calculateTemperatureWithGeometricSpacing (const int rank, const int size) {
    if (rank == 0) return T1;
    if (rank == size-1) return TM;
    return T1 * pow(TM / T1, rank / (size - 1));
}

void generateInitialConfiguration(mt19937 &gen, uniform_real_distribution<double> &dist, int config[L][L]) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            auto u = dist(gen);
            config[i][j] = u < 0.5 ? -1:1;
        }
    }
    print_config(config);
}

int main(int argc, char **argv) {
    // MPI Initialization
    int tag = 0;
    int rank, size; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Randoms
    unsigned seed = hash<int>{}(rank);
    mt19937 gen(seed);
    uniform_real_distribution<double>uniform_continuous(0.0, 1.0);
    uniform_int_distribution<int>uniform_discrete(0, L*L-1);

    // Variables
    int config[L][L];
    int swap = 0;
    int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int next = rank == size-1 ? MPI_PROC_NULL : rank+1;
    int Ti = calculateTemperatureWithGeometricSpacing(rank, size);
    string filename = "energies_" + to_string(Ti) + ".txt";
    Buffer buf(filename, 1000);
    MPI_Datatype msg_type = createMessageType();
    MPI_Request req;


    // Generate initial config and broadcast it to other processes
    if (rank == 0) {
        generateInitialConfiguration(gen, uniform_continuous, config);
    }
    MPI_Bcast(config, L*L, MPI_INT, 0, MPI_COMM_WORLD);
    double prev_energy = energy(config);
    buf.add(prev_energy);
    

    for (int step = 0; step < n_steps; step++) {
        // if step % period != 0 then do normal Metropolis Hastings
        // else attempt swap


        if (step % period == 0) {
            Message msgTo(config, prev_energy);
            Message msgFrom;
            if (swap % 2 == 0 && rank % 2 == 0) {
                // En los swaps pares, los ranks pares son los que mandan
                MPI_Irecv(&msgFrom, 1, msg_type, next, tag, MPI_COMM_WORLD, &req);
            }
            // Send both? Or send one first?
            swap++;
        }

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
            if (u < exp(-delta_energy / (kB * Ti)))
                prev_energy = new_energy;
            else
                config[i][j] *= -1;
        }
        buf.add(prev_energy);
        
        
    }
    MPI_Type_free(&msg_type);
    MPI_Finalize();
    return 0;

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