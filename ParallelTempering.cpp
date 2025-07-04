#include <iostream>
#include <string>
#include <mpi.h>
#include <random>
#include <cmath>
#include "buffer.cpp"

using namespace std;

// Parámetros
constexpr int n_steps = 1000000;
constexpr double T1 = 1.0;
constexpr double TM  = 4.0;
constexpr int period = 1000;
constexpr int L = 5;
constexpr double kB = 1.0;

// Funciones

// Calcular los observables energía y magnetización
pair<double, double> calculateObservables(int config [L][L], double J[L*L][L*L]);
// Imprimir configuración de tamaño L x L
void printConfig(int config[L][L]);
// Definir el tipo de mensaje que se manda entre procesos
MPI_Datatype createMessageType();
// Calcular la temperatura GEOMÉTRICAMENTE que le corresponde al proceso con identificador rank
double calculateTemperatureWithGeometricSpacing (const int rank, const int size);
// Calcular la temperatura UNIFORMEMENTE que le corresponde al proceso con identificador rank
double calculateTemperatureWithUniformSpacing (const int rank, const int size);
// Generar la configuración inicial de los spins
void generateInitialConfiguration(mt19937 &gen, uniform_real_distribution<double> &dist, int config[L][L]);
// Generar interacciones Jij de una muestra Gaussiana (0, 1), sólo para spins adyacentes
void generateGaussianInteractions(
    std::mt19937 &gen,
    std::normal_distribution<double> &normal,
    double J[L*L][L*L],
    bool display
);
void computeAcceptanceRates(int * global_acceptances, const int &size);


// Mensaje que se pasa de proceso a proceso
struct Message {
    int config[L][L];
    double energy;
    double temperature;
    double magnetization;

    Message() = default;
    Message(int config_[L][L], double energy_, double temperature_, double magnetization_) {
        energy = energy_;
        temperature = temperature_;
        magnetization = magnetization_;
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                config[i][j] = config_[i][j];
            }
        }
    }
};

int calculatePeer(const int &swap, const int &size, const int&rank) {
    int peer = MPI_PROC_NULL;

    // El emparejamiento cambia
    if (swap % 2 == 0) {
        if (rank % 2 == 0 && rank + 1 < size) peer = rank + 1;
        else if (rank % 2 == 1) peer = rank - 1;
    } else {
        if (rank % 2 == 0 && rank - 1 >= 0) peer = rank - 1;
        else if (rank % 2 == 1 && rank + 1 < size) peer = rank + 1;
    }
    return peer;
}


int main(int argc, char **argv) {
    // MPI Initialization
    int tag = 0;
    int rank, size; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    // Randoms
    unsigned seed = hash<int>{}(rank);
    mt19937 gen(seed);
    uniform_real_distribution<double>uniform_continuous(0.0, 1.0);
    uniform_int_distribution<int>uniform_discrete(0, L*L-1);

    // Variables
    int config[L][L];
    double J[L*L][L*L];
    int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int next = rank == size-1 ? MPI_PROC_NULL : rank+1;
    double Ti = calculateTemperatureWithGeometricSpacing(rank, size);
    printf("Temperature at rank %i, %.4f\n", rank, Ti);
    string filename = "energies/energies_" + to_string(rank) + ".txt";
    Buffer buf(filename, 1000);
    MPI_Datatype msg_type = createMessageType();
    int swap = 0;
    int acceptances[] = {0, 0};
    int *global_acceptances = nullptr;
    if (rank == 0) {
        global_acceptances = new int[2*size];
    }

    // Generate initial config and J_ij interactions and broadcast it to other processes
    if (rank == 0) {
        normal_distribution<double>normal(0.0, 1.0);

        generateInitialConfiguration(gen, uniform_continuous, config);
        generateGaussianInteractions(gen, normal, J, true);
    }
    MPI_Bcast(config, L*L, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(J, L*L*L*L, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto observables = calculateObservables(config, J);
    double prev_energy = observables.first;
    double prev_magnetization = observables.second;
    buf.setTemperature(Ti);
    buf.add(prev_energy, prev_magnetization);
    

    for (int step = 0; step < n_steps; step++) {
        // Intercambio con otro proceso
        if (step % period == 0) {
            int peer = calculatePeer(swap, size, rank);

            if (peer != MPI_PROC_NULL) {
                // printf("Entered exchange\n");
                Message msgTo(config, prev_energy, Ti, prev_magnetization);
                Message msgFrom;

                MPI_Sendrecv(&msgTo, 1, msg_type, peer, tag,
                            &msgFrom, 1, msg_type, peer, tag,
                            MPI_COMM_WORLD, &status);

                double Tj = msgFrom.temperature;
                double delta = (1.0 / Ti - 1.0 / Tj) * (msgFrom.energy - prev_energy);


                double u = uniform_continuous(gen);
                if (delta < 0.0 || u < exp(-delta)) {
                    // Ver si aceptó (sólo los pares corroboran):
                    if (rank % 2 == 0) {
                        if (swap % 2 == 0) acceptances[1] += 1;
                        else acceptances[0] += 1;
                    }
                    // Intercambio
                    for (int i = 0; i < L; ++i)
                        for (int j = 0; j < L; ++j)
                            config[i][j] = msgFrom.config[i][j];
                    prev_energy = msgFrom.energy;
                }
            }
            tag++;
            swap++;
            buf.add(prev_energy, prev_magnetization);
            continue;
        }

        // Metropolis normal
        int temp = uniform_discrete(gen);
        int i = temp / L;
        int j = temp % L;
        config[i][j] *= -1;

        observables = calculateObservables(config, J);
        double new_energy = observables.first;
        double new_magnetization = observables.second;
        double delta_energy = new_energy - prev_energy;

        if (delta_energy < 0.0) {
            prev_energy = new_energy;
            prev_magnetization = new_magnetization;
        }
        else {
            double u = uniform_continuous(gen);
            if (u < exp(-delta_energy / (kB * Ti))) {
                prev_energy = new_energy;
                prev_magnetization = new_magnetization;
            }
            else {
                config[i][j] *= -1;
            }
        }
        buf.add(prev_energy, prev_magnetization);    
        
    }

    // Calcular probabilidad de aceptar un swap
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(acceptances, 2, MPI_INT, global_acceptances, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        computeAcceptanceRates(global_acceptances, size);
        delete [] global_acceptances;
    }

    MPI_Type_free(&msg_type);
    MPI_Finalize();
    return 0;

}

void computeAcceptanceRates(int * global_acceptances, const int &size) {
    printf("----------------- ACCEPTANCE RATES BY PAIRS -----------------\n");
    for (int i = 0; i < size / 2; i++ ) {
        double rate1 = double(global_acceptances[4*i]) / double(n_steps / (2 * period));
        double rate2 = double(global_acceptances[4*i+1]) / double(n_steps / (2 * period));

        printf("Pair (%i, %i): %.4f acceptance probability.\n", 2*i-1, 2*i, rate1);
        printf("Pair (%i, %i): %.4f acceptance probability.\n", 2*i, 2*i+1, rate2);
    }
    printf("-------------------------------------------------------------\n");
}


pair<double, double> calculateObservables(int config [L][L], double J[L*L][L*L]) {
    double e = 0.0; // energy
    double m = 0.0; // magnetization per spin
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int spin = config[i][j];
            if (i > 0) { // i > 0 implica estar en una fila que no es la primera
                e = e + J[i-1][j] * spin * config[i-1][j];
            }
            if (j > 0) { // una columna que no es la primera
                e = e + J[i][j-1] * spin * config[i][j-1];
            }
            m += spin;
        }
    }
    m /= (L*L);
    return {e, m};
}

MPI_Datatype createMessageType() {
    MPI_Datatype message_type;

    const int count = 4;  // Now three fields
    int block_lengths[count] = { L * L, 1, 1, 1 };
    MPI_Datatype types[count] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Aint displacements[count];
    Message dummy;

    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.config, &displacements[0]);
    MPI_Get_address(&dummy.energy, &displacements[1]);
    MPI_Get_address(&dummy.temperature, &displacements[2]);
    MPI_Get_address(&dummy.magnetization, &displacements[3]);

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
    return T1 * pow(TM / T1, double(rank) / double(size - 1));
}

double calculateTemperatureWithUniformSpacing (const int rank, const int size) {
    double diff = (TM - T1) / double (size - 1);
    return T1 + diff * rank;
}


void generateInitialConfiguration(mt19937 &gen, uniform_real_distribution<double> &dist, int config[L][L]) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            auto u = dist(gen);
            config[i][j] = u < 0.5 ? -1:1;
        }
    }
    printConfig(config);
}

void generateGaussianInteractions(
    std::mt19937 &gen,
    std::normal_distribution<double> &normal,
    double J[L*L][L*L],
    bool display=false
) {
    for (int i = 0; i < L * L; ++i)
        for (int j = 0; j < L * L; ++j)
            J[i][j] = 0.0;

    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            int i = x * L + y;

            // Only add non-periodic neighbors: right and down
            if (x + 1 < L) {
                int j = (x + 1) * L + y;
                double Jij = normal(gen);
                J[i][j] = Jij;
                J[j][i] = Jij;
            }

            if (y + 1 < L) {
                int j = x * L + (y + 1);
                double Jij = normal(gen);
                J[i][j] = Jij;
                J[j][i] = Jij;
            }
        }
    }
    if (display) {
        for (int i = 0; i < L*L; ++i) {
            for (int j = 0; j < L*L; ++j) {
                printf("%.2f ", J[i][j]);  // Prints each value with padding
            }
        printf("\n");
    }
    }
}


void printConfig(int config[L][L]) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            printf("%3d ", config[i][j]);  // Prints each value with padding
        }
        printf("\n");
    }
}