#include <iostream>
#include <string>
#include <mpi.h>
#include <random>
#include <cmath>
#include <fstream>
#include "buffer.cpp"

constexpr int n_steps = 1000000;
constexpr double T1 = 0.1;
constexpr double TM  = 2.0;
constexpr int period = 1000;
constexpr int L = 5;
constexpr double kB = 1.0;
constexpr int buffer_size = 1000;

struct Message {
    double temperature;
    int config1[L][L];
    int config2[L][L];
    double energy1;
    double energy2;
    double magnetization1;
    double magnetization2;
    double q;

    Message() = default;
    Message(
        int config1_[L][L],
        int config2_[L][L],
        double energy1_,
        double energy2_,
        double temperature_,
        double magnetization1_,
        double magnetization2_,
        double q_
    ) {
        temperature = temperature_;
        energy1 = energy1_;
        energy2 = energy2_;
        magnetization1 = magnetization1_;
        magnetization2 = magnetization2_;
        q = q_;

        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                config1[i][j] = config1_[i][j];
                config2[i][j] = config2_[i][j];
            }
        }
    }
};

class OverlapWriter {
    string filename = "";
    int max_size;
    double *qs = nullptr;
    int counter;

public:
    OverlapWriter(string filename_ = "overlaps.txt", int max_size_ = 1000) {
        counter = 0;
        filename = filename_;
        max_size = max_size_;
        qs = new double[max_size];
    }

    void writeToFile() {
        std::ofstream file(filename, std::ios::app);
        if (file.is_open()) {
            for (int i = 0; i < counter; i++) {
                file << qs[i] << "\n";
            }
            file.close();
        }
        counter = 0;
    }

    void add(double q) {
        if (counter == max_size) {
            writeToFile();
        }
        qs[counter++] = q;
    }

    ~OverlapWriter() {
        if (counter > 0)
            writeToFile();
        delete[] qs;
    }
};

MPI_Datatype createMessageType() {
    MPI_Datatype message_type;

    const int count = 8;
    int block_lengths[count] = {
        1,              // temperature
        L * L,          // config1
        L * L,          // config2
        1,              // energy1
        1,              // energy2
        1,              // magnetization1
        1,              // magnetization2
        1               // q
    };

    MPI_Datatype types[count] = {
        MPI_DOUBLE,     // temperature
        MPI_INT,        // config1
        MPI_INT,        // config2
        MPI_DOUBLE,     // energy1
        MPI_DOUBLE,     // energy2
        MPI_DOUBLE,     // magnetization1
        MPI_DOUBLE,     // magnetization2
        MPI_DOUBLE      // q
    };

    MPI_Aint displacements[count];
    Message dummy;

    MPI_Aint base_address;
    MPI_Get_address(&dummy, &base_address);
    MPI_Get_address(&dummy.temperature,       &displacements[0]);
    MPI_Get_address(&dummy.config1,           &displacements[1]);
    MPI_Get_address(&dummy.config2,           &displacements[2]);
    MPI_Get_address(&dummy.energy1,           &displacements[3]);
    MPI_Get_address(&dummy.energy2,           &displacements[4]);
    MPI_Get_address(&dummy.magnetization1,    &displacements[5]);
    MPI_Get_address(&dummy.magnetization2,    &displacements[6]);
    MPI_Get_address(&dummy.q,                 &displacements[7]);

    for (int i = 0; i < count; i++) {
        displacements[i] -= base_address;
    }

    MPI_Type_create_struct(count, block_lengths, displacements, types, &message_type);
    MPI_Type_commit(&message_type);

    return message_type;
}


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

void printConfig(int config[L][L], bool writeToFile = false) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            printf("%3d ", config[i][j]); 
        }
        printf("\n");
    }
    if (writeToFile) {
        std::ofstream outFile("config.txt");
        if (outFile.is_open()) {
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    outFile << config[i][j] << " ";
                }
            }
            outFile.close();
        } else {
            printf("Error: Unable to open file for writing.\n");
        }
    }
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

void generateInitialConfiguration(mt19937 &gen, uniform_real_distribution<double> &dist, int config[L][L]) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            auto u = dist(gen);
            config[i][j] = u < 0.5 ? -1:1;
        }
    }
    printConfig(config, true);
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

void calculateObservables(
    int config1[L][L],
    int config2[L][L],
    double J[L*L][L*L],
    double &energy1,
    double &energy2,
    double &magnetization1,
    double &magnetization2,
    double &q
) {
    energy1 = 0.0;
    energy2 = 0.0;
    magnetization1 = 0.0;
    magnetization2 = 0.0;
    q = 0.0;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int idx = i * L + j;
            int spin1 = config1[i][j];
            int spin2 = config2[i][j];

            if (i > 0) {
                int neighbor_idx = (i - 1) * L + j;
                energy1 += J[idx][neighbor_idx] * spin1 * config1[i - 1][j];
                energy2 += J[idx][neighbor_idx] * spin2 * config2[i - 1][j];
            }

            if (j > 0) {
                int neighbor_idx = i * L + (j - 1);
                energy1 += J[idx][neighbor_idx] * spin1 * config1[i][j - 1];
                energy2 += J[idx][neighbor_idx] * spin2 * config2[i][j - 1];
            }

            // magnetization
            magnetization1 += spin1;
            magnetization2 += spin2;

            // overlap
            q += (spin1 * spin2);
        }
    }

    magnetization1 /= (double)(L * L);
    magnetization2 /= (double)(L * L);
    q /= (double)(L * L);
}

double calculateEnergy(
    int config[L][L],
    double J[L*L][L*L]
) {
    double energy = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int idx = i * L + j;
            int spin = config[i][j];

            // Up neighbor (non-periodic)
            if (i > 0) {
                int neighbor_idx = (i - 1) * L + j;
                energy += J[idx][neighbor_idx] * spin * config[i - 1][j];
            }

            // Left neighbor (non-periodic)
            if (j > 0) {
                int neighbor_idx = i * L + (j - 1);
                energy += J[idx][neighbor_idx] * spin * config[i][j - 1];
            }
        }
    }
    return energy;
}

void calculateOverlap(
    int config1[L][L],
    int config2[L][L],
    double &q
) {
    q = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            q += config1[i][j] * config2[i][j];
        }
    }
    q /= (double)(L * L);
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

int main (int argc, char ** argv) {
    int tag = 0;
    int rank, size; 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    // Randoms
    unsigned seed1 = hash<double>{}((double(rank) + 1.0));
    unsigned seed2 = hash<double>{}((double(rank) + 1.0) * 2.0);
    mt19937 gen1(seed1);
    mt19937 gen2(seed2);
    uniform_real_distribution<double> uniform_continuous(0.0, 1.0);
    uniform_int_distribution<int> uniform_discrete(0, L*L-1);

    // Variables
    int config1[L][L];
    int config2[L][L];
    double J[L*L][L*L];
    double Ti = calculateTemperatureWithGeometricSpacing(rank, size);
    printf("Temperature at rank %i: %.4f\n", rank, Ti);
    string filename1 = "observables/observables_" + to_string(rank) + "_1.txt";
    string filename2 = "observables/observables_" + to_string(rank) + "_2.txt";
    string overlap_filename = "overlaps/overlaps_" + to_string(rank) + ".txt";
    Buffer buf1(filename1, buffer_size);
    Buffer buf2(filename2, buffer_size);
    OverlapWriter overlapWriter(overlap_filename, buffer_size);

    MPI_Datatype msg_type = createMessageType();
    int swap = 0;
    int acceptances[] = {0, 0};
    int *global_acceptances = nullptr;
    if (rank == 0) {
        global_acceptances = new int[2 * size];
    }

    // Observable variables
    double energy1, energy2, magnetization1, magnetization2, q;
    double new_energy1, new_energy2, new_magnetization1, new_magnetization2;

    if (rank == 0) {
        normal_distribution<double>normal(0.0, 1.0);
        
        generateInitialConfiguration(gen1, uniform_continuous, config1);
        generateGaussianInteractions(gen2, normal, J, true);
    }
    MPI_Bcast(config1, L*L, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(J, L*L*L*L, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            config2[i][j] = config1[i][j];  // Copy initial configuration
        }
    }

    calculateObservables(
        config1,
        config2, 
        J,
        energy1, 
        energy2,
        magnetization1, 
        magnetization2,
        q
    );

    buf1.setTemperature(Ti);
    buf2.setTemperature(Ti);
    buf1.add(energy1, magnetization1);
    buf2.add(energy2, magnetization2);

    for (int step = 0; step < n_steps; step++) {
        if (step % period == 0) {
            int peer = calculatePeer(swap, size, rank);

            if (peer != MPI_PROC_NULL) {
                Message msgTo(
                    config1, config2, 
                    energy1, energy2, 
                    Ti, magnetization1, magnetization2, q
                );
                Message msgFrom;

                MPI_Sendrecv(
                    &msgTo, 1, msg_type, peer, tag,
                    &msgFrom, 1, msg_type, peer, tag,
                    MPI_COMM_WORLD, &status
                );

                double Tj = msgFrom.temperature;
                double delta1 = (1.0 / Ti - 1.0 / Tj) * (msgFrom.energy1 - energy1);
                double delta2 = (1.0 / Ti - 1.0 / Tj) * (msgFrom.energy2 - energy2);

                double u1 = uniform_continuous(gen1);
                double u2 = uniform_continuous(gen2);

                if (delta1 < 0.0 || u1 < exp(-delta1)) {
                    // Ver si aceptó (sólo los pares corroboran):
                    if (rank % 2 == 0) {
                        if (swap % 2 == 0) acceptances[1] += 1;
                        else acceptances[0] += 1;
                    }
                    // Intercambio
                    for (int i = 0; i < L; ++i)
                        for (int j = 0; j < L; ++j) {
                            config1[i][j] = msgFrom.config1[i][j];
                        }
                    energy1 = msgFrom.energy1;
                    magnetization1 = msgFrom.magnetization1;
                }

                if (delta2 < 0.0 || u2 < exp(-delta2)) {
                    // Ver si aceptó (sólo los pares corroboran):
                    if (rank % 2 == 0) {
                        if (swap % 2 == 0) acceptances[0] += 1;
                        else acceptances[1] += 1;
                    }
                    // Intercambio
                    for (int i = 0; i < L; ++i)
                        for (int j = 0; j < L; ++j) {
                            config2[i][j] = msgFrom.config2[i][j];
                        }
                    energy2 = msgFrom.energy2;
                    magnetization2 = msgFrom.magnetization2;
                }
                calculateOverlap(config1, config2, q);
            }
            tag++;
            swap++;
            buf1.add(energy1, magnetization1);
            buf2.add(energy2, magnetization2);
            overlapWriter.add(q);
            continue;
        }
        // Metropolis normal
        int temp1 = uniform_discrete(gen1);
        int i1 = temp1 / L;
        int j1 = temp1 % L;
        config1[i1][j1] *= -1;

        int temp2 = uniform_discrete(gen2);
        int i2 = temp2 / L;
        int j2 = temp2 % L;
        config2[i2][j2] *= -1;

        calculateObservables(
            config1,
            config2, 
            J,
            new_energy1, 
            new_energy2,
            new_magnetization1, 
            new_magnetization2,
            q
        );

        double delta_energy1 = new_energy1 - energy1;
        double delta_energy2 = new_energy2 - energy2;

        if (delta_energy1 < 0.0) {
            energy1 = new_energy1;
            magnetization1 = new_magnetization1;
        }
        else {
            double u1 = uniform_continuous(gen1);
            if (u1 < exp(-delta_energy1 / (kB * Ti))) {
                energy1 = new_energy1;
                magnetization1 = new_magnetization1;
            } else {
                config1[i1][j1] *= -1;  // Revert change
            }
        }

        if (delta_energy2 < 0.0) {
            energy2 = new_energy2;
            magnetization2 = new_magnetization2;
        }
        else {
            double u2 = uniform_continuous(gen2);
            if (u2 < exp(-delta_energy2 / (kB * Ti))) {
                energy2 = new_energy2;
                magnetization2 = new_magnetization2;
            } else {
                config2[i2][j2] *= -1;  // Revert change
            }
        }

        calculateOverlap(config1, config2, q);
        buf1.add(energy1, magnetization1);
        buf2.add(energy2, magnetization2);
        overlapWriter.add(q);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Type_free(&msg_type);
    MPI_Finalize();
    return 0;
}