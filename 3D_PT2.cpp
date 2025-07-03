#include <iostream>
#include <string>
#include <mpi.h>
#include <random>
#include <cmath>
#include <fstream>
#include "buffer.cpp"

constexpr int n_steps = 10000000;
constexpr double T1 = 0.1;
constexpr double TM  = 2.0;
constexpr int period = 1000;
constexpr int L = 5;
constexpr double kB = 1.0;
constexpr int buffer_size = 1000;

struct Message {
    double temperature;
    int config1[L][L][L];
    int config2[L][L][L];
    double energy1;
    double energy2;
    double magnetization1;
    double magnetization2;
    double q;

    Message() = default;
    Message(
        int config1_[L][L][L],
        int config2_[L][L][L],
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
                for (int k = 0; k < L; k++) {
                    config1[i][j][k] = config1_[i][j][k];
                    config2[i][j][k] = config2_[i][j][k];

                }
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
        L * L * L,      // config1
        L * L * L,      // config2
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

void printConfig(int config[L][L][L], bool writeToFile = false) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                printf("%3d ", config[i][j][k]); 
            }
        }
        printf("\n");
    }
    if (writeToFile) {
        std::ofstream outFile("config.txt");
        if (outFile.is_open()) {
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    for (int k = 0; k < L; ++k) {
                        outFile << config[i][j][k] << " ";
                    }
                }
            }
            outFile.close();
        } else {
            printf("Error: Unable to open file for writing.\n");
        }
    }
}

void generateGaussianInteractions3D(
    std::mt19937 &gen,
    std::normal_distribution<double> &normal,
    double J[L*L*L][L*L*L],
    bool display = false
) {
    // Initialize interaction matrix
    for (int i = 0; i < L * L * L; ++i)
        for (int j = 0; j < L * L * L; ++j)
            J[i][j] = 0.0;

    for (int x = 0; x < L; ++x) {
        for (int y = 0; y < L; ++y) {
            for (int z = 0; z < L; ++z) {
                int i = x * L * L + y * L + z;

                // Right neighbor (x+1, y, z)
                if (x + 1 < L) {
                    int j = (x + 1) * L * L + y * L + z;
                    double Jij = normal(gen);
                    J[i][j] = Jij;
                    J[j][i] = Jij;
                }

                // Up neighbor (x, y+1, z)
                if (y + 1 < L) {
                    int j = x * L * L + (y + 1) * L + z;
                    double Jij = normal(gen);
                    J[i][j] = Jij;
                    J[j][i] = Jij;
                }

                // Forward neighbor (x, y, z+1)
                if (z + 1 < L) {
                    int j = x * L * L + y * L + (z + 1);
                    double Jij = normal(gen);
                    J[i][j] = Jij;
                    J[j][i] = Jij;
                }
            }
        }
    }

    if (display) {
        for (int i = 0; i < L * L * L; ++i) {
            for (int j = 0; j < L * L * L; ++j)
                printf("%.2f ", J[i][j]);
            printf("\n");
        }
    }
}


void generateInitialConfiguration(mt19937 &gen, uniform_real_distribution<double> &dist, int config[L][L][L]) {
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                auto u = dist(gen);
                config[i][j][k] = u < 0.5 ? -1:1;
            }
        }
    }
    printConfig(config, false);
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
    int config1[L][L][L],
    int config2[L][L][L],
    double J[L*L*L][L*L*L],
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

    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            for (int z = 0; z < L; z++) {
                int idx = x * L * L + y * L + z;
                int s1 = config1[x][y][z];
                int s2 = config2[x][y][z];

                // Energy: sum over neighbors in +x, +y, +z directions
                if (x + 1 < L) {
                    int j = (x + 1) * L * L + y * L + z;
                    energy1 += -J[idx][j] * s1 * config1[x + 1][y][z];
                    energy2 += -J[idx][j] * s2 * config2[x + 1][y][z];
                }
                if (y + 1 < L) {
                    int j = x * L * L + (y + 1) * L + z;
                    energy1 += -J[idx][j] * s1 * config1[x][y + 1][z];
                    energy2 += -J[idx][j] * s2 * config2[x][y + 1][z];
                }
                if (z + 1 < L) {
                    int j = x * L * L + y * L + (z + 1);
                    energy1 += -J[idx][j] * s1 * config1[x][y][z + 1];
                    energy2 += -J[idx][j] * s2 * config2[x][y][z + 1];
                }

                // Magnetization and overlap
                magnetization1 += s1;
                magnetization2 += s2;
                q += s1 * s2;
            }
        }
    }

    int volume = L * L * L;
    magnetization1 /= (double)volume;
    magnetization2 /= (double)volume;
    q /= (double)volume;
}


double calculateEnergy(
    int config[L][L][L],
    double J[L*L*L][L*L*L]
) {
    double energy = 0.0;

    for (int x = 0; x < L; x++) {
        for (int y = 0; y < L; y++) {
            for (int z = 0; z < L; z++) {
                int idx = x * L * L + y * L + z;
                int spin = config[x][y][z];

                // Right neighbor (+x)
                if (x + 1 < L) {
                    int neighbor_idx = (x + 1) * L * L + y * L + z;
                    energy += -J[idx][neighbor_idx] * spin * config[x + 1][y][z];
                }

                // Up neighbor (+y)
                if (y + 1 < L) {
                    int neighbor_idx = x * L * L + (y + 1) * L + z;
                    energy += -J[idx][neighbor_idx] * spin * config[x][y + 1][z];
                }

                // Forward neighbor (+z)
                if (z + 1 < L) {
                    int neighbor_idx = x * L * L + y * L + (z + 1);
                    energy += -J[idx][neighbor_idx] * spin * config[x][y][z + 1];
                }
            }
        }
    }

    return energy;
}


void calculateOverlap(
    int config1[L][L][L],
    int config2[L][L][L],
    double &q
) {
    q = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                q += config1[i][j][k] * config2[i][j][k];

            }
        }
    }
    q /= (double)(L * L * L);
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
    uniform_int_distribution<int> uniform_discrete(0, L*L*L-1);

    // Variables
    int config1[L][L][L];
    int config2[L][L][L];
    double J[L*L*L][L*L*L];
    double Ti = calculateTemperatureWithGeometricSpacing(rank, size);
    printf("Temperature at rank %i: %.4f\n", rank, Ti);
    string filename1 = "observables3D/observables_" + to_string(rank) + "_1.txt";
    string filename2 = "observables3D/observables_" + to_string(rank) + "_2.txt";
    string overlap_filename = "overlaps3D/overlaps_" + to_string(rank) + ".txt";
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
        generateGaussianInteractions3D(gen2, normal, J, false);
    }
    MPI_Bcast(config1, L*L*L, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(J, L*L*L*L*L*L, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                config2[i][j][k] = config1[i][j][k];  // Copy initial configuration
            }
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
                            for (int k = 0; k < L; ++k) {
                                config1[i][j][k] = msgFrom.config1[i][j][k];
                            }
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
                            for (int k = 0; k < L; ++k) {
                                config2[i][j][k] = msgFrom.config2[i][j][k];
                            }
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
        int i1 = temp1 / (L * L);
        int j1 = (temp1 / L) % L;
        int k1 = temp1 % L;
        config1[i1][j1][k1] *= -1;

        int temp2 = uniform_discrete(gen2); 
        int i2 = temp2 / (L * L);
        int j2 = (temp2 / L) % L;
        int k2 = temp2 % L;
        config2[i2][j2][k2] *= -1;

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
                config1[i1][j1][k1] *= -1;  // Revert change
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
                config2[i2][j2][k2] *= -1;  // Revert change
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