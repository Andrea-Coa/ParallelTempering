#include <iostream>
#include <fstream>
using namespace std;

class Buffer {
    string filename = "";
    int max_size;
    double *energies = nullptr;
    double *magnetizations = nullptr;
    int counter;

public:
    Buffer (string filename_ = "energies.txt", int max_size_ = 1000) {
        counter = 0;
        filename = filename_;
        max_size = max_size_;
        energies = new double[max_size];
        magnetizations = new double[max_size];
    }

    void writeToFile() {
        ofstream file;
        file.open(filename, ios::app);
        if (file.is_open()) {
            for (int i = 0; i < counter; i++) {
                file << energies[i] << " " << magnetizations[i];
                file << "\n";
            }
            file.close();
        }
        counter = 0;
    }
    void setTemperature(double temperature) {
        ofstream file;
        file.open(filename, ios::app);
        if (file.is_open()) {
            file << temperature;
            file << "\n";
            file.close();
        }
    }

    void add(double energy, double magnetization) {
        if (counter == max_size) {
            writeToFile();
        }
        energies[counter]  = energy;
        magnetizations[counter] = magnetization;
        counter++;
    }

    ~Buffer() {
        if (counter > 0)
            writeToFile();
        delete [] energies;
        delete [] magnetizations;
    }
};
