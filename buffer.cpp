#include <iostream>
#include <fstream>
using namespace std;

class Buffer {
    string filename = "";
    int max_size;
    double *contents = nullptr;
    int counter;

public:
    Buffer (string filename_ = "energies.txt", int max_size_ = 1000) {
        counter = 0;
        filename = filename_;
        max_size = max_size_;
        contents = new double[max_size];
    }

    void writeToFile() {
        ofstream file;
        file.open(filename, ios::app);
        if (file.is_open()) {
            for (int i = 0; i < counter; i++) {
                file << contents[i];
                file << "\n";
                file.close();
            }
        }
        counter = 0;
    }

    void add(double val) {
        if (counter == max_size) {
            writeToFile();
        }
        contents[counter]  = val;
        counter ++;
    }

    ~Buffer() {
        if (counter > 0)
            writeToFile();
        delete [] contents;
    }
};