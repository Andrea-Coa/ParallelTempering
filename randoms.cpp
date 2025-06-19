#include <random>
#include <iostream>

using namespace std;
constexpr int seed = 202; // hardcoded for reproductibility
constexpr int L = 3;

int main() {
    std::mt19937 gen(seed);
    uniform_int_distribution<int>uniform_discrete(0, L*L);
    for (int i = 0; i < 50; i++)
    {
        cout << uniform_discrete(gen) << endl;
    }
    return 0;
} 