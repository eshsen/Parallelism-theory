#include <iostream>
#include <vector>
#include <cmath>

#ifdef USE_DOUBLE
    using num_t = double;
#else
    using num_t = float;
#endif

#define N 10000000
#define PI 3.1415926535

int main() {
    std::vector<num_t> arr;
    num_t sum = 0;

    for (size_t i = 0; i < N; ++i) {
        num_t x = (num_t)sin((2 * PI * i) / N);
        arr.push_back(x);
        sum += x;
    }

    std::cout << "Sum: " << sum << std::endl;
    return 0;
}