#include <iostream>
#include <vector>
#include <cmath>

#ifdef USE_DOUBLE
    using num_t = double;
#else
    using num_t = float;
#endif

int main() {
    const size_t N = 10000000;
    std::vector<num_t> arr(N);
    num_t sum = 0;

    for (size_t i = 0; i < N; ++i) {
        num_t x = (2 * M_PI * i) / N;;
        arr[i] = std::sin(x);
        sum += arr[i];
    }

    std::cout << "Sum: " << sum << std::endl;
    return 0;
}