#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct Record {
    std::size_t task_id{};
    int func_id{};
    std::vector<double> args;
    double result{};
};

double func_sin(double x) {
    return std::sin(x);
}

double func_sqrt(double x) {
    return std::sqrt(x);
}

double func_pow(double x, double y) {
    return std::pow(x, y);
}

std::vector<Record> read_records(const std::string& filename) {
    std::vector<Record> records;
    std::ifstream in(filename);
    std::string line;

    while (std::getline(in, line)) {
        std::istringstream iss(line);
        Record rec;
        std::size_t argc = 0;
        iss >> rec.task_id >> rec.func_id >> argc;
        rec.args.resize(argc);
        for (std::size_t i = 0; i < argc; ++i) {
            iss >> rec.args[i];
        }
        iss >> rec.result;
        records.push_back(rec);
    }

    return records;
}

double expected_value(int func_id, const std::vector<double>& args) {
    if (func_id == 1 && args.size() == 1) {
        return func_sin(args[0]);
    }
    if (func_id == 2 && args.size() == 1) {
        return func_sqrt(args[0]);
    }
    if (func_id == 3 && args.size() == 2) {
        return func_pow(args[0], args[1]);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

std::string func_name(int func_id) {
    if (func_id == 1) return "sin";
    if (func_id == 2) return "sqrt";
    if (func_id == 3) return "pow";
    return "unknown";
}

void print_args(const std::vector<double>& args) {
    std::cout << "(";
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << args[i];
    }
    std::cout << ")";
}

void check_file(const std::string& filename) {
    auto records = read_records(filename);

    std::size_t wrong = 0;
    std::cout << "Checking: " << filename << "\n";
    std::cout << "Total records: " << records.size() << "\n";

    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& rec = records[i];
        double expected = expected_value(rec.func_id, rec.args);

        double local_eps = 1e-4;
        if (rec.func_id == 3) {
            local_eps = 1e-2;
        }

        double diff = std::abs(expected - rec.result);
        if (std::isnan(expected) || diff > local_eps) {
            ++wrong;
            std::cout << "Error in test #" << (i + 1)
                      << " (task_id=" << rec.task_id << ", func=" << func_name(rec.func_id) << ")\n";
            std::cout << "  args      = ";
            print_args(rec.args);
            std::cout << "\n";
            std::cout << "  expected  = " << std::setprecision(15) << expected << "\n";
            std::cout << "  got       = " << std::setprecision(15) << rec.result << "\n";
            std::cout << "  diff      = " << std::setprecision(15) << diff << "\n\n";
        }
    }

    double accuracy = records.empty() ? 0.0
        : 100.0 * (records.size() - wrong) / records.size();

    std::cout << "Wrong: " << wrong << "\n";
    std::cout << "Accuracy: " << std::fixed
              << std::setprecision(2) << accuracy << "%\n\n";
}

int main() {
    check_file("results/client1_sin.txt");
    check_file("results/client2_sqrt.txt");
    check_file("results/client3_pow.txt");
    return 0;
}