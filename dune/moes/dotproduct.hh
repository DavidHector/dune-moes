#ifndef DUNE_MOES_DOTPRODUCT_HH
#define DUNE_MOES_DOTPRODUCT_HH
#include <vector>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <dune/moes/vectorclass/vectorclass.h>


double dP(const std::vector<double>& u, const std::vector<double>& v){
    double sum = 0.0;
    for (size_t i = 0; i < u.size(); i++)
    {
        sum += u[i] * v[i];
    }
    return sum;
}

double dPV(const std::vector<double>& u, const std::vector<double>& v){
    Vec4d sumV = 0.0;
    double sum = 0.0;
    Vec4d ui = 0.0;
    Vec4d vi = 0.0;
    for (size_t i = 0; i < u.size(); i+=4)
    {
        ui.load(&u[i]);
        vi.load(&v[i]);
        sumV = mul_add(ui, vi, sumV);
    }
    sum = horizontal_add(sumV);
    return sum;
}

template<typename DPF>
void timeDP(DPF& dPF, const size_t N, const size_t repetitions){
    std::vector<double> u(N, 0.0);
    std::vector<double> v(N, 0.0);
    for (size_t i = 0; i < N; i++)
    {
        u[i] = static_cast<double> (std::rand()) / RAND_MAX;
        v[i] = static_cast<double> (std::rand()) / RAND_MAX;
    }
    double tmp = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        tmp = dPF(u, v);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double) duration.count()/repetitions;
    double gflops = 2.0*N / averageDuration; // flop/ns = Gflop/s
    double memoryBW = 2*N*8/averageDuration; // in GB/s
    std::cout << "GFLOPS: " << gflops << " (Max: 64GFLOPS)" << std::endl;
    std::cout << "Memory Bandwidth: " << memoryBW << "GB/s (Max: 25.6GB/s)" << std::endl;
    std::cout << "Memory Usage: " << 2.0*N*8.0/1e6 << "MB" << std::endl;
    std::cout << "Average Duration: " << averageDuration/1e6 << "ms" << std::endl;
}




#endif // DUNE_MOES_DOTPRODUCT_HH