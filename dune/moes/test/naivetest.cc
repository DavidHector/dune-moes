#include <vector>
#include <iostream>
#include <chrono>

#include <dune/moes/naive.hh>
#include <dune/moes/dotproduct.hh>

size_t flopsFactors(const size_t& N, const size_t& rhsWidth){
    return 2*N + rhsWidth*(2*N + 1);
}

double flopsQR(const size_t& N, const size_t& W){
    return W*(4.0*N) + 0.5 * W*(W-1) * (2.0*N + 1.0 + 2.0*N);
}

double memBandwidthGB(const size_t&N, const size_t&W, const double averageTimeNs){
    double memoryReq = N*W*8.0;
    double bW = memoryReq/averageTimeNs; // Bytes/nanosecond = Gigabytes/second
    return bW;
}

template<typename QR, typename F, typename QTYPE>
void getGFLOPSqr(QTYPE& Q, QR qr, F f, size_t N, size_t rhsWidth, size_t repetitions){
    double flops = f(N, rhsWidth);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        qr(Q, 5000);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double) duration.count()/repetitions;
    auto gFlops = (flops/averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    std::cout << "Average Duration: " << averageDuration/1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N*rhsWidth*8.0/1e6 << "MB" << std::endl; 
    std::cout << "Memory Bandwidth: " << memBandwidthGB(N, rhsWidth, averageDuration) << "GB/s (max: 25.6GB/s)" << std::endl;
    std::cout << "GFLOPS: " << gFlops << std::endl;
}

template<typename GF, typename F, typename QTYPE, typename FACTORSTYPE>
void getGFLOPSfactors(QTYPE& Q, GF gf, F f, size_t N, size_t rhsWidth, size_t repetitions, FACTORSTYPE& factors, const size_t& orthIndex){
    double flops = f(N, rhsWidth);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        gf(Q, factors, orthIndex);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double) duration.count()/repetitions;
    auto gFlops = (flops/averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    std::cout << "GFLOPS: " << gFlops << std::endl;
}


int main(int argc, char const *argv[])
{
    size_t N = 1000;
    size_t rhsWidth = 256;
    size_t repetitions = 10;
    const double tolerance = 1e-6;

    if (argc > 1)
    {
        if (1 == std::sscanf(argv[1], "%zu", &N))
        {
        } else {
            std::cout << "Please enter an unsigned integer!" << std::endl;
            return -1;
        }
    }
    if (argc > 2)
    {
        if (1 == std::sscanf(argv[2], "%zu", &rhsWidth))
        {
        } else {
            std::cout << "Please enter a power of 32!" << std::endl;
            return -1;
        }
    }
    if (argc > 3)
    {
        if (1 == std::sscanf(argv[3], "%zu", &repetitions))
        {
        } else {
            std::cout << "Please enter an unsigned integer!" << std::endl;
            return -1;
        }
    }
    std::vector<std::vector<double>> Q(rhsWidth, std::vector<double>(N, 0.0)); // The matrix

    /*
    std::vector<double> factors(rhsWidth, 0.0);

    const int NConst = 1000000;
    const int rhsWidthConst = 256;

    std::cout << "Before creating the array" << std::endl;
    double** QNative;
    double* factorsNative;
    QNative = new double *[rhsWidthConst];
    for (int i = 0; i < rhsWidthConst; i++)
    {
        QNative[i] = new double[NConst];
    }
    
    factorsNative = new double[rhsWidthConst];
    fillMatrixRandom(Q);
    std::cout << "Before filling the array" << std::endl;
    fillMatrixRandomNative<1000000, 256>(QNative);
    std::cout << "Yay we could fill the native array" << std::endl;
    std::cout << "Repetitions: " << repetitions << std::endl;
    auto startN = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        getFactorsSingleFunctionNative<NConst, rhsWidthConst>(QNative, factorsNative, 0);
        //getFactorsSingleFunction(Q, factors, 0);
    }
    auto stopN = std::chrono::high_resolution_clock::now();
    auto durationN = std::chrono::duration_cast<std::chrono::nanoseconds>(stopN - startN);
    auto averageDurationN = (double) durationN.count()/repetitions;
    double flopsgetFactors = (double) flopsFactors(N, rhsWidth);
    double flopsgetFactorsNative = (double) flopsFactors(NConst, rhsWidthConst);
    auto gFlopsNative = (flopsgetFactorsNative/averageDurationN) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    std::cout << "getFactors: " <<std::endl;
    getGFLOPSfactors(Q, getFactors, flopsFactors, N, rhsWidth, repetitions, factors, 0);
    std::cout << "getFactorsNative(average GFLOPS): " << gFlopsNative << std::endl;
    */
   /*
    std::cout << "QR-algorithm (non-vectorized): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qr, flopsQR, N, rhsWidth, repetitions);
    checkOrthoNormality(Q, tolerance);
    std::cout << std::endl;
    */

    /*
    std::cout << "QR-algorithm (vectorized): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qrV, flopsQR, N, rhsWidth, repetitions);
    checkOrthoNormality(Q, tolerance);
    std::cout << std::endl;
    */
    
    /*
    std::cout << "QR-algorithm (naively vectorized): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qrVN, flopsQR, N, rhsWidth, repetitions);
    checkOrthoNormality(Q, tolerance);
    std::cout << std::endl;
    */

    /*
    std::cout << "QR-algorithm (parallel columns, non-vectorized): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qrPar, flopsQR, N, rhsWidth, repetitions);
    checkOrthoNormality(Q, tolerance);
    std::cout << std::endl;
    */
    
    std::cout << "QR-algorithm (parallel columns, vectorized): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qrParV, flopsQR, N, rhsWidth, repetitions);
    checkOrthoNormality(Q, tolerance);
    std::cout << std::endl;
    
    /*
    std::cout << "QR-algorithm (non-vectorized, tiled): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qrTiled, flopsQR, N, rhsWidth, repetitions);
    */

    /*
    std::cout << "Dot product (non-vectorized): " << std::endl;
    timeDP(dP, N, repetitions);
    std::cout << std::endl;

    std::cout << "Dot product (Vectorized): " << std::endl;
    timeDP(dPV, N, repetitions);
    std::cout << std::endl;
    */

    return 0;
}

