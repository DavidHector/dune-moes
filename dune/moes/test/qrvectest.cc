#include <vector>
#include <iostream>
#include <chrono>

#include <dune/moes/qrvec.hh>

void printMatrix(const std::vector<std::vector<double>>& Q){
    std::cout << std::endl;
    for (size_t row = 0; row < Q.size(); row++)
    {
        for (size_t col = 0; col < Q[0].size(); col++)
        {
            std::cout << Q[row][col] << "  \t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
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
        qr(Q);
    }
    std::cout << "Algorithm finished! Calculating metrics..." << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double) duration.count()/repetitions;
    auto gFlops = (flops/averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    std::cout << "Average Duration: " << averageDuration/1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N*rhsWidth*8.0/1e6 << "MB" << std::endl; 
    std::cout << "Memory Bandwidth: " << memBandwidthGB(N, rhsWidth, averageDuration) << "GB/s (max: 25.6GB/s)" << std::endl;
    std::cout << "GFLOPS: " << gFlops << std::endl;
}

template<typename QR, typename F>
void getGFLOPSqrArr(double** Q, QR qr, F f, size_t N, size_t rhsWidth, size_t repetitions){
    double flops = f(N, rhsWidth);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        qr(Q, N, rhsWidth);
    }
    std::cout << "Algorithm finished! Calculating metrics..." << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double) duration.count()/repetitions;
    auto gFlops = (flops/averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    std::cout << "Average Duration: " << averageDuration/1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N*rhsWidth*8.0/1e6 << "MB" << std::endl; 
    std::cout << "Memory Bandwidth: " << memBandwidthGB(N, rhsWidth, averageDuration) << "GB/s (max: 25.6GB/s)" << std::endl;
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
    std::vector<std::vector<double>> Q(N, std::vector<double>(rhsWidth, 0.0)); // The matrix
    double** QArr = new double*[N];
    for (size_t i = 0; i < N; i++)
    {
        QArr[i] = new double[rhsWidth];
    }
     

    
    std::cout << "QR-algorithm (scalar, row major): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qr<16>, flopsQR, N, rhsWidth, repetitions);
    // checkOrthoNormality(Q, tolerance);
    std::cout << std::endl;

    std::cout << "QR-algorithm (vectorized, row major): " << std::endl;
    fillMatrixRandom(Q);
    getGFLOPSqr(Q, qrV<16>, flopsQR, N, rhsWidth, repetitions);
    // checkOrthoNormality(Q, tolerance);
    std::cout << std::endl;

    std::cout << "QR-algorithm (vectorized, row major, no std::vector): " << std::endl;
    fillMatrixRandom(QArr, N, rhsWidth);
    getGFLOPSqrArr(QArr, qrVArr<16>, flopsQR, N, rhsWidth, repetitions);
    // checkOrthoNormality(QArr, tolerance, N, rhsWidth);
    std::cout << std::endl;
    return 0;
}