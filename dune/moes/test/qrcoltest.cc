#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>

#include <dune/moes/qrcol.hh>
#include <dune/moes/naive.hh>
#include <dune/moes/vectorclass/vectorclass.h>

std::mutex printMutex;

double flopsQR(const size_t &N, const size_t &W)
{
    return W * (4.0 * N) + 0.5 * W * (W - 1) * (2.0 * N + 1.0 + 2.0 * N);
}

double average(std::vector<double> Arr)
{
    double sum = 0.0;
    for (double i : Arr)
    {
        sum += i;
    }
    return sum / Arr.size();
}

double standard_deviation(std::vector<double> Arr, double average)
{
    double stddev = 0.0;
    for (double i : Arr)
    {
        stddev += (i - average) * (i - average);
    }
    stddev /= Arr.size();
    return std::sqrt(stddev);
}

double memBandwidthGB(const size_t &N, const size_t &W, const double averageTimeNs)
{
    double memoryReq = N * W * 8.0;
    double bW = memoryReq / averageTimeNs; // Bytes/nanosecond = Gigabytes/second
    return bW;
}

template <typename QR, typename F>
void getGFLOPSqr(Vec4d *Q, QR qr, F f, size_t N, size_t rhsWidth, size_t repetitions, double &gFlops, size_t blockSize = 2, size_t uBlockSize = 1)
{
    double flops = f(N, rhsWidth);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        qr(Q, N, rhsWidth / blockSize / 4, blockSize, uBlockSize);
    }
    std::cout << "Algorithm finished! Calculating metrics..." << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double)duration.count() / repetitions;
    gFlops = (flops / averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    std::cout << "Average Duration: " << averageDuration / 1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N * rhsWidth * 8.0 / 1e6 << "MB" << std::endl;
    std::cout << "Memory Bandwidth: " << memBandwidthGB(N, rhsWidth, averageDuration) << "GB/s (max: 25.6GB/s)" << std::endl;
    std::cout << "GFLOPS: " << gFlops << std::endl;
}

template <typename QR, typename F>
void getGFLOPSqr(std::unique_ptr<double[]> &Q, QR qr, F f, size_t N, size_t rhsWidth, size_t repetitions, double &gFlops, size_t threadNumber = 0, size_t blockSize = 2, size_t uBlockSize = 1)
{
    double flops = f(N, rhsWidth);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        qr(Q, N, rhsWidth / blockSize / 4, blockSize, uBlockSize);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double)duration.count() / repetitions;
    gFlops = (flops / averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    const std::lock_guard<std::mutex> lock(printMutex);
    std::cout << "Thread Number (0 if single threaded): " << threadNumber << std::endl;
    std::cout << "Average Duration: " << averageDuration / 1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N * rhsWidth * 8.0 / 1e6 << "MB" << std::endl;
    std::cout << "GFLOPS: " << gFlops << std::endl;
}

template <typename QR, typename F>
void getGFLOPSqrNaive(double *Q, QR qr, F f, size_t N, size_t rhsWidth, size_t repetitions, double &gFlops, size_t threadNumber = 0, size_t blockSize = 2, size_t uBlockSize = 1)
{
    double flops = f(N, rhsWidth);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        qr(Q, N, rhsWidth);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double)duration.count() / repetitions;
    gFlops = (flops / averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    const std::lock_guard<std::mutex> lock(printMutex);
    std::cout << "Thread Number (0 if single threaded): " << threadNumber << std::endl;
    std::cout << "Average Duration: " << averageDuration / 1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N * rhsWidth * 8.0 / 1e6 << "MB" << std::endl;
    std::cout << "GFLOPS: " << gFlops << std::endl;
}

void singleThreadTest(size_t N, size_t rhsWidth, size_t repetitions, const double tolerance, size_t threadNumber, double &gFlops)
{
    size_t matrixSizeDouble = N * rhsWidth;
    // double *Qdouble = new double[matrixSizeDouble];
    std::unique_ptr<double[]> Qdouble(new double[matrixSizeDouble]);
    fillMatrixRandom(Qdouble, matrixSizeDouble);
    getGFLOPSqr(Qdouble, qrFixedBlockOptimizedDoubleUnique, flopsQR, N, rhsWidth, repetitions, gFlops, threadNumber);
    checkOrthoNormalityFixed(Qdouble, N, rhsWidth / 8, tolerance);
    // printMatrix(Qdouble, N, rhsWidth);
    // delete[] Qdouble;
}

void singleThreadTestNaive(size_t N, size_t rhsWidth, size_t repetitions, const double tolerance, size_t threadNumber, double &gFlops)
{
    size_t matrixSizeDouble = N * rhsWidth;
    double *Qdouble = new double[matrixSizeDouble];
    fillMatrixRandom(Qdouble, matrixSizeDouble);
    getGFLOPSqrNaive(Qdouble, qrNaiveQNaive, flopsQR, N, rhsWidth, repetitions, gFlops, threadNumber);
    checkOrthoNormalityNaive(Qdouble, N, rhsWidth, tolerance);
    delete[] Qdouble;
}

void autotest(const double tolerance)
{
    const int lenN = 7;
    const int lenrhsWidth = 8;
    size_t Ns[lenN] = {1000, 5000, 10000, 20000, 50000, 100000, 1000000};
    size_t repetitions[lenN] = {500, 100, 50, 10, 10, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlops;
    std::ofstream outputFile;
    outputFile.open("QRGFLOPs.csv");
    outputFile << "N,rhsWidth,repetitions,GFLOPs,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            singleThreadTest(Ns[i], rhsWidths[j], repetitions[i], tolerance, 0, gFlops);
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlops << ",\n";
        }
    }
    outputFile.close();
}

void autotestNaive(const double tolerance)
{
    const int lenN = 5;
    const int lenrhsWidth = 6;
    size_t Ns[lenN] = {1000, 5000, 10000, 20000, 50000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 10};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256};
    double gFlops;
    std::ofstream outputFile;
    outputFile.open("QRGFLOPsNaive.csv");
    outputFile << "N,rhsWidth,repetitions,GFLOPs,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            singleThreadTestNaive(Ns[i], rhsWidths[j], repetitions[i], tolerance, 0, gFlops);
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlops << ",\n";
        }
    }
    outputFile.close();
}

void multithreadedTest(size_t threadNumber, size_t N, size_t rhsWidth, size_t repetitions, const double tolerance)
{
    std::vector<std::thread> threads;
    std::vector<double> gFlops(threadNumber, 0.0);
    for (size_t i = 0; i < threadNumber; i++)
    {
        threads.push_back(std::thread(singleThreadTest, N, rhsWidth, repetitions, tolerance, i, std::ref(gFlops[i])));
    }

    for (size_t i = 0; i < threadNumber; i++)
    {
        threads[i].join();
    }
    double avg = average(gFlops);
    double stddev = standard_deviation(gFlops, avg);
    std::cout << "Average Performance: " << avg << " +- " << stddev << " GFLOPs" << std::endl;
}

int main(int argc, char const *argv[])
{
    size_t N = 1000;
    size_t rhsWidth = 256;
    size_t repetitions = 10;
    size_t blockSize = 2;
    const double tolerance = 1e-6;

    if (argc > 1)
    {
        if (1 == std::sscanf(argv[1], "%zu", &N))
        {
        }
        else
        {
            std::cout << "Please enter an unsigned integer!" << std::endl;
            return -1;
        }
    }
    if (argc > 2)
    {
        if (1 == std::sscanf(argv[2], "%zu", &rhsWidth))
        {
        }
        else
        {
            std::cout << "Please enter a power of 32!" << std::endl;
            return -1;
        }
    }
    if (argc > 3)
    {
        if (1 == std::sscanf(argv[3], "%zu", &repetitions))
        {
        }
        else
        {
            std::cout << "Please enter an unsigned integer!" << std::endl;
            return -1;
        }
    }
    size_t matrixSize = N * rhsWidth / 4;
    size_t matrixSizeDouble = N * rhsWidth;
    double gFlops;
    Vec4d *Q = new Vec4d[matrixSize]; // The matrix
    double *Qdouble = new double[matrixSizeDouble];
    std::cout << "Matrix Size: " << matrixSizeDouble << std::endl;

    /* 
    std::cout << "QR-algorithm (Vectorized, variable block Size): " << std::endl;
    fillMatrixRandom(Q, matrixSize);
    getGFLOPSqr(Q, qr, flopsQR, N, rhsWidth, repetitions);
    checkOrthoNormality(Q, N, rhsWidth/blockSize/4, blockSize, tolerance);
    std::cout << std::endl;

    std::cout << "QR-algorithm (Vectorized, fixed block Size): " << std::endl;
    fillMatrixRandom(Q, matrixSize);
    getGFLOPSqr(Q, qrunblocked, flopsQR, N, rhsWidth, repetitions, gFlops);
    checkOrthoNormalityFixed(Q, N, rhsWidth/8, tolerance);
    std::cout << std::endl;

    std::cout << "QR-algorithm (Vectorized, fixed block Size, Optimized): " << std::endl;
    fillMatrixRandom(Q, matrixSize);
    getGFLOPSqr(Q, qrFixedBlockOptimizedVec4d, flopsQR, N, rhsWidth, repetitions, gFlops);
    checkOrthoNormalityFixed(Q, N, rhsWidth/8, tolerance);
    std::cout << std::endl;

    std::cout << "QR-algorithm (Vectorized, fixed block Size, Optimized, Doubles): " << std::endl;
    fillMatrixRandom(Qdouble, matrixSizeDouble);
    getGFLOPSqr(Qdouble, qrFixedBlockOptimizedDouble, flopsQR, N, rhsWidth, repetitions, gFlops);
    checkOrthoNormalityFixed(Qdouble, N, rhsWidth/8, tolerance);
    std::cout << std::endl;
    */
    /*
    std::cout << "Automatic Test" << std::endl;
    autotest(tolerance);
    

    std::cout << "Multithreaded Test: " << std::endl;
    multithreadedTest(128, N, rhsWidth, repetitions, tolerance);
    */

    std::cout << "Naive test" << std::endl;
    // singleThreadTestNaive(N, rhsWidth, repetitions, tolerance, 0, gFlops);
    autotestNaive(tolerance);

    std::cout << std::endl
              << std::endl;

    std::cout << "Optimized test" << std::endl;
    //singleThreadTest(N, rhsWidth, repetitions, tolerance, 0, gFlops);
    autotest(tolerance);
    // printMatrix(Q, N, rhsWidth);
    delete[] Q;
    delete[] Qdouble;

    return 0;
}