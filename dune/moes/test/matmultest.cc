#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include <string>
#include <memory>

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/test/identity.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/moes/MatrixMult.hh>
#include <dune/moes/qrcol.hh>

std::mutex printMutex;

// This is not so easy to get
double flopsMatmul(const size_t N, const size_t populatedColsPerRow, const size_t &W)
{
    return N * (2 * populatedColsPerRow - 1) * W;
}

void checkEquality(Vec4d *Qold, Vec4d *Qnew, int matrixSize)
{
    Vec4ib equal;
    for (size_t i = 0; i < matrixSize; i++)
    {
        if (!horizontal_and(Qold[i] == Qnew[i]))
        {
            std::cout << "Vectors are different! Index = " << i << std::endl;
            return;
        }
    }
}

void checkEquality(std::unique_ptr<double[]> &Qold, std::unique_ptr<double[]> &Qnew, const int matrixSize)
{
    Vec4ib equal;
    for (size_t i = 0; i < matrixSize; i++)
    {
        if (Qold[i] != Qnew[i])
        {
            std::cout << "Vectors are different! Index = " << i << std::endl;
            std::cout << "Qold[i] = " << Qold[i] << ", Qnew[i] = " << Qnew[i] << std::endl;
            return;
        }
    }
    std::cout << "Successfully checked for equality!" << std::endl;
}

void printQ(Vec4d *Q, int matrixSize)
{
    std::cout << "Printing Q: " << std::endl;
    for (size_t i = 0; i < matrixSize; i++)
    {
        std::cout << "[" << Q[i][0] << "," << Q[i][1] << "," << Q[i][2] << "," << Q[i][3] << "]"
                  << "\t";
    }
    std::cout << std::endl;
}

template <typename Matmul, typename F, typename MT>
void getGFLOPSMatMul(std::unique_ptr<double[]> &Qold, std::unique_ptr<double[]> &Qnew, const MT &M, Matmul matmul, F &f, size_t N, size_t rhsWidth, size_t repetitions, double &gFlops, size_t entriesPerRow, size_t threadNumber = 0)
{
    double flops = f(N, entriesPerRow, rhsWidth); // N, 1, rhsWidth for identity matrix
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        matmul(M, Qold, Qnew, rhsWidth / 8, N);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double)duration.count() / repetitions;
    gFlops = (flops / averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    /*
    const std::lock_guard<std::mutex> lock(printMutex);
    std::cout << "Thread Number (0 if single threaded): " << threadNumber << std::endl;
    std::cout << "Average Duration: " << averageDuration / 1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N * rhsWidth * 8.0 / 1e6 << "MB" << std::endl;
    std::cout << "GFLOPS: " << gFlops << std::endl;
    */
}

template <typename MT, typename Matmul>
void singleThreadTest(const MT &M, Matmul matmul, size_t N, size_t rhsWidth, size_t repetitions, size_t threadNumber, double &gFlops, size_t entriesPerRow)
{
    size_t matrixSizeDouble = N * rhsWidth;
    std::unique_ptr<double[]> Qold(new double[matrixSizeDouble]);
    std::unique_ptr<double[]> Qnew(new double[matrixSizeDouble]);
    fillMatrixRandom(Qold, matrixSizeDouble);
    getGFLOPSMatMul(Qold, Qnew, M, matmul, flopsMatmul, N, rhsWidth, repetitions, gFlops, entriesPerRow, threadNumber);
    // checkEquality(Qold, Qnew, matrixSizeDouble);
    //delete[] Qold;
    //delete[] Qnew;
}

template <typename Matmul, typename F, typename MT>
void getGFLOPSMatMulQNaive(std::unique_ptr<double[]> &Qold, std::unique_ptr<double[]> &Qnew, const MT &M, Matmul matmul, F &f, size_t N, size_t rhsWidth, size_t repetitions, double &gFlops, size_t entriesPerRow, size_t threadNumber = 0)
{
    double flops = f(N, entriesPerRow, rhsWidth); // N, 1, rhsWidth for identity matrix
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        matmul(M, Qold, Qnew, rhsWidth, N);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    auto averageDuration = (double)duration.count() / repetitions;
    gFlops = (flops / averageDuration) / 1.0; // 1e6 when duration in ms, 1e3 when using µs
    /*
    const std::lock_guard<std::mutex> lock(printMutex);
    std::cout << "Thread Number (0 if single threaded): " << threadNumber << std::endl;
    std::cout << "Average Duration: " << averageDuration / 1e6 << "ms" << std::endl;
    std::cout << "Memory usage (full Matrix): " << N * rhsWidth * 8.0 / 1e6 << "MB" << std::endl;
    std::cout << "GFLOPS: " << gFlops << std::endl;
    */
}

template <typename MT, typename Matmul>
void singleThreadTestQNaive(const MT &M, Matmul matmul, size_t N, size_t rhsWidth, size_t repetitions, size_t threadNumber, double &gFlops, size_t entriesPerRow)
{
    size_t matrixSizeDouble = N * rhsWidth;
    std::unique_ptr<double[]> Qold(new double[matrixSizeDouble]);
    std::unique_ptr<double[]> Qnew(new double[matrixSizeDouble]);
    fillMatrixRandom(Qold, matrixSizeDouble);
    getGFLOPSMatMulQNaive(Qold, Qnew, M, matmul, flopsMatmul, N, rhsWidth, repetitions, gFlops, entriesPerRow, threadNumber);
    // checkEquality(Qold, Qnew, matrixSizeDouble);
    //delete[] Qold;
    //delete[] Qnew;
}

template <typename MT, typename Matmul>
void autotest(const MT &M, Matmul matmul, const std::string &filename, size_t entriesPerRow)
{
    const int lenN = 5;
    const int lenrhsWidth = 8;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlops;
    size_t threadNumber = 0;
    std::ofstream outputFile;
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            std::cout << "singleThreadTest with N = " << Ns[i] << ", rhsWidth = " << rhsWidths[j] << ", repetitions = " << repetitions[i] << std::endl;
            singleThreadTest(M, matmul, Ns[i], rhsWidths[j], repetitions[i], threadNumber, gFlops, entriesPerRow);
            std::cout << "Finished singleThreadTest" << std::endl;
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlops << ",\n";
        }
    }
    outputFile.close();
}

void autotestIdentity()
{
    size_t entriesPerRow = 1;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlops;
    size_t threadNumber = 0;
    std::ofstream outputFile;
    std::string filename = "matmulSimple_identity_gflops.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat identity;
            setupIdentity(identity, Ns[i] / BS);
            singleThreadTest(identity, MultQSimpleUnique<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], threadNumber, gFlops, entriesPerRow);
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlops << ",\n";
        }
    }
    outputFile.close();
}

void autotestIdentityNaive()
{
    size_t entriesPerRow = 1;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlops;
    size_t threadNumber = 0;
    std::ofstream outputFile;
    std::string filename = "matmulSimpleNaive_identity_gflops.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat identity;
            setupIdentity(identity, Ns[i] / BS);
            singleThreadTestQNaive(identity, MultQSimpleNaiveQNaive<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], threadNumber, gFlops, entriesPerRow);
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlops << ",\n";
        }
    }
    outputFile.close();
}

void autotestLaplacian()
{
    size_t entriesPerRow = 5;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlops;
    size_t threadNumber = 0;
    std::ofstream outputFile;
    std::string filename = "matmulSimple_laplacian_gflops.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat laplacian;
            setupLaplacian(laplacian, std::sqrt(Ns[i]));
            singleThreadTest(laplacian, MultQSimpleUnique<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], threadNumber, gFlops, entriesPerRow);
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlops << ",\n";
        }
    }
    outputFile.close();
}

void autotestLaplacianNaive()
{
    size_t entriesPerRow = 5;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlops;
    size_t threadNumber = 0;
    std::ofstream outputFile;
    std::string filename = "matmulSimpleNaive_laplacian_gflops.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat laplacian;
            setupLaplacian(laplacian, std::sqrt(Ns[i]));
            singleThreadTestQNaive(laplacian, MultQSimpleNaiveQNaive<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], threadNumber, gFlops, entriesPerRow);
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlops << ",\n";
        }
    }
    outputFile.close();
}

void autotestIdentityMT(const size_t threadCount)
{
    size_t entriesPerRow = 1;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlopsAvg;
    std::ofstream outputFile;
    std::string filename = "matmulSimple_identity_gflopsMT.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,threadCount\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            std::cout << "N = " << Ns[i] << ", rhsWidth = " << rhsWidths[j] << std::endl;
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat identity;
            setupIdentity(identity, Ns[i] / BS);
            std::vector<std::thread> threads;
            std::vector<double> gFlops(threadCount, 0.0);
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads.push_back(std::thread(singleThreadTest<BCRSMat, decltype(MultQSimpleUnique<BCRSMat>)>, std::ref(identity), MultQSimpleUnique<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], tN, std::ref(gFlops[tN]), entriesPerRow));
            }
            gFlopsAvg = 0.0;
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads[tN].join();
                gFlopsAvg += gFlops[tN];
            }
            gFlopsAvg /= threadCount;
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlopsAvg << "," << threadCount << ",\n";
        }
    }
    outputFile.close();
}

void autotestIdentityNaiveMT(const size_t threadCount)
{
    size_t entriesPerRow = 1;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlopsAvg;
    std::ofstream outputFile;
    std::string filename = "matmulSimpleNaive_identity_gflopsMT.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,threadCount,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            std::cout << "N = " << Ns[i] << ", rhsWidth = " << rhsWidths[j] << std::endl;
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat identity;
            setupIdentity(identity, Ns[i] / BS);
            std::vector<std::thread> threads;
            std::vector<double> gFlops(threadCount, 0.0);
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads.push_back(std::thread(singleThreadTestQNaive<BCRSMat, decltype(MultQSimpleNaiveQNaive<BCRSMat>)>, std::ref(identity), MultQSimpleNaiveQNaive<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], tN, std::ref(gFlops[tN]), entriesPerRow));
            }
            gFlopsAvg = 0.0;
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads[tN].join();
                gFlopsAvg += gFlops[tN];
            }
            gFlopsAvg /= threadCount;
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlopsAvg << "," << threadCount << ",\n";
        }
    }
    outputFile.close();
}

void autotestLaplacianMT(const size_t threadCount)
{
    size_t entriesPerRow = 5;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlopsAvg;
    std::ofstream outputFile;
    std::string filename = "matmulSimple_laplacian_gflopsMT.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,threadCount,\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            std::cout << "N = " << Ns[i] << ", rhsWidth = " << rhsWidths[j] << std::endl;
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat laplacian;
            setupLaplacian(laplacian, std::sqrt(Ns[i]));
            std::vector<std::thread> threads;
            std::vector<double> gFlops(threadCount, 0.0);
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads.push_back(std::thread(singleThreadTest<BCRSMat, decltype(MultQSimpleUnique<BCRSMat>)>, std::ref(laplacian), MultQSimpleUnique<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], tN, std::ref(gFlops[tN]), entriesPerRow));
            }
            gFlopsAvg = 0.0;
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads[tN].join();
                gFlopsAvg += gFlops[tN];
            }
            gFlopsAvg /= threadCount;
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlopsAvg << "," << threadCount << ",\n";
        }
    }
    outputFile.close();
}

void autotestLaplacianNaiveMT(const size_t threadCount)
{
    size_t entriesPerRow = 5;
    const int lenN = 5;
    const int lenrhsWidth = 8;
    const int BS = 1;
    size_t Ns[lenN] = {10000, 40000, 160000, 490000, 1000000};
    size_t repetitions[lenN] = {50, 10, 5, 1, 1};
    size_t rhsWidths[lenrhsWidth] = {8, 16, 32, 64, 128, 256, 512, 1024};
    double gFlopsAvg;
    std::ofstream outputFile;
    std::string filename = "matmulSimpleNaive_laplacian_gflopsMT.csv";
    outputFile.open(filename);
    outputFile << "N,rhsWidth,repetitions,GFLOPs,threadCount\n";
    for (size_t i = 0; i < lenN; i++)
    {
        for (size_t j = 0; j < lenrhsWidth; j++)
        {
            std::cout << "N = " << Ns[i] << ", rhsWidth = " << rhsWidths[j] << std::endl;
            // Obviously I have to build the matrix with the N here, duh
            typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
            typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
            BCRSMat laplacian;
            setupLaplacian(laplacian, std::sqrt(Ns[i]));
            std::vector<std::thread> threads;
            std::vector<double> gFlops(threadCount, 0.0);
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads.push_back(std::thread(singleThreadTestQNaive<BCRSMat, decltype(MultQSimpleNaiveQNaive<BCRSMat>)>, std::ref(laplacian), MultQSimpleNaiveQNaive<BCRSMat>, Ns[i], rhsWidths[j], repetitions[i], tN, std::ref(gFlops[tN]), entriesPerRow));
            }
            gFlopsAvg = 0.0;
            for (size_t tN = 0; tN < threadCount; tN++)
            {
                threads[tN].join();
                gFlopsAvg += gFlops[tN];
            }
            gFlopsAvg /= threadCount;
            outputFile << Ns[i] << "," << rhsWidths[j] << "," << repetitions[i] << "," << gFlopsAvg << "," << threadCount << ",\n";
        }
    }
    outputFile.close();
}

int main(int argc, char const *argv[])
{
    // Make a Block Matrix
    size_t N = 1e6; //
    double Nd = 1e4;
    size_t rhsWidth = 256;
    double rhsWidthD = 256;
    size_t matrixSize = N * rhsWidth; // / 4;
    size_t repetitions = 10;
    size_t threadNumber = 0;
    size_t identityEPR = 1; // Entries per row
    size_t laplacianEPR = 5;
    // double flops = Nd * 4 * rhsWidthD * 2.0 * repetitions; // N matrix blocks, 16 entries per block, 2 operations, rhsWidth vectors
    static const int BS = 1;
    typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
    BCRSMat identity;
    setupIdentity(identity, N / BS);

    BCRSMat laplacian;
    setupLaplacian(laplacian, std::sqrt(N));

    double gFlops;

    /*
    std::cout << "Matrix Multiplication MultQSimple, singleThreadTest, identity" << std::endl;
    singleThreadTest(identity, MultQSimple<BCRSMat>, N, rhsWidth, repetitions, threadNumber, gFlops, identityEPR);

    std::cout << std::endl
              << "Matrix Multiplication MultQSimpleNaive, singleThreadTest, identity" << std::endl;
    singleThreadTest(identity, MultQSimpleNaive<BCRSMat>, N, rhsWidth, repetitions, threadNumber, gFlops, identityEPR);

    std::cout << std::endl
              << "Matrix Multiplication MultQSimple, singleThreadTest, laplacian" << std::endl;
    singleThreadTest(laplacian, MultQSimple<BCRSMat>, N, rhsWidth, repetitions, threadNumber, gFlops, laplacianEPR);

    std::cout << std::endl
              << "Matrix Multiplication MultQSimpleNaive, singleThreadTest, laplacian" << std::endl;
    singleThreadTest(laplacian, MultQSimpleNaive<BCRSMat>, N, rhsWidth, repetitions, threadNumber, gFlops, laplacianEPR);
    */

    // const std::string filename("matmulSimple_identity_gflops.csv");
    //autotest(identity, MultQSimple<BCRSMat>, filename, identityEPR);

    /*
    std::cout << "Optimized Multiplication (identity): " << std::endl;
    autotestIdentity();
    std::cout << std::endl
              << std::endl
              << "Naive Multiplication (identity): " << std::endl;
    autotestIdentityNaive();
    std::cout << std::endl
              << std::endl
              << "Optimized Multiplication (laplacian): " << std::endl;
    autotestLaplacian();
    std::cout << std::endl
              << std::endl
              << "Naive Multiplication (laplacian): " << std::endl;
    autotestLaplacianNaive();
    */
    const size_t threadCount = 128;
    std::cout << "Optimized Multiplication (identity, MT): " << std::endl;
    autotestIdentityMT(threadCount);
    std::cout << std::endl
              << std::endl
              << "Naive Multiplication (identity, MT): " << std::endl;
    autotestIdentityNaiveMT(threadCount);
    std::cout << std::endl
              << std::endl
              << "Optimized Multiplication (laplacian, MT): " << std::endl;
    autotestLaplacianMT(threadCount);
    std::cout << std::endl
              << std::endl
              << "Naive Multiplication (laplacian, MT): " << std::endl;
    autotestLaplacianNaiveMT(threadCount);
    return 0;
}
