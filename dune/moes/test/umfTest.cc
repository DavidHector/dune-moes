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
#include <dune/moes/umfpackMoes.hh>
#include <dune/moes/qrcol.hh>

int main(int argc, char const *argv[])
{

    static const int BS = 1;
    size_t N = 100;
    size_t rhsWidth = 16;
    typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;
    using UMFPACKSOLVER = Dune::UMFPackMOES<BCRSMat>;
    BCRSMat laplacian;
    setupLaplacian(laplacian, std::sqrt(N));
    size_t matrixSizeDouble = N * rhsWidth;
    std::shared_ptr<double[]> Qold(new double[matrixSizeDouble]);
    std::shared_ptr<double[]> Qnew(new double[matrixSizeDouble]);
    fillMatrixRandom(Qold, matrixSizeDouble);
    auto solver = std::make_shared<UMFPACKSOLVER>(laplacian, true);
    solver->moesInversePowerIteration(Qold, Qnew, N, rhsWidth);
    std::cout << "Finished Test" << std::endl;
    return 0;
}
