#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#ifdef HAVE_CONFIG_H // Whelp, this should always be included, but I didnt
#include "config.h"
#endif
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/istl/test/identity.hh>
#include <dune/istl/matrixmatrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/common/fmatrix.hh>
#include <dune/moes/MatrixMult.hh>
#include <dune/moes/moes.hh>
#include <dune/moes/Utils.hh>
#include <dune/moes/arpack_geneo_wrapper.hh>
#include <dune/moes/flopUtils.hh>
#include <dune/moes/convergenceTests.hh>

int main(int argc, char const *argv[])
{
    // TODO: Flop calculation for different matrix combinations (laplace, neumann, I guess)
    // multithreaded of the same (I guess)

    // Types
    static const int BS = 1;
    typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> MAT; // Matrix Type
    typedef Dune::BlockVector<Dune::FieldVector<double, BS>> VEC;

    std::string fileA = "neumann_problem_A_1_3.txt";
    std::string fileB = "neumann_problem_B_1_3.txt";
    std::string outFile = "neumann_problem_1_3.csv";
    std::string MToutFile = "neumann_problem_1_3_MT.csv";
    std::string genMinApproxcsnOutFile = "neumann_problem_1_3_csn.csv";
    std::string genMinLapcsnOutFile = "laplacian_identity_gen_csn.csv";
    std::string genMinLapNeucsnOutFile = "laplacian_neumann_gen_csn.csv";
    std::string genMinLapSeqTimingOutFile = "laplacian_identity_seq_gen_timing.csv";
    std::string genMinLapParTimingOutFile = "laplacian_identity_par_gen_timing.csv";

    std::cout << "Benchmarking moes.computeGenMinMagnitude() vs arpack (timing) on matrices with non-intersecting kernels..." << std::endl;
    // flopsSeqGenMinMagLap<MAT, VEC>(genMinLapSeqTimingOutFile);
    std::cout << "Complete." << std::endl;

    std::cout << "Benchmarking moes.computeGenMinMagnitude() vs arpack (timing) on matrices with non-intersecting kernels..." << std::endl;
    flopsParGenMinMagLap<MAT, VEC>(genMinLapParTimingOutFile);
    std::cout << "Complete." << std::endl;

    std::cout << "Benchmarking moes.computeGenMinMagnitudeApprox() on matrices with intersecting kernels... " << std::endl;
    // flopsSeqGenMinApproxFileRead<MAT, VEC>(fileA, fileB, outFile);
    std::cout << "Complete." << std::endl;

    std::cout << "MT benchmarking moes.computeGenMinMagnitudeApprox() on matrices with intersecting kernels... " << std::endl;
    // flopsParGenMinApproxFileRead<MAT, VEC>(fileA, fileB, MToutFile);
    std::cout << "Complete." << std::endl;

    std::cout << "Column Sum Norm vs Arpack using moes.computeGenMinMagnitudeApprox() on matrices with intersecting kernels... " << std::endl;
    // csnGenMinApprox<MAT, VEC>(fileA, fileB, genMinApproxcsnOutFile);
    std::cout << "Complete." << std::endl;

    std::cout << "Column Sum Norm vs Arpack using moes.computeGenMinMagnitude() on matrices with non-intersecting kernels (laplacian and identity)... " << std::endl;
    // csnGenMinLap<MAT, VEC>(genMinLapcsnOutFile);
    std::cout << "Complete." << std::endl;

    std::cout << "Column Sum Norm vs Arpack using moes.computeGenMinMagnitude() on matrices with non-intersecting kernels (laplacian and neumann(DLD))... " << std::endl;
    // csnGenMinLapNeu<MAT, VEC>(genMinLapNeucsnOutFile);
    std::cout << "Complete." << std::endl;
    return 0;
};
