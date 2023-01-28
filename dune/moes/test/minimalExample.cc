#include <iostream>
#include <vector>
#ifdef HAVE_CONFIG_H // Always include this
#include "config.h"
#endif
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/moes/moes.hh>
#include <dune/moes/Utils.hh>

int main(int argc, char const *argv[])
{
    const size_t N = 100;
    const size_t evs = 8;
    const size_t GSfreq = 1;
    const double sigma = 0.01;
    const double tolerance = 1e-8;
    static const int BS = 1;
    typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat; // Matrix Type
    typedef Dune::BlockVector<Dune::FieldVector<double, BS>> VEC;
    BCRSMat laplacian;
    BCRSMat identity;
    setupLaplacian(laplacian, std::sqrt(N));
    setupIdentity(identity, N);
    moes<BCRSMat, VEC> moes(laplacian);
    VEC vec(N);
    vec = 0.0;
    std::vector<VEC> eigenvecs(evs, vec);
    std::vector<double> eigenvals(evs, 0.0);
    size_t L, U, iterations;
    double LUflops;
    moes.computeGenMinMagnitude(identity, tolerance, eigenvecs, eigenvals, evs, GSfreq, sigma, L, U, LUflops, iterations);
    std::cout << "The smallest 8 eigenvalues of the Laplacian are: " << std::endl;
    for (auto &&i : eigenvals)
    {
        std::cout << i << std::endl;
    }
    return 0;
}
