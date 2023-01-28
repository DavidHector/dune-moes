#include <iostream>
#include <vector>
#ifdef HAVE_CONFIG_H // Whelp, this should always be included, but I didnt
#include "config.h"
#endif
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/matrixmarket.hh>
#include <dune/common/fmatrix.hh>
#include <dune/moes/MatrixMult.hh>
#include <dune/moes/moes.hh>
#include <dune/moes/Utils.hh>
#include <dune/moes/arpack_geneo_wrapper.hh>

int main(int argc, char const *argv[])
{
    const double tolerance = 1e-8;
    size_t EVNumber = 8;
    size_t L, U, iterations;
    double LUflops;
    double shift = 0.01;
    double idshift = 0.1;
    static const int BS = 1;
    typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat; // Matrix Type
    typedef Dune::BlockVector<Dune::FieldVector<double, BS>> VEC;
    BCRSMat A;
    BCRSMat B;
    BCRSMat identity;
    std::string fileA = "neumann_problem_A_1_3.txt";
    std::string fileB = "neumann_problem_B_1_3.txt";
    Dune::loadMatrixMarket(A, fileA);
    Dune::loadMatrixMarket(B, fileB);
    BCRSMat Aslightshift(A);
    setupIdentity(identity, A.N());
    BCRSMat identitynegfour(identity);
    identitynegfour.axpy(-5.0, identity);
    Aslightshift.axpy(idshift, identity);
    ArpackMLGeneo::ArPackPlusPlus_Algorithms<BCRSMat, VEC> arpack(Aslightshift);
    moes<BCRSMat, VEC> moestest(A);
    moes<BCRSMat, VEC> moesid(identitynegfour);
    VEC vec(A.N()); //for template
    vec = 0.0;      //just copying from multilevel_geneo_preconditioner.hh
    std::vector<VEC> eigenvecs(EVNumber, vec);
    std::vector<VEC> moeseigenvecs(EVNumber, vec);
    std::vector<double> eigenvals(EVNumber, 0.0);
    std::vector<double> moeseigenvals(EVNumber, 0.0);
    moestest.computeGenMinMagnitudeApprox(B, tolerance, moeseigenvecs, moeseigenvals, EVNumber, 1, shift, idshift, L, U, LUflops, iterations);
    // moesid.computeStdMinMagnitude(tolerance, moeseigenvecs, moeseigenvals, EVNumber, 1, 0.0);
    // moestest.computeStdMinMagnitude(tolerance, moeseigenvecs, moeseigenvals, EVNumber, 2, 82000.0);
    // moestest.computeGenMinMagnitude(B, tolerance, moeseigenvecs, moeseigenvals, EVNumber, 1, 1.5);
    // moestest.computeStdMaxMagnitude(tolerance, moeseigenvecs, moeseigenvals, EVNumber, 1);
    // moestest.computeGenMaxMagnitude(B, tolerance, moeseigenvecs, moeseigenvals, EVNumber, 1, 0.0);
    // Should do the same as smallest EVs Iterative
    // arpack.computeStdNonSymMinMagnitude(B, tolerance, eigenvecs, eigenvals, -0.5);
    arpack.computeGenSymShiftInvertMinMagnitude(B, tolerance, eigenvecs, eigenvals, shift);
    double csN = moestest.columnSumNorm(moeseigenvecs, eigenvecs);
    std::cout << "moes took " << iterations << " iterations."
              << std::endl
              << "The smallest " << EVNumber << " moes Eigenvalues are: " << std::endl;
    for (size_t i = 0; i < EVNumber; i++)
    {
        std::cout << moeseigenvals[i] << std::endl;
    }

    std::cout << std::endl
              << "The smallest " << EVNumber << " arpack Eigenvalues are: " << std::endl;
    for (size_t i = 0; i < EVNumber; i++)
    {
        std::cout << eigenvals[i] << std::endl;
    }
    std::cout << "The column sum norm is: " << csN << std::endl;

    return 0;
}
