#include <iostream>
#include <vector>
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

int main(int argc, char const *argv[])
{
    size_t N = 10000;
    size_t rhsWidth = 256;
    size_t repetitions = 10;
    const double tolerance = 1e-10;

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
    size_t Qsize = N * rhsWidth;
    size_t qCols = rhsWidth / 8;
    size_t EVNumber = rhsWidth;
    double shift = 0.0;
    std::unique_ptr<double[]> Q(new double[Qsize]);
    std::shared_ptr<double[]> Qs(new double[Qsize]);
    std::vector<double> EVs(EVNumber, 0.0);
    static const int BS = 1;
    typedef Dune::FieldMatrix<double, BS, BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat; // Matrix Type
    typedef Dune::BlockVector<Dune::FieldVector<double, BS>> VEC;
    BCRSMat laplacian;
    BCRSMat identity;
    BCRSMat identitylap;
    BCRSMat B;
    BCRSMat neumann;
    setupLaplacian(B, std::sqrt(N));
    setupLaplacianWithBoundary(laplacian, std::sqrt(N));
    setupLaplacianWithoutBoundary(neumann, std::sqrt(N));
    setupIdentity(identity, N);
    /*
    std::cout << "Laplacian with naive boundary: " << std::endl;
    printBCRS(B);
    std::cout << "Laplacian with boundary: " << std::endl;
    printBCRS(laplacian);
    std::cout << "Neumann: " << std::endl;
    printBCRS(neumann);
    */
    // setupLaplacian(laplacian, std::sqrt(N)); //AAAAAArgghh, this sets the matrix to size N*N x N*N (with N*N*5 entries, not really sure what the BlockSize does)
    // setupIdentity(neumann, N);
    // setupIdentity(identity, N);
    /*
    std::string slap = "toMatlabWriter.txt";
    Dune::writeMatrixToMatlab(laplacian, slap);
    std::string slapM = "toMatrixMarket.txt";
    Dune::storeMatrixMarket(laplacian, slapM);
    setupIdentityWithLaplacianSparsityPattern(identitylap, N);
    */
    // neumann[0][0] = 0.0;
    // neumann[N - 1][N - 1] = 0.0;
    // transposeMatMultMat(B, neumann, laplacian);
    // matMultMat(B, laplacian, neumann);
    ArpackMLGeneo::ArPackPlusPlus_Algorithms<BCRSMat, VEC> arpack(laplacian);
    moes<BCRSMat, VEC> moes(laplacian);
    // ArpackMLGeneo::ArPackPlusPlus_Algorithms<BCRSMat, VEC> arpack(identity);
    // moes<BCRSMat, VEC> moes(identity);
    // How to produce the right hand-side?
    VEC vec(N); //for template
    vec = 0.0;  //just copying from multilevel_geneo_preconditioner.hh
    std::vector<VEC> eigenvecs(EVNumber + 8, vec);
    std::vector<VEC> moeseigenvecs(EVNumber + 8, vec);
    std::vector<double> eigenvals(EVNumber + 8, 0.0);
    std::vector<double> moeseigenvals(EVNumber + 8, 0.0);
    std::cout << "init of arpack was successfull" << std::endl
              << "starting arpack computeGenSymShiftInvertMinMagnitude: " << std::endl;
    // arpack.computeGenSymShiftInvertMinMagnitude(B, tolerance, eigenvecs, eigenvals, shift);
    // can get matrix rows via const int nrows/ncols = A.nrows()/A.ncols()
    // The shift happens inside here with the MAT ashiftb(A) ashiftb.axpy(-sigma, b_) function
    // arpack.computeGenSymShiftInvertMinMagnitudeAdaptive(B,tolerance,eigenvecs,eigenvals,shift,eigenvalue_fine_threshold_,n_eigenvectors_fine_used_);

    // How to get B?
    // moes.computeStdMaxMagnitude(tolerance, moeseigenvecs, eigenvals, EVNumber, 2);
    /*
    moes.computeGenMaxMagnitude(identitylap, tolerance, moeseigenvecs, eigenvals, EVNumber * 2, 10, -0.5);
    largestEVsIterative(laplacian, Q, qCols, N, 10000, 1);
    getEigenvalues(laplacian, Q, qCols, N, EVs);
    std::cout << "The largest " << EVNumber << " Eigenvalues are: " << std::endl;
    for (size_t i = 0; i < EVNumber; i++)
    {
        std::cout << EVs[i] << std::endl;
    }

    std::cout << "The largest " << EVNumber << " moes Eigenvalues are: " << std::endl;
    for (size_t i = 0; i < EVNumber; i++)
    {
        std::cout << eigenvals[i] << std::endl;
    }
    */
    // moes.computeStdMinMagnitude(tolerance, moeseigenvecs, moeseigenvals, EVNumber, 2, -0.5); // Gotta use a negative shift, otherwise weird things happen
    size_t L, U, iterations;
    double LUflops;
    // moes.computeGenMinMagnitudeApprox(neumann, tolerance, moeseigenvecs, moeseigenvals, EVNumber + 8, 1, -1.1, 0.001, L, U, LUflops, iterations);
    moes.computeGenMinMagnitude(neumann, tolerance, moeseigenvecs, moeseigenvals, EVNumber + 8, 1, -0.05, L, U, LUflops, iterations);
    //moes.computeGenMinMagnitudeIterations(neumann, moeseigenvecs, moeseigenvals, 180, EVNumber + 8, 1, -0.05);
    std::cout << "Iterations: " << iterations << std::endl;
    // Should do the same as smallest EVs Iterative
    //arpack.computeGenSymShiftInvertMinMagnitude(neumann, tolerance, eigenvecs, eigenvals, -0.05);
    arpack.computeGenNonSymShiftInvertMinMagnitude(neumann, tolerance, eigenvecs, eigenvals, -0.05);
    // arpack.computeStdNonSymMinMagnitude(neumann, tolerance, eigenvecs, eigenvals, -0.5);
    std::vector<VEC> redeigenvecs(EVNumber, vec);
    for (size_t i = 0; i < redeigenvecs.size(); i++)
    {
        redeigenvecs[i] = eigenvecs[i];
    }
    std::vector<VEC> redmoeseigenvecs(moeseigenvecs.size(), vec);
    for (size_t i = 0; i < redmoeseigenvecs.size(); i++)
    {
        redmoeseigenvecs[i] = moeseigenvecs[i];
    }
    // double csN = moes.columnSumNorm(redmoeseigenvecs, redeigenvecs);
    // double csN = moes.columnSumNorm(moeseigenvecs, eigenvecs);

    /*
    for (size_t i = 0; i < eigenvecs.size(); i++)
    {
        normalizeVec(eigenvecs[i]);
    }
    */

    double csN = moes.columnSumNorm(moeseigenvecs, redeigenvecs);
    double csNAlt = moes.columnSumNormAlt(moeseigenvecs, redeigenvecs);
    std::cout << std::endl
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

    std::cout << std::endl
              << "The smallest " << EVNumber << " moes Eigenvectors are: " << std::endl;
    for (size_t i = 0; i < EVNumber; i++)
    {
        std::cout << printNormVec(moeseigenvecs[i]) << std::endl;
    }

    std::cout << std::endl
              << "The smallest " << EVNumber << " arpack Eigenvecs are: " << std::endl;
    for (size_t i = 0; i < EVNumber; i++)
    {
        std::cout << printNormVec(eigenvecs[i]) << std::endl;
    }

    std::cout << "The column sum norm is: " << csN << std::endl; // What da fuck, ich kann nicht die exakt gleichen Vektoren haben und was anderes raus kriegen.
    std::cout << "The csNAlt is: " << csNAlt << std::endl;
    return 0;
}
