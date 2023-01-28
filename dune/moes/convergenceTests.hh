#ifndef DUNE_MOES_CONVERGENCETESTS_HH
#define DUNE_MOES_CONVERGENCETESTS_HH

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
#include <dune/istl/matrixmarket.hh>
#include <dune/common/fmatrix.hh>
#include <dune/moes/MatrixMult.hh>
#include <dune/moes/moes.hh>
#include <dune/moes/Utils.hh>
#include <dune/moes/arpack_geneo_wrapper.hh>

template <typename MAT, typename VEC>
void csnGenMinApprox(const std::string Afile, const std::string Bfile, const std::string filenameOut, const double tolerance = 1e-8, const double sigma = 0.01, const double alpha = 0.1, const size_t qrFrequency = 1)
{
    MAT A, B;
    Dune::loadMatrixMarket(A, Afile);
    Dune::loadMatrixMarket(B, Bfile);
    const size_t lenIterations = 14;
    const size_t lenrhsWidths = 5;
    const size_t rhsWidths[lenrhsWidths] = {8, 16, 32, 40, 64};
    size_t iterations[lenIterations] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    VEC vec(A.N());
    vec = 0.0;
    double LUflops, csn;
    size_t L, U;
    moes<MAT, VEC> csnMoes(A);
    MAT identity;
    MAT Aslightshift(A);
    setupIdentity(identity, A.N());
    Aslightshift.axpy(alpha, identity);
    ArpackMLGeneo::ArPackPlusPlus_Algorithms<MAT, VEC> arpack(Aslightshift);

    std::ofstream outputFile;
    outputFile.open(filenameOut);
    outputFile << "rhsWidth, iterations, columnSumNorm, arpackTolerance, ";
    for (size_t irhs = 0; irhs < lenrhsWidths; irhs++)
    {
        std::vector<VEC> evs(rhsWidths[irhs] + 8, vec);
        std::vector<VEC> arpacksevs(rhsWidths[irhs], vec);
        std::vector<double> lambdas(rhsWidths[irhs] + 8, 0.0);
        std::vector<double> arlambdas(rhsWidths[irhs], 0.0);
        arpack.computeGenNonSymShiftInvertMinMagnitude(B, tolerance, arpacksevs, arlambdas, sigma);
        for (size_t i = 0; i < lenIterations; i++)
        {
            csnMoes.computeGenMinMagnitudeApproxIterations(B, evs, lambdas, rhsWidths[irhs] + 8, qrFrequency, sigma, alpha, L, U, LUflops, iterations[i]);
            csn = csnMoes.columnSumNorm(evs, arpacksevs);
            outputFile << "\n"
                       << rhsWidths[irhs] << "," << iterations[i] << "," << csn << "," << tolerance << ",";
        }
    }
    outputFile.close();
}

template <typename VEC>
void reducevectorlength(const std::vector<VEC> &vold, std::vector<VEC> &vnew)
{
    for (size_t i = 0; i < vnew.size(); i++)
    {
        vnew[i] = vold[i];
    }
}

template <typename MAT, typename VEC>
void csnGenMinLap(const std::string filenameOut, const double tolerance = 1e-8, const double sigma = -0.5, const size_t qrFrequency = 1)
{
    const size_t lenNs = 5;
    size_t Ns[lenNs] = {144, 400, 900, 1600, 2500}; // Must be square numbers
    const size_t lenIterations = 13;
    size_t iterations[lenIterations] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const size_t lenrhsWidths = 4;
    size_t rhsWidths[lenrhsWidths] = {8, 16, 32, 64}; // Must be multiples of 8
    double csn, csnAlt;
    std::ofstream outputFile;
    outputFile.open(filenameOut);
    outputFile << "N, rhsWidth, iterations, columnSumNorm, arpackTolerance, ";
    for (size_t iN = 0; iN < lenNs; iN++)
    {
        MAT A, B;
        // setupLaplacianWithBoundary(A, std::sqrt(Ns[iN]));
        setupLaplacian(A, std::sqrt(Ns[iN]));
        // setupLaplacianWithoutBoundary(B, std::sqrt(Ns[iN]));
        setupIdentity(B, Ns[iN]);
        VEC vec(Ns[iN]);
        vec = 0.0;
        ArpackMLGeneo::ArPackPlusPlus_Algorithms<MAT, VEC> arpack(A);
        moes<MAT, VEC> csnMoes(A);
        for (size_t irhsW = 0; irhsW < lenrhsWidths; irhsW++)
        {
            std::vector<VEC> moesevs(rhsWidths[irhsW] + 8, vec); // moesevs must be a superset of arevs
            std::vector<VEC> arevs(rhsWidths[irhsW], vec);
            std::vector<double> moeslambdas(rhsWidths[irhsW] + 8, 0.0);
            std::vector<double> arlambdas(rhsWidths[irhsW], 0.0);
            arpack.computeGenNonSymShiftInvertMinMagnitude(B, tolerance, arevs, arlambdas, sigma);
            // arpack.computeGenSymShiftInvertMinMagnitude(B, tolerance, arevs, arlambdas, sigma);
            // int arpackits = arpack.getIterationCount(); // Get Arpack iterations
            std::cout << "rhsWidth = " << rhsWidths[irhsW] << std::endl;
            for (size_t iIts = 0; iIts < lenIterations; iIts++)
            {
                csnMoes.computeGenMinMagnitudeIterations(B, moesevs, moeslambdas, iterations[iIts], rhsWidths[irhsW] + 8, qrFrequency, sigma);
                // could also do flop calculation here
                csn = csnMoes.columnSumNorm(moesevs, arevs);
                // csnAlt = csnMoes.columnSumNormAlt(moesevs, arevs);
                std::cout << "csn = " << csn << ", csnAlt = " << csnAlt << std::endl;
                outputFile << "\n"
                           << Ns[iN] << "," << rhsWidths[irhsW] << "," << iterations[iIts] << "," << csn << "," << tolerance << ",";
            }
        }
        std::cout << "N = " << Ns[iN] << " . Finished." << std::endl;
    }
    outputFile.close();
}

template <typename MAT, typename VEC>
void csnGenMinLapNeu(const std::string filenameOut, const double tolerance = 1e-8, const double sigma = -0.05, const size_t qrFrequency = 1)
{
    const size_t lenNs = 5;
    size_t Ns[lenNs] = {144, 400, 900, 1600, 2500}; // Must be square numbers
    const size_t lenIterations = 13;
    size_t iterations[lenIterations] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const size_t lenrhsWidths = 4;
    size_t rhsWidths[lenrhsWidths] = {8, 16, 32, 64}; // Must be multiples of 8
    double csn, csnAlt;
    std::ofstream outputFile;
    outputFile.open(filenameOut);
    outputFile << "N, rhsWidth, iterations, columnSumNorm, arpackTolerance, ";
    for (size_t iN = 0; iN < lenNs; iN++)
    {
        MAT A, B;
        // setupLaplacianWithBoundary(A, std::sqrt(Ns[iN]));
        setupLaplacianWithBoundary(A, std::sqrt(Ns[iN]));
        // setupLaplacianWithoutBoundary(B, std::sqrt(Ns[iN]));
        setupLaplacianWithoutBoundary(B, std::sqrt(Ns[iN]));
        VEC vec(Ns[iN]);
        vec = 0.0;
        ArpackMLGeneo::ArPackPlusPlus_Algorithms<MAT, VEC> arpack(A);
        moes<MAT, VEC> csnMoes(A);
        for (size_t irhsW = 0; irhsW < lenrhsWidths; irhsW++)
        {
            std::vector<VEC> moesevs(rhsWidths[irhsW] + 8, vec); // moesevs must be a superset of arevs
            std::vector<VEC> arevs(rhsWidths[irhsW], vec);
            std::vector<double> moeslambdas(rhsWidths[irhsW] + 8, 0.0);
            std::vector<double> arlambdas(rhsWidths[irhsW], 0.0);
            arpack.computeGenNonSymShiftInvertMinMagnitude(B, tolerance, arevs, arlambdas, sigma);
            /*
            for (size_t i = 0; i < arevs.size(); i++)
            {
                normalizeVec(arevs[i]);
            }
            */
            // arpack.computeGenSymShiftInvertMinMagnitude(B, tolerance, arevs, arlambdas, sigma);
            // int arpackits = arpack.getIterationCount(); // Get Arpack iterations
            std::cout << "rhsWidth = " << rhsWidths[irhsW] << std::endl;
            for (size_t iIts = 0; iIts < lenIterations; iIts++)
            {
                csnMoes.computeGenMinMagnitudeIterations(B, moesevs, moeslambdas, iterations[iIts], rhsWidths[irhsW] + 8, qrFrequency, sigma);
                // could also do flop calculation here
                csn = csnMoes.columnSumNorm(moesevs, arevs);
                // csnAlt = csnMoes.columnSumNormAlt(moesevs, arevs);
                std::cout << "csn = " << csn << ", csnAlt = " << csnAlt << std::endl;
                outputFile << "\n"
                           << Ns[iN] << "," << rhsWidths[irhsW] << "," << iterations[iIts] << "," << csn << "," << tolerance << ",";
            }
        }
        std::cout << "N = " << Ns[iN] << " . Finished." << std::endl;
    }
    outputFile.close();
}
#endif // convergenceTests