#ifndef DUNE_MOES_HH
#define DUNE_MOES_HH

#include <dune/moes/MatrixMult.hh>
#include <dune/moes/qrcol.hh>
#include <dune/moes/vectorclass/vectorclass.h>
#include <vector>
#include <dune/moes/Utils.hh>
#include <dune/moes/umfpackMoes.hh>

/**
 * @brief Algorithms to approximate the eigenspace of a Matrix A
 * 
 * @tparam MAT Type of a DUNE-ISTL BCRSMatrix whose eigenvalues
 *                     respectively singular values shall be considered;
 *                     is assumed to have blocklevel 2.
 * @tparam VEC Type of the associated vectors; compatible with the
 *                     rows of a BCRSMatrix object (if #rows >= #ncols) or
 *                     its columns (if #rows < #ncols).
 */
template <typename MAT, typename VEC>
class moes
{
private:
    const MAT &A_;
    const size_t nIterationsMax_;
    const size_t N;
    void MultQ(const std::shared_ptr<double[]> &Qin, std::shared_ptr<double[]> &Qout, const size_t qCols) const
    {
        Vec4d productFirst, productSecond, inFirst, inSecond, entryA;
        size_t qinIndex, qoutIndex, vectorStart;
        auto endRow = A_.end();
        auto endCol = (*(A_.begin())).end();
        auto rowMatrix = A_.begin().index();
        auto colMatrix = endCol.index();

        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            vectorStart = 8 * N * qCol;
            for (auto rowIterator = A_.begin(); rowIterator != endRow; rowIterator++)
            {
                endCol = (*rowIterator).end();
                rowMatrix = rowIterator.index(); // Matrix Block Row Index
                qoutIndex = vectorStart + 8 * rowMatrix;
                productFirst = 0.0;
                productSecond = 0.0;
                for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
                {
                    colMatrix = colIterator.index();
                    qinIndex = vectorStart + 8 * colMatrix;
                    inFirst.load(&Qin[qinIndex]);
                    inSecond.load(&Qin[qinIndex + 4]);
                    entryA = (*colIterator)[0][0];
                    productFirst += entryA * inFirst;
                    productSecond += entryA * inSecond;
                }
                productFirst.store(&Qout[qoutIndex]);
                productSecond.store(&Qout[qoutIndex + 4]);
            }
        }
    }

public:
    /**
     * @brief Construct a new moes object
     * 
     * @param A Matrix whose eigenspace should be approximated
     * @param nIterationsMax Maximum number of iterations
     */
    moes(const MAT &A, const size_t nIterationsMax = 100000) : A_(A), nIterationsMax_(nIterationsMax), N(A.M()){};

    /**
     * @brief Computes the largest Eigenvalue/Eigenvector pairs for the standard eigenvalue problem Ax= \lambda x
     * 
     * @param epsilon convergence criterion |\lambda_k - \lambda_{k-1}| < epsilon
     * @param x the eigenvectors in the same format as in the multilevel_geneo
     * @param lambda eigenvalues
     * @param nev How many eigenpairs will be calculated. Only x.size() smallest will be returned
     * @param qrFrequency After how many power iteration steps should the qr algorithm be applied
     */
    inline void computeStdMaxMagnitude(const double &epsilon, std::vector<VEC> &x, std::vector<double> &lambda, int nev, int qrFrequency, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeStdMaxMagnitude(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeStdMaxMagnitude(): x and lambda must have same length!" << std::endl;
            return;
        }

        bool cb = true; //continue boolean
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        size_t it = 0;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> Qtmp(new double[matrixSize]);
        std::vector<double> lambdatmp(lambda);
        fillMatrixRandom(Q, matrixSize);
        while (cb)
        {
            powerIteration(A_, Q, Qtmp, qCols, N);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Qtmp, N, qCols, 2, 1);
            }
            powerIteration(A_, Qtmp, Q, qCols, N);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
            // Put EV stuff here
            getEigenvalues(Q, lambda, nev);
            if (largestEvDiff(lambdatmp, lambda) < epsilon || it > nIterationsMax_)
            {
                qToVEC(Q, x);
                if (checkOrthonormality)
                {
                    checkOrthoNormalityFixed(Q, N, qCols, epsilon);
                }
                cb = false;
                if (it > nIterationsMax_)
                {
                    std::cout << "Moes.computeStdMaxMagnitude reached iteration limit. Aborting." << std::endl;
                }
            }
            lambdatmp = lambda;
        }
        std::cout << "Moes.computeStdMaxMagnitude iterations: " << it << std::endl;
    };
    inline void computeGenMaxMagnitude(const MAT &B, const double &epsilon, std::vector<VEC> &x, std::vector<double> &lambda, const int &nev, const int &qrFrequency, const double &sigma, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeGenMaxMagnitude(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeGenMaxMagnitude(): x and lambda must have same length!" << std::endl;
            return;
        }

        bool cb = true; //continue boolean
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        size_t it = 0;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> AQ(new double[matrixSize]);
        MAT bshifta(B);
        try
        {
            bshifta.axpy(-sigma, A_);
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            std::cout << "moes.computeGenMaxMagnitude(): Likely cause: Different sparsity patterns for the matrices" << std::endl;
            return;
        }
        std::vector<double> lambdatmp(lambda);
        fillMatrixRandom(Q, matrixSize);
        // Make the shifted Matrix first and don't forget to recalc the eigenvalues

        auto solver = std::make_shared<Dune::UMFPackMOES<MAT>>(bshifta, false);
        while (cb)
        {
            MultQSimpleShared(A_, Q, AQ, qCols, N);
            solver->moesInversePowerIteration(AQ, Q, N, nev);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
            // Put EV stuff here
            getGenEigenvalues(bshifta, A_, Q, lambda, nev, sigma); // lambda is wrong
            if (largestEvDiff(lambdatmp, lambda) < epsilon || it > nIterationsMax_)
            {
                qToVEC(Q, x);
                if (checkOrthonormality)
                {
                    checkOrthoNormalityFixed(Q, N, qCols, epsilon);
                }
                for (size_t i = 0; i < lambda.size(); i++)
                {
                    lambda[i] = 1.0 / lambda[i];
                }
                cb = false;
                if (it > nIterationsMax_)
                {
                    std::cout << "Moes.computeGenMaxMagnitude reached iteration limit. Aborting." << std::endl;
                }
            }
            lambdatmp = lambda;
        }
        std::cout << "Moes.computeGenMaxMagnitude iterations: " << it << std::endl;
    }

    /**
     * @brief Computes the largest eigenpairs for the problem (B - \sigma B - \alpha I) x = (\lambda - \sigma) A x
     * 
     * @param B 
     * @param epsilon 
     * @param x 
     * @param lambda 
     * @param nev 
     * @param qrFrequency 
     * @param sigma 
     * @param alpha 
     */
    inline void computeGenMaxMagnitudeApprox(const MAT &B, const double &epsilon, std::vector<VEC> &x, std::vector<double> &lambda, const int &nev, const int &qrFrequency, const double &sigma, const double &alpha, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeGenMaxMagnitudeApprox(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeGenMaxMagnitudeApprox(): x and lambda must have same length!" << std::endl;
            return;
        }

        bool cb = true; //continue boolean
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        size_t it = 0;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> AQ(new double[matrixSize]);
        MAT bshifta(B);
        bshifta.axpy(-sigma, A_);
        MAT identity;
        setupIdentity(identity, N);
        bshifta.axpy(-alpha, identity);
        std::vector<double> lambdatmp(lambda);
        fillMatrixRandom(Q, matrixSize);
        // Make the shifted Matrix first and don't forget to recalc the eigenvalues

        auto solver = std::make_shared<Dune::UMFPackMOES<MAT>>(bshifta, false);
        while (cb)
        {
            MultQSimpleShared(A_, Q, AQ, qCols, N);
            solver->moesInversePowerIteration(AQ, Q, N, nev);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
            // Put EV stuff here
            getGenEigenvalues(bshifta, A_, Q, lambda, nev, sigma);
            if (largestEvDiff(lambdatmp, lambda) < epsilon || it > nIterationsMax_)
            {
                qToVEC(Q, x);
                if (checkOrthonormality)
                {
                    checkOrthoNormalityFixed(Q, N, qCols, epsilon);
                }
                for (size_t i = 0; i < lambda.size(); i++)
                {
                    lambda[i] = 1.0 / lambda[i];
                }
                cb = false;
                if (it > nIterationsMax_)
                {
                    std::cout << "Moes.computeGenMaxMagnitudeApprox reached iteration limit. Aborting." << std::endl;
                }
            }
            lambdatmp = lambda;
        }
        std::cout << "Moes.computeGenMaxMagnitudeApprox iterations: " << it << std::endl;
    };

    inline void computeStdMinMagnitude(const double &epsilon, std::vector<VEC> &x, std::vector<double> &lambda, const int &nev, const int &qrFrequency, const double &sigma, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeStdMinMagnitude(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeStdMinMagnitude(): x and lambda must have same length!" << std::endl;
            return;
        }

        bool cb = true; //continue boolean
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        size_t it = 0;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> Qtmp(new double[matrixSize]);
        MAT identity;
        setupIdentity(identity, N);
        MAT ashifted(A_);
        ashifted.axpy(-sigma, identity);
        std::vector<double> lambdatmp(lambda);
        fillMatrixRandom(Q, matrixSize);
        // Make the shifted Matrix first and don't forget to recalc the eigenvalues

        auto solver = std::make_shared<Dune::UMFPackMOES<MAT>>(ashifted, false);
        while (cb)
        {
            solver->moesInversePowerIteration(Q, Qtmp, N, nev);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Qtmp, N, qCols, 2, 1);
            }
            solver->moesInversePowerIteration(Qtmp, Q, N, nev);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
            // Put EV stuff here
            getEigenvalues(Q, lambda, nev, sigma);
            if (largestEvDiff(lambdatmp, lambda) < epsilon || it > nIterationsMax_)
            {
                qToVEC(Q, x);
                if (checkOrthonormality)
                {
                    checkOrthoNormalityFixed(Q, N, qCols, epsilon);
                }
                cb = false;
                if (it > nIterationsMax_)
                {
                    std::cout << "Moes.computeStdMinMagnitude reached iteration limit. Aborting." << std::endl;
                }
            }
            lambdatmp = lambda;
        }
        std::cout << "Moes.computeStdMinMagnitude iterations: " << it << std::endl;
    };

    inline void computeGenMinMagnitude(const MAT &B, const double &epsilon, std::vector<VEC> &x, std::vector<double> &lambda, const int &nev, const int &qrFrequency, const double &sigma, size_t &L, size_t &U, double &LUflops, size_t &iterations, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeGenMinMagnitude(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeGenMinMagnitude(): x and lambda must have same length!" << std::endl;
            return;
        }

        bool cb = true; //continue boolean
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        size_t it = 0;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> BQ(new double[matrixSize]);
        MAT ashiftb(A_);
        ashiftb.axpy(-sigma, B);
        std::vector<double> lambdatmp(lambda);
        fillMatrixRandom(Q, matrixSize);
        // Make the shifted Matrix first and don't forget to recalc the eigenvalues

        auto solver = std::make_shared<Dune::UMFPackMOES<MAT>>(ashiftb, false);
        L = solver->getNonZeroL();
        U = solver->getNonZeroU();
        LUflops = solver->getFlops();
        while (cb)
        {
            MultQSimpleShared(B, Q, BQ, qCols, N);
            solver->moesInversePowerIteration(BQ, Q, N, nev);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
            // Put EV stuff here
            getGenEigenvalues(ashiftb, B, Q, lambda, nev, sigma);
            if (largestEvDiff(lambdatmp, lambda) < epsilon || it > nIterationsMax_)
            {
                // printQ(Q, qCols);
                qToVEC(Q, x);
                if (checkOrthonormality)
                {
                    checkOrthoNormalityFixed(Q, N, qCols, epsilon);
                }
                cb = false;
                if (it > nIterationsMax_)
                {
                    std::cout << "Moes.computeGenMinMagnitude reached iteration limit. Aborting." << std::endl;
                }
            }
            lambdatmp = lambda;
        }
        iterations = it;
    };

    void printQ(const std::shared_ptr<double[]> &Q, const size_t qCols) const
    {
        size_t qIndex = 0;
        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            std::cout << "Raw Q is: " << std::endl;
            std::cout << std::endl;
            for (size_t i = 0; i < N; i++)
            {
                std::cout << Q[qIndex] << ", " << Q[qIndex + 1] << ", " << Q[qIndex + 2] << ", " << Q[qIndex + 3] << ", " << Q[qIndex + 4] << ", " << Q[qIndex + 5] << ", " << Q[qIndex + 6] << ", " << Q[qIndex + 6] << ", " << std::endl;
                qIndex += 8;
            }
            std::cout << std::endl;
        }
    }

    inline void computeGenMinMagnitudeIterations(const MAT &B, std::vector<VEC> &x, std::vector<double> &lambda, const size_t &iterations, const int &nev, const int &qrFrequency, const double &sigma, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeGenMinMagnitude(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeGenMinMagnitude(): x and lambda must have same length!" << std::endl;
            return;
        }
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> BQ(new double[matrixSize]);
        MAT ashiftb(A_);
        ashiftb.axpy(-sigma, B);
        std::vector<double> lambdatmp(lambda);
        fillMatrixRandom(Q, matrixSize);
        // Make the shifted Matrix first and don't forget to recalc the eigenvalues

        auto solver = std::make_shared<Dune::UMFPackMOES<MAT>>(ashiftb);
        for (size_t i = 0; i <= iterations; i++)
        {
            MultQSimpleShared(B, Q, BQ, qCols, N);
            solver->moesInversePowerIteration(BQ, Q, N, nev);
            if (i % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
        }
        qToVEC(Q, x);
        getGenEigenvalues(ashiftb, B, Q, lambda, nev, sigma);
    };

    /**
     * @brief Computes the smallest eigenpairs for the problem (A - \sigma B - \alpha I) x = (\lambda - \sigma) B x
     * 
     * @param B 
     * @param epsilon 
     * @param x 
     * @param lambda 
     * @param nev 
     * @param qrFrequency 
     * @param sigma 
     * @param alpha 
     */
    inline void computeGenMinMagnitudeApprox(const MAT &B, const double &epsilon, std::vector<VEC> &x, std::vector<double> &lambda, const int &nev, const int &qrFrequency, const double &sigma, const double &alpha, size_t &L, size_t &U, double &LUflops, size_t &iterations, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeGenMinMagnitudeApprox(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeGenMinMagnitudeApprox(): x and lambda must have same length!" << std::endl;
            return;
        }

        bool cb = true; //continue boolean
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        size_t it = 0;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> BQ(new double[matrixSize]);
        std::vector<double> evs(nev, 0.0);
        std::vector<double> evstmp(nev, 0.0);
        MAT ashiftb(A_);
        ashiftb.axpy(-sigma, B);
        MAT identity;
        setupIdentity(identity, N);
        ashiftb.axpy(alpha, identity);
        fillMatrixRandom(Q, matrixSize);
        // Make the shifted Matrix first and don't forget to recalc the eigenvalues

        auto solver = std::make_shared<Dune::UMFPackMOES<MAT>>(ashiftb);
        L = solver->getNonZeroL();
        U = solver->getNonZeroU();
        LUflops = solver->getFlops();
        while (cb)
        {
            MultQSimpleShared(B, Q, BQ, qCols, N);
            solver->moesInversePowerIteration(BQ, Q, N, nev);
            it++;
            if (it % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
            // Put EV stuff here
            getGenEigenvalues(ashiftb, B, Q, evs, nev, sigma);
            if (largestEvDiff(evstmp, evs) < epsilon || it > nIterationsMax_)
            {
                qToVEC(Q, x);
                for (size_t i = 0; i < lambda.size(); i++)
                {
                    lambda[i] = evs[i];
                }

                if (checkOrthonormality)
                {
                    checkOrthoNormalityFixed(Q, N, qCols, epsilon);
                }
                cb = false;
                if (it > nIterationsMax_)
                {
                    std::cout << "Moes.computeGenMinMagnitudeApprox reached iteration limit. Aborting." << std::endl;
                }
            }
            evstmp = evs;
        }

        // std::cout << "Moes.computeGenMinMagnitudeApprox iterations: " << it << std::endl;
        iterations = it;
    };

    /**
     * @brief Computes the smallest eigenpairs for the problem (A - \sigma B - \alpha I) x = (\lambda - \sigma) B x with a fixed number of iterations
     * 
     * @param B 
     * @param epsilon 
     * @param x 
     * @param lambda 
     * @param nev 
     * @param qrFrequency 
     * @param sigma 
     * @param alpha 
     */
    inline void computeGenMinMagnitudeApproxIterations(const MAT &B, std::vector<VEC> &x, std::vector<double> &lambda, const int &nev, const int &qrFrequency, const double &sigma, const double &alpha, size_t &L, size_t &U, double &LUflops, const size_t &iterations, bool checkOrthonormality = false) const
    {
        if (nev % 8 != 0)
        {
            std::cout << "moes.computeGenMinMagnitudeApprox(): Must request multiple of 8 eigenvalues!" << std::endl;
            return;
        }
        if (x.size() != lambda.size())
        {
            std::cout << "moes.computeGenMinMagnitudeApprox(): x and lambda must have same length!" << std::endl;
            return;
        }
        size_t matrixSize = N * nev;
        size_t qCols = nev / 8;
        std::shared_ptr<double[]> Q(new double[matrixSize]);
        std::shared_ptr<double[]> BQ(new double[matrixSize]);
        std::vector<double> evs(nev, 0.0);
        MAT ashiftb(A_);
        ashiftb.axpy(-sigma, B);
        MAT identity;
        setupIdentity(identity, N);
        ashiftb.axpy(alpha, identity);
        fillMatrixRandom(Q, matrixSize);
        auto solver = std::make_shared<Dune::UMFPackMOES<MAT>>(ashiftb);
        L = solver->getNonZeroL();
        U = solver->getNonZeroU();
        LUflops = solver->getFlops();
        for (size_t i = 0; i < iterations; i++)
        {
            MultQSimpleShared(B, Q, BQ, qCols, N);
            solver->moesInversePowerIteration(BQ, Q, N, nev);
            if (i % qrFrequency == 0)
            {
                qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            }
        }
        getGenEigenvalues(ashiftb, B, Q, evs, nev, sigma);
        qToVEC(Q, x);
        for (size_t i = 0; i < lambda.size(); i++)
        {
            lambda[i] = evs[i];
        }
    };

    inline void qToVEC(const std::shared_ptr<double[]> &Q, std::vector<VEC> &v) const
    {
        const size_t rhsWidth = v.size();
        size_t qIndex = 0;
        size_t vIndex;
        if (v[0].N() != N)
        {
            std::cout << "qToVEC: Dimension mismatch! Vectors need to be same length as matrix!" << std::endl;
            return;
        }

        const size_t qCols = rhsWidth / 8;
        const size_t leftover = rhsWidth % 8;
        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            vIndex = 8 * qCol;
            for (size_t i = 0; i < N; i++)
            {
                v[vIndex + 0][i][0] = Q[qIndex];
                v[vIndex + 1][i][0] = Q[qIndex + 1];
                v[vIndex + 2][i][0] = Q[qIndex + 2];
                v[vIndex + 3][i][0] = Q[qIndex + 3];
                v[vIndex + 4][i][0] = Q[qIndex + 4];
                v[vIndex + 5][i][0] = Q[qIndex + 5];
                v[vIndex + 6][i][0] = Q[qIndex + 6];
                v[vIndex + 7][i][0] = Q[qIndex + 7];
                qIndex += 8;
            }
        }
        vIndex = 8 * qCols;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < leftover; j++)
            {
                v[vIndex + j][i][0] = Q[qIndex + j];
                qIndex += 8;
            }
        }
    };

    void getBiggestDifference(const std::vector<VEC> &qa, const std::vector<VEC> &qe)
    {
        double bd = 0.0;
        double tmp = 0.0;
        size_t rhsIndex = 0;
        size_t rowIndex = 0;
        for (size_t rhs = 0; rhs < qa.size(); rhs++)
        {
            for (size_t row = 0; row < N; row++)
            {
                tmp = std::abs(qa[rhs][row][0] - qe[rhs][row][0]);
                if (tmp > bd)
                {
                    bd = tmp;
                    rhsIndex = rhs;
                    rowIndex = row;
                }
            }
        }
        std::cout << "Biggest difference: " << bd << " at rhs = " << rhsIndex << " and row = " << rowIndex << std::endl;
    }

    // Get the norm || (I - QaQaT) QeQeT||
    // This just takes too long, need to find better way (maybe the problem is also memory requirement, NxN is too much 8e12B = 8TB RAM, yeahhhh, 8e8 B= 800MB for the lowest case)
    double columnSumNorm(const std::vector<VEC> &qa, const std::vector<VEC> &qe)
    {
        if (qa[0].N() != N)
        {
            std::cout << "compareToArpack: Length of vectors in qa != matrix size N !" << std::endl;
            return 100.0;
        }
        if (qe[0].N() != N)
        {
            std::cout << "compareToArpack: Length of vectors in qe != matrix size N !" << std::endl;
            return 100.0;
        }

        // Need two NxN matrices
        std::unique_ptr<double[]> QaQaT(new double[N * N]); // store as row major or column major? (If I do Qa row major and Qe column major, the multiplication is faster)
        std::unique_ptr<double[]> QeQeT(new double[N * N]); // Column major
        double atmp;
        double etmp;
        double csn = 0.0; // column sum norm
        double csntmp;
        // Construct I-QaQaT
        for (size_t row = 0; row < N; row++)
        {
            for (size_t col = 0; col < N; col++)
            {
                atmp = 0.0;
                for (size_t k = 0; k < qa.size(); k++)
                {
                    atmp += qa[k][row][0] * qa[k][col][0];
                }
                if (row == col)
                {
                    // diagonal element
                    QaQaT[row * N + col] = 1.0 - atmp;
                }
                else
                {
                    QaQaT[row * N + col] = -1.0 * atmp;
                }
            }
        }

        // Construct QeQeT
        for (size_t row = 0; row < N; row++)
        {
            for (size_t col = 0; col < N; col++)
            {
                etmp = 0.0;
                for (size_t k = 0; k < qe.size(); k++)
                {
                    etmp += qe[k][row][0] * qe[k][col][0];
                }
                // row major storage
                QeQeT[row * N + col] = etmp; // row major storage (is the same as column major in this case)
            }
        }

        // Do the multiplication (can propably also do the norm calculation here)
        for (size_t col = 0; col < N; col++)
        {
            csntmp = 0.0;
            for (size_t row = 0; row < N; row++)
            {
                atmp = 0.0;
                for (size_t i = 0; i < N; i++)
                {
                    atmp += QaQaT[row * N + i] * QeQeT[i * N + col];
                }
                csntmp += std::abs(atmp);
            }
            if (csntmp > csn)
            {
                csn = csntmp;
            }
        }
        return csn;
    }

    // Alternative csn calc to check against
    double columnSumNormAlt(const std::vector<VEC> &qa, const std::vector<VEC> &qe)
    {
        if (qa[0].N() != qe[0].N())
        {
            std::cout << "compareToArpack: Size mismatch between qa and qe!" << std::endl;
            return 100.0;
        }
        if (qa[0].N() != N)
        {
            std::cout << "compareToArpack: Length of vectors in qa != matrix size N !" << std::endl;
            return 100.0;
        }

        // Need three NxN matrices
        std::unique_ptr<double[]> QaQaT(new double[N * N]);
        std::unique_ptr<double[]> QeQeT(new double[N * N]);
        std::unique_ptr<double[]> QaQe(new double[N * N]);

        double atmp;
        double etmp;
        double csn = 0.0; // column sum norm
        double csntmp;
        // Construct QaQaT and QeQeT
        for (size_t row = 0; row < N; row++)
        {
            for (size_t col = 0; col < N; col++)
            {
                atmp = 0.0;
                for (size_t k = 0; k < qa.size(); k++)
                {
                    atmp += qa[k][row][0] * qa[k][col][0];
                }
                QaQaT[row * N + col] = atmp;
                etmp = 0.0;
                for (size_t k = 0; k < qe.size(); k++)
                {
                    etmp += qe[k][row][0] * qe[k][col][0];
                }
                QeQeT[row * N + col] = etmp;
            }
        }
        // Matrix mult
        for (size_t row = 0; row < N; row++)
        {
            for (size_t col = 0; col < N; col++)
            {
                atmp = 0.0;
                for (size_t i = 0; i < N; i++)
                {
                    atmp += QaQaT[N * row + i] * QeQeT[N * i + col];
                }
                QaQe[row * N + col] = atmp;
            }
        }

        // Matrix subtract
        for (size_t row = 0; row < N; row++)
        {
            for (size_t col = 0; col < N; col++)
            {
                QeQeT[N * row + col] -= QaQe[row * N + col];
            }
        }

        // Norm calculation
        for (size_t col = 0; col < N; col++)
        {
            csntmp = 0.0;
            for (size_t row = 0; row < N; row++)
            {
                csntmp += std::abs(QeQeT[row * N + col]);
            }
            if (csntmp > csn)
            {
                csn = csntmp;
            }
        }
        return csn;
    }

    /**
     * @brief get the eigenvalues for the standard Eigenvalue problem Ax = \lambda x
     * 
     * @param[in] Q The multivector containing the eigenvectors
     * @param[out] lambda The eigenvalues
     * @param[in] rhsWidth The number of calculated eigenvectors 
     * @param[in] sigma The shift A - \sigma I
     */
    inline void getEigenvalues(const std::shared_ptr<double[]> &Q, std::vector<double> &lambda, const size_t rhsWidth, const double sigma = 0.0) const
    {
        size_t matrixSize = N * rhsWidth;
        size_t qCols = rhsWidth / 8;
        std::shared_ptr<double[]> Qtmp(new double[matrixSize]);
        std::unique_ptr<double[]> evs(new double[qCols * 8]);
        MAT identity;
        setupIdentity(identity, N);
        MAT ashifted(A_);
        ashifted.axpy(-sigma, identity);
        MultQSimpleShared(ashifted, Q, Qtmp, qCols, N);
        size_t qIndex = 0;
        Vec4d qNewFirst, qNewSecond, qOldFirst, qOldSecond, evFirst, evSecond, normFirst, normSecond;
        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            normFirst = 0.0;
            normSecond = 0.0;
            evFirst = 0.0;
            evSecond = 0.0;
            for (size_t row = 0; row < N; row++)
            {
                qNewFirst.load(&Qtmp[qIndex]);
                qNewSecond.load(&Qtmp[qIndex + 4]);
                qOldFirst.load(&Q[qIndex]);
                qOldSecond.load(&Q[qIndex + 4]);

                evFirst += qOldFirst * qNewFirst; // b A b
                normFirst += square(qOldFirst);   // b*b
                evSecond += qOldSecond * qNewSecond;
                normSecond += square(qOldSecond);

                qIndex += 8;
            }
            evFirst /= normFirst;
            evFirst.store(&evs[qCol * 8]);
            evSecond /= normSecond;
            evSecond.store(&evs[qCol * 8 + 4]);
        }
        for (size_t i = 0; i < lambda.size(); i++)
        {
            lambda[i] = evs[i] + sigma;
        }
    }

    /**
     * @brief Calculate eigenvalues for the generalized problem Ax = \lambda B x using Rayleigh Quotient \frac{b*Ab}{b*Bb}
     * 
     * @param B The matrix on the right hand side
     * @param Q The multivector that contains the eigenvectors
     * @param lambda The eigenvalues
     * @param rhsWidth The number of eigenvalues calculated
     * @param sigma The shift A - \sigma B
     */
    inline void getGenEigenvalues(const MAT &ashiftb, const MAT &B, const std::shared_ptr<double[]> &Q, std::vector<double> &lambda, const size_t rhsWidth, const double sigma = 0.0) const
    {
        size_t matrixSize = N * rhsWidth;
        size_t qCols = rhsWidth / 8;
        std::shared_ptr<double[]> Qtmp(new double[matrixSize]);
        std::shared_ptr<double[]> BQ(new double[matrixSize]);
        std::unique_ptr<double[]> evs(new double[qCols * 8]);
        MultQSimpleShared(ashiftb, Q, Qtmp, qCols, N); // (A - \sigma B)x
        MultQSimpleShared(B, Q, BQ, qCols, N);         // Bx
        size_t qIndex = 0;
        Vec4d qNewFirst, qNewSecond, qOldFirst, qOldSecond, evFirst, evSecond, bBbFirst, bBbSecond, qBFirst, qBSecond;
        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            bBbFirst = 0.0;
            bBbSecond = 0.0;
            evFirst = 0.0;
            evSecond = 0.0;
            for (size_t row = 0; row < N; row++)
            {
                qNewFirst.load(&Qtmp[qIndex]);
                qNewSecond.load(&Qtmp[qIndex + 4]);
                qOldFirst.load(&Q[qIndex]);
                qOldSecond.load(&Q[qIndex + 4]);
                qBFirst.load(&BQ[qIndex]);
                qBSecond.load(&BQ[qIndex + 4]);
                evFirst += qOldFirst * qNewFirst; // b A-\sigma b
                bBbFirst += qOldFirst * qBFirst;  // b*Bb
                evSecond += qOldSecond * qNewSecond;
                bBbSecond += qOldSecond * qBSecond;

                qIndex += 8;
            }
            evFirst /= bBbFirst;
            evFirst.store(&evs[qCol * 8]);
            evSecond /= bBbSecond;
            evSecond.store(&evs[qCol * 8 + 4]);
        }
        for (size_t i = 0; i < lambda.size(); i++)
        {
            lambda[i] = evs[i] + sigma;
        }
    }

    double largestEvDiff(const std::vector<double> &lambdaOld, const std::vector<double> &lambdaNew) const
    {
        double lD = 0.0; // largest Diff
        double d;
        for (size_t i = 0; i < lambdaOld.size(); i++)
        {
            d = std::abs(lambdaOld[i] - lambdaNew[i]);
            if (d > lD)
            {
                lD = d;
            }
        }
        return d;
    }
}; // class moes

void printMultivector(std::shared_ptr<double[]> &x, const size_t N, const size_t qCols, const size_t col)
{
    size_t qCol = col / 8;
    size_t intraCol = col % 8;
    size_t xIndex = 8 * N * qCol + intraCol;
    std::cout << std::endl;
    std::cout << "x = [";
    for (size_t i = 0; i < N - 1; i++)
    {
        std::cout << x[xIndex] << ", ";
        xIndex += 8;
    }
    std::cout << x[xIndex];
    std::cout << "]" << std::endl;
}

/*
    Checks, whether the sum of absolute differences between vector elements between iterations is lower than the given tolerance.
    
    Returns true if the check has passed, else returns false
*/
bool checkIterationTolerance(std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, const size_t N, const size_t qCols, const double tolerance)
{
    double difference[8];
    Vec4d dOne, dTwo;
    Vec4d QinOne, QinTwo, QoutOne, QoutTwo;
    size_t qIndex;
    for (size_t col = 0; col < qCols; col++)
    {
        // reset difference
        dOne = 0.0;
        dTwo = 0.0;
        for (size_t row = 0; row < N; row++)
        {
            qIndex = col * N * 8 + row * 8;

            // loading both blocks
            QinOne.load(&Qin[qIndex]);
            QinTwo.load(&Qin[qIndex + 4]);
            QoutOne.load(&Qout[qIndex]);
            QoutTwo.load(&Qout[qIndex + 4]);

            dOne += square(QoutOne - QinOne);
            dTwo += square(QoutTwo - QinTwo);
        }
        dOne.store(&difference[0]);
        dTwo.store(&difference[4]);
        if (horizontal_max(sqrt(dOne)) > tolerance)
        {
            return false;
        }
        if (horizontal_max(sqrt(dTwo)) > tolerance)
        {
            return false;
        }
    }
    return true;
}

template <typename T>
void pointerSwap(T **a, T **b)
{
    T *tmp = *a;
    *a = *b;
    *b = tmp;
}

// add your classes here
template <typename MT>
void largestEVs(const MT &M, std::unique_ptr<double[]> &Q, const size_t qCols, const size_t N, const double tolerance, const size_t qrFrequency)
{
    bool stop = false;
    size_t iterationCounter = 0;
    size_t matrixSize = N * qCols * 8; // watch out, overflow error might occur, 8 because of col width
    // double *Qtmp = new double[matrixSize];
    std::unique_ptr<double[]> Qtmp(new double[matrixSize]);
    fillMatrixRandom(Qtmp, matrixSize);
    // printMatrix(Qtmp, N, qCols * 8);
    fillMatrixRandom(M, Qtmp, Q, qCols, N);
    // printMatrix(Q, N, qCols * 8);
    while (!stop)
    {
        fillMatrixRandom(M, Qtmp, Q, qCols, N);
        // Why do the pointer swap, I could just do two multiplications in each step
        fillMatrixRandom(M, Q, Qtmp, qCols, N);

        // Call QR Algorithm and check tolerance
        if (iterationCounter % qrFrequency == 0)
        {
            //std::cout << "Before QR: Q[0] = " << Q[0] << std::endl;
            //printMatrix(Q, N, qCols * 8);
            qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
            //std::cout << "After QR: Q[0] = " << Q[0] << std::endl;
            // printMatrix(Q, N, qCols * 8);
            stop = checkIterationTolerance(Q, Qtmp, N, qCols, tolerance);
            if (stop)
            {
                std::cout << "largestEVs: Returning Q = " << std::endl;
                // printMatrix(Q, N, qCols * 8);
                // delete Qtmp;
                std::cout << "largestEVs took " << iterationCounter << " iterations to complete" << std::endl;
                return;
            }
        }
        iterationCounter++;
    }
}

template <typename MT>
void largestEVsIterative(const MT &M, std::unique_ptr<double[]> &Q, const size_t qCols, const size_t N, const size_t iterations, const size_t qrFrequency)
{
    bool stop = false;
    size_t matrixSize = N * qCols * 8; // watch out, overflow error might occur, 8 because of col width
    std::unique_ptr<double[]> Qtmp(new double[matrixSize]);
    size_t it = 0;
    fillMatrixRandom(Q, matrixSize);
    while (it < iterations)
    {
        powerIteration(M, Q, Qtmp, qCols, N);
        it++;
        if (it % qrFrequency == 0)
        {
            qrFixedBlockOptimizedDouble(Qtmp, N, qCols, 2, 1);
        }
        powerIteration(M, Qtmp, Q, qCols, N);
        it++;
        if (it % qrFrequency == 0)
        {
            qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
        }
    }
    return;
}

template <typename MT>
void smallestEVsIterative(const MT &M, std::shared_ptr<double[]> &Q, const size_t qCols, const size_t N, const size_t iterations, const size_t qrFrequency)
{
    bool stop = false;
    size_t matrixSize = N * qCols * 8; // watch out, overflow error might occur, 8 because of col width
    size_t rhsWidth = qCols * 8;
    std::shared_ptr<double[]> Qtmp(new double[matrixSize]);
    std::shared_ptr<double[]> xtmp(new double[matrixSize]);
    // Have to build stuff like the solver and then apply it later
    fillMatrixRandom(Qtmp, matrixSize);
    auto solver = std::make_shared<Dune::UMFPackMOES<MT>>(M, true);
    solver->moesInversePowerIteration(Qtmp, Q, xtmp, N, rhsWidth);
    for (size_t i = 0; i < iterations; i++)
    {
        solver->moesInversePowerIteration(Qtmp, Q, xtmp, N, rhsWidth); // Why is n_col different here?
        // Why do the pointer swap, I could just do two multiplications in each step
        solver->moesInversePowerIteration(Q, Qtmp, xtmp, N, rhsWidth);
        //printMultivector(Q, N, qCols, 5);

        // Call QR Algorithm and check tolerance
        if (i % qrFrequency == 0)
        {
            qrFixedBlockOptimizedDouble(Qtmp, N, qCols, 2, 1);
        }
    }
    qrFixedBlockOptimizedDouble(Q, N, qCols, 2, 1);
    return;
}

/*
    getEigenvalues
    gets EVs by using the \mu_k = \frac{b_k^\ast A b_k}{b_k^\ast b_k} Rayleigh-Quotient
*/
template <typename MT>
void getEigenvalues(const MT &M, const std::unique_ptr<double[]> &Q, const size_t qCols, const size_t N, std::vector<double> &EVs)
{
    size_t matrixSize = N * qCols * 8; // watch out, overflow error might occur, 8 because of col width
    std::unique_ptr<double[]> Qtmp(new double[matrixSize]);
    double EV = 0.0;
    double bkbk, u; // b_k^\ast b_k
    size_t col, uIndex, offset;
    MultQSimple(M, Q, Qtmp, qCols, N); // Qtmp = A b_k
    for (size_t EVIndex = 0; EVIndex < EVs.size(); EVIndex++)
    {
        col = EVIndex / 8;
        offset = EVIndex % 8;
        uIndex = col * N * 8 + offset;
        EV = 0.0;
        bkbk = 0.0;
        for (size_t qRow = 0; qRow < N; qRow++)
        {
            u = Q[uIndex];
            EV += u * Qtmp[uIndex];
            bkbk += u * u;
            uIndex += 8;
        }
        EVs[EVIndex] = EV / bkbk;
    }
}

/*
    getEigenvalues
    gets EVs by using the \mu_k = \frac{b_k^\ast A b_k}{b_k^\ast b_k} Rayleigh-Quotient
*/
template <typename MT>
void getEigenvalues(const MT &M, const std::shared_ptr<double[]> &Q, const size_t qCols, const size_t N, std::vector<double> &EVs)
{
    size_t matrixSize = N * qCols * 8; // watch out, overflow error might occur, 8 because of col width
    std::shared_ptr<double[]> Qtmp(new double[matrixSize]);
    double EV = 0.0;
    double bkbk, u; // b_k^\ast b_k
    size_t col, uIndex, offset;
    MultQSimple(M, Q, Qtmp, qCols, N); // Qtmp = A b_k
    for (size_t EVIndex = 0; EVIndex < EVs.size(); EVIndex++)
    {
        col = EVIndex / 8;
        offset = EVIndex % 8;
        uIndex = col * N * 8 + offset;
        EV = 0.0;
        bkbk = 0.0;
        for (size_t qRow = 0; qRow < N; qRow++)
        {
            u = Q[uIndex];
            EV += u * Qtmp[uIndex];
            bkbk += u * u;
            uIndex += 8;
        }
        EVs[EVIndex] = EV / bkbk;
    }
}
#endif // DUNE_MOES_HH
