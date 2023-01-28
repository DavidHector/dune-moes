#ifndef DUNE_MOES_MATRIXMULT_HH
#define DUNE_MOES_MATRIXMULT_HH

#include <dune/moes/vectorclass/vectorclass.h>
#include <dune/istl/bcrsmatrix.hh>
#include <iostream>

/**
 * @brief Sparse matrix multiplication, only for matrices with BlockSize = 1
 * Qout = M*Qin
 * 
 * @tparam MT Matrix type
 * @param M Matrix
 * @param Qin Multivector in
 * @param Qout Multivector out
 * @param qCols Multivector width in blocks = W/8
 * @param N Multivector length
 */
template <typename MT>
void MultQSimple(const MT &M, const std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, const size_t qCols, const size_t N)
{
    Vec4d productFirst, productSecond, inFirst, inSecond, outFirst, outSecond, entryM;
    size_t qinIndex, qoutIndex, vectorStart;
    auto endRow = M.end();
    auto endCol = (*(M.begin())).end();
    auto rowMatrix = M.begin().index();
    auto colMatrix = endCol.index();

    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        vectorStart = 8 * N * qCol;
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
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
                entryM = (*colIterator)[0][0];
                productFirst += entryM * inFirst;
                productSecond += entryM * inSecond;
            }
            productFirst.store(&Qout[qoutIndex]);
            productSecond.store(&Qout[qoutIndex + 4]);
        }
    }
}

/**
 * @brief Sparse matrix multiplication, only for matrices with BlockSize = 1
 * Qout = M*Qin
 * 
 * @tparam MT Matrix type
 * @param M Matrix
 * @param Qin Multivector in
 * @param Qout Multivector out
 * @param qCols Multivector width in blocks = W/8
 * @param N Multivector length
 */
template <typename MT>
void MultQSimple(const MT &M, const std::shared_ptr<double[]> &Qin, std::shared_ptr<double[]> &Qout, const size_t qCols, const size_t N)
{
    Vec4d productFirst, productSecond, inFirst, inSecond, outFirst, outSecond, entryM;
    size_t qinIndex, qoutIndex, vectorStart;
    auto endRow = M.end();
    auto endCol = (*(M.begin())).end();
    auto rowMatrix = M.begin().index();
    auto colMatrix = endCol.index();

    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        vectorStart = 8 * N * qCol;
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
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
                entryM = (*colIterator)[0][0];
                productFirst += entryM * inFirst;
                productSecond += entryM * inSecond;
            }
            productFirst.store(&Qout[qoutIndex]);
            productSecond.store(&Qout[qoutIndex + 4]);
        }
    }
}

// This power iteration uses same loop ordering as above
template <typename MT>
void powerIteration(const MT &M, std::shared_ptr<double[]> &Qin, std::shared_ptr<double[]> &Qout, const size_t qCols, const size_t N)
{
    Vec4d productFirst, productSecond, inFirst, inSecond, outFirst, outSecond, normFirst, normSecond, entryM;
    size_t qinIndex, qoutIndex, vectorStart;

    // set Qout to 0
    for (size_t i = 0; i < 8 * N * qCols; i++)
    {
        Qout[i] = 0.0;
    }

    auto endRow = M.end();
    auto endCol = (*(M.begin())).end();
    auto rowMatrix = M.begin().index();
    auto colMatrix = endCol.index();

    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        vectorStart = 8 * N * qCol;
        normFirst = 0.0;
        normSecond = 0.0;
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
        {
            endCol = (*rowIterator).end();
            rowMatrix = rowIterator.index(); // Matrix Block Row Index
            qoutIndex = vectorStart + 8 * rowMatrix;
            productFirst = 0.0;
            productSecond = 0.0;
            for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
            {
                auto colMatrix = colIterator.index();
                qinIndex = vectorStart + 8 * colMatrix;
                inFirst.load(&Qin[qinIndex]);
                inSecond.load(&Qin[qinIndex + 4]);
                entryM = (*colIterator)[0][0];
                productFirst += entryM * inFirst;
                productSecond += entryM * inSecond;
            }
            productFirst.store(&Qout[qoutIndex]);
            productSecond.store(&Qout[qoutIndex + 4]);
            normFirst += square(productFirst);
            normSecond += square(productSecond);
        }
        // Reduce calcs by only normalizing non-zero entries
        normFirst = sqrt(normFirst);
        normSecond = sqrt(normSecond);
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
        {
            rowMatrix = rowIterator.index();
            qoutIndex = vectorStart + 8 * rowMatrix;
            inFirst.load(&Qout[qoutIndex]);
            inSecond.load(&Qout[qoutIndex + 4]);
            outFirst = inFirst / normFirst;
            outSecond = inSecond / normSecond;
            outFirst.store(&Qout[qoutIndex]);
            outSecond.store(&Qout[qoutIndex + 4]);
        }
    }
}

template <typename MT>
void powerIteration(const MT &M, std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, const size_t qCols, const size_t N)
{
    Vec4d productFirst, productSecond, inFirst, inSecond, outFirst, outSecond, normFirst, normSecond, entryM;
    size_t qinIndex, qoutIndex, vectorStart;

    // set Qout to 0
    for (size_t i = 0; i < 8 * N * qCols; i++)
    {
        Qout[i] = 0.0;
    }

    auto endRow = M.end();
    auto endCol = (*(M.begin())).end();
    auto rowMatrix = M.begin().index();
    auto colMatrix = endCol.index();

    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        vectorStart = 8 * N * qCol;
        normFirst = 0.0;
        normSecond = 0.0;
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
        {
            endCol = (*rowIterator).end();
            rowMatrix = rowIterator.index(); // Matrix Block Row Index
            qoutIndex = vectorStart + 8 * rowMatrix;
            productFirst = 0.0;
            productSecond = 0.0;
            for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
            {
                auto colMatrix = colIterator.index();
                qinIndex = vectorStart + 8 * colMatrix;
                inFirst.load(&Qin[qinIndex]);
                inSecond.load(&Qin[qinIndex + 4]);
                entryM = (*colIterator)[0][0];
                productFirst += entryM * inFirst;
                productSecond += entryM * inSecond;
            }
            productFirst.store(&Qout[qoutIndex]);
            productSecond.store(&Qout[qoutIndex + 4]);
            normFirst += square(productFirst);
            normSecond += square(productSecond);
        }
        // Reduce calcs by only normalizing non-zero entries
        normFirst = sqrt(normFirst);
        normSecond = sqrt(normSecond);
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
        {
            rowMatrix = rowIterator.index();
            qoutIndex = vectorStart + 8 * rowMatrix;
            inFirst.load(&Qout[qoutIndex]);
            inSecond.load(&Qout[qoutIndex + 4]);
            outFirst = inFirst / normFirst;
            outSecond = inSecond / normSecond;
            outFirst.store(&Qout[qoutIndex]);
            outSecond.store(&Qout[qoutIndex + 4]);
        }
    }
}

/*
    MultQSimpleNaiveQNaive
    Matrix multiplication with double Q, but Q no longer has blocks, just columns of vectors
    This simplified Multiplication assumes that the Matrix consists of 1x1 submatrices
    No vectorization
*/
template <typename MT>
void MultQSimpleNaiveQNaive(const MT &M, const std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, size_t rhsWidth, size_t N)
{
    double product, entryM;
    size_t qinIndex, qoutIndex;
    auto endRow = M.end();
    for (size_t qCol = 0; qCol < rhsWidth; qCol++)
    {
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
        {
            auto endCol = (*rowIterator).end();
            auto rowMatrix = rowIterator.index();
            qoutIndex = N * qCol + rowMatrix;
            product = 0.0;
            for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
            {
                auto colMatrix = colIterator.index();
                qinIndex = N * qCol + colMatrix;
                entryM = (*colIterator)[0][0];
                product += entryM * Qin[qinIndex];
            }
            Qout[qoutIndex] = product;
        }
    }
}

template <typename MT>
void MultQSimpleNaiveQNaive(const MT &M, const std::shared_ptr<double[]> &Qin, std::shared_ptr<double[]> &Qout, size_t rhsWidth, size_t N)
{
    double product, entryM;
    size_t qinIndex, qoutIndex;
    auto endRow = M.end();
    for (size_t qCol = 0; qCol < rhsWidth; qCol++)
    {
        for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
        {
            auto endCol = (*rowIterator).end();
            auto rowMatrix = rowIterator.index();
            qoutIndex = N * qCol + rowMatrix;
            product = 0.0;
            for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
            {
                auto colMatrix = colIterator.index();
                qinIndex = N * qCol + colMatrix;
                entryM = (*colIterator)[0][0];
                product += entryM * Qin[qinIndex];
            }
            Qout[qoutIndex] = product;
        }
    }
}

#endif