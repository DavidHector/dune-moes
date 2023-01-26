#ifndef DUNE_MOES_INVERSEPOWERITERATION_HH
#define DUNE_MOES_INVERSEPOWERITERATION_HH

#include <dune/moes/vectorclass/vectorclass.h>
#include <dune/istl/bcrsmatrix.hh>
#include <iostream>
/**
 * @brief Applies the inverse power iteration to the problem Ax = \lambda x
 * 
 * @tparam MT The Matrix type
 * @param M the matrix
 * @param Qin 
 * @param Qout 
 * @param qCols 
 * @param N 
 */
template <typename MT>
void inversePowerIteration(const MT &M, std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, const size_t qCols, const size_t N)
{
    Vec4d productFirst, productSecond, sumFirst, sumSecond, inFirst, inSecond, outFirst, outSecond, normFirst, normSecond, entryM;
    size_t qinIndex, qoutIndex;
    std::unique_ptr<double[]> norms(new double[qCols * 8]);

    // set Qout to 0
    for (size_t i = 0; i < 8 * N * qCols; i++)
    {
        Qout[i] = 0.0;
    }
    //initialize norms
    for (size_t i = 0; i < qCols * 8; i++)
    {
        norms[i] = 0.0;
    }

    auto endRow = M.end();
    for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
    {
        auto endCol = (*rowIterator).end();
        auto rowMatrix = rowIterator.index(); // Matrix Block Row Index
        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            qoutIndex = 8 * N * qCol + rowMatrix * 8;
            productFirst = 0.0;
            productSecond = 0.0;
            for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
            {
                auto colMatrix = colIterator.index();
                qinIndex = 8 * N * qCol + 8 * colMatrix;
                inFirst.load(&Qin[qinIndex]);
                inSecond.load(&Qin[qinIndex + 4]);
                entryM = (*colIterator)[0][0];
                productFirst += entryM * inFirst;
                productSecond += entryM * inSecond;
            }
            sumFirst.load(&norms[8 * qCol]);
            sumSecond.load(&norms[8 * qCol + 4]);
            sumFirst += square(productFirst);
            sumSecond += square(productSecond);
            sumFirst.store(&norms[8 * qCol]);
            sumSecond.store(&norms[8 * qCol + 4]);
            productFirst.store(&Qout[qoutIndex]);
            productSecond.store(&Qout[qoutIndex + 4]);
        }
    }
    // re-normalize
    // I can probably save some access time, if I figure out where there are non-zero entries. But it is probably better to assume dense vectors
    qoutIndex = 0;
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        normFirst.load(&norms[8 * qCol]);
        normSecond.load(&norms[8 * qCol + 4]);
        for (size_t qRow = 0; qRow < N; qRow++)
        {
            inFirst.load(&Qout[qoutIndex]);
            inSecond.load(&Qout[qoutIndex + 4]);
            outFirst = inFirst / sqrt(normFirst);
            outSecond = inSecond / sqrt(normSecond);
            outFirst.store(&Qout[qoutIndex]);
            outSecond.store(&Qout[qoutIndex + 4]);
            qoutIndex += 8;
        }
    }
}
#endif