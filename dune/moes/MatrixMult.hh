#ifndef DUNE_MOES_MATRIXMULT_HH
#define DUNE_MOES_MATRIXMULT_HH

#include <dune/moes/vectorclass/vectorclass.h>
#include <dune/istl/bcrsmatrix.hh>
#include <iostream>

/*
    input: M: Matrix to calculate EVs for, Q: 

*/
template <typename MT>
void MultQ(MT &M, Vec4d *Qin, Vec4d *Qout, size_t qCols, size_t N)
{
    // How to iterate over M?
    // So I think for a BCRS Matrix, this only iterates over the blocks (which are fieldmatrices I guess)
    // Set Qout to zero
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        for (size_t qRow = 0; qRow < 2 * N; qRow++)
        {
            Qout[qCol * 2 * N + qRow] = 0.0;
        }
    }

    int rows = 0;
    auto endRow = M.end();
    for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
    {
        auto endCol = (*rowIterator).end();
        auto mBrI = rowIterator.index(); // Matrix Block Row Index
        rows++;
        int cols = 0;
        for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
        {
            cols++;
            // *colIterator is probably the FieldMatrix (which uses the dense matrix multAssign)
            auto mBcI = colIterator.index();
            auto fMatRows = (*colIterator).rows; // Should be blockSize of the underlying field matrix
            auto fMatCols = (*colIterator).cols;
            for (auto i = 0; i < fMatRows; i++)
            {
                auto mIr = mBrI * fMatRows + i;
                for (size_t qCol = 0; qCol < qCols; qCol++)
                {
                    for (auto j = 0; j < fMatCols; j++)
                    {
                        auto mIc = mBcI * fMatCols + j;
                        auto qinIndex = qCol * 2 * N + mIc * 2;  // columns in the matrix are rows in the vector
                        auto qoutIndex = qCol * 2 * N + mIr * 2; //col row
                        Qout[qoutIndex] += (*colIterator)[i][j] * Qin[qinIndex];
                        Qout[qoutIndex + 1] += (*colIterator)[i][j] * Qin[qinIndex + 1];
                        // Problem: Q has BlockSize 2, so at every qrow (i.e. mIr/mIc, I have to also look at the other block)
                        // basically what I want is:
                        // Qout[qCol][mIr] += M[mBrI][mBcI][i][j] * Qin[qCol][mIc];
                        // reorder the loop to avoid striding

                        // (*colIterator)[i][j] should be the actual entry, while rowIterator.index() amd colIterator.index() are the blockIndices of the bcrsmatrix
                        // How do I do the multiplication?
                        // We are going going through the columns, but are always fMatRows wide. I have to multiply it with everything anyway, so maybe the rows dont matter
                        // What to do about the fact that the matrix has doubles as entries. But Q has Vec4ds
                    }
                }
            }
        }
    }
}
/*
    Matrix multiplication with double Q 
    WTF this shouldnt even work, we only store the matrix multiplication of the last BlockMatrix, nothing here makes sense
*/

template <typename MT>
void MultQ(MT &M, const double *Qin, double *Qout, const size_t qCols, const size_t N)
{
    Vec4d v, u;
    Vec4d zeroVec = 0.0;
    Vec4d productFirst, productSecond;
    Vec4d inFirst, inSecond;
    Vec4d outFirst, outSecond;
    Vec4d entryM;
    size_t mIr, mBcI, mBrI, mIc, qinIndex, qoutIndex, fMatRows, fMatCols;
    size_t qColNEight;
    // How to iterate over M?
    // So I think for a BCRS Matrix, this only iterates over the blocks (which are fieldmatrices I guess)
    // Set Qout to zero
    for (size_t i = 0; i < qCols * N * 8; i += 4)
    {
        zeroVec.store(&Qout[i]);
    }

    int rows = 0;
    auto endRow = M.end();
    for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
    {
        auto endCol = (*rowIterator).end();
        auto mBrI = rowIterator.index(); // Matrix Block Row Index
        rows++;
        int cols = 0;
        for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
        {
            cols++;
            // *colIterator is probably the FieldMatrix (which uses the dense matrix multAssign)
            mBcI = colIterator.index();     // Matrix Block Column Index
            fMatRows = (*colIterator).rows; // Should be blockSize of the underlying field matrix
            fMatCols = (*colIterator).cols;
            // std::cout << "fMatRows = " << fMatRows << ", fMatCols = " << fMatCols << std::endl;
            for (auto i = 0; i < fMatRows; i++)
            {
                mIr = mBrI * fMatRows + i; // Matrix Row Index
                for (size_t qCol = 0; qCol < qCols; qCol++)
                {
                    productFirst = 0.0;
                    productSecond = 0.0;
                    qoutIndex = qCol * N * 8 + mIr * 8; //col row  this should go to 64, but only goes to 24, why?
                    //std::cout << "qoutIndex = " << qoutIndex << ", mIr = " << mIr << ", mBrI = " << mBrI << std::endl;
                    mIc = mBcI * fMatCols; // Matrix Column Index, this is somehow very wrong
                    qColNEight = qCol * N * 8;
                    for (auto j = 0; j < fMatCols; j++)
                    {
                        // std::cout << "qCol = " << qCol << ", mBcI = " << mBcI << ", fMatCols = " << fMatCols << ", mIc = " << mIc << ", mIr = " << mIr << std::endl;
                        qinIndex = qColNEight + mIc * 8; // columns in the matrix are rows in the vector
                        inFirst.load(&Qin[qinIndex]);
                        inSecond.load(&Qin[qinIndex + 4]);
                        entryM = (*colIterator)[i][j]; // This seems fine
                        //std::cout << entryM << std::endl;

                        productFirst += entryM * inFirst;
                        productSecond += entryM * inSecond;
                        mIc++;
                    }
                    productFirst.store(&Qout[qoutIndex]);
                    productSecond.store(&Qout[qoutIndex + 4]);
                }
            }
        }
    }
}

/*
    MultQSimple
    Matrix multiplication with double Q 
    This simplified Multiplication assumes that the Matrix consists of 1x1 submatrices
    Uses Vec4d for vectorization
    I dunno, maybe reordering, so the qCol loop is outside, maybe reducing the index calcs?
*/
template <typename MT>
void MultQSimple(const MT &M, const std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, const size_t qCols, const size_t N)
{
    Vec4d productFirst, productSecond, inFirst, inSecond, outFirst, outSecond, entryM;
    size_t qinIndex, qoutIndex;
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
            productFirst.store(&Qout[qoutIndex]);
            productSecond.store(&Qout[qoutIndex + 4]);
        }
    }
}

template <typename MT>
void MultQSimpleUnique(const MT &M, const std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, const size_t qCols, const size_t N)
{
    Vec4d productFirst, productSecond, inFirst, inSecond, outFirst, outSecond, entryM;
    size_t qinIndex, qoutIndex, vectorStart;
    // pull initalization out of loop
    // How to make this faster? Take the qCol loop outside? Why would that make it faster
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

template <typename MT>
void MultQSimpleShared(const MT &M, const std::shared_ptr<double[]> &Qin, std::shared_ptr<double[]> &Qout, const size_t qCols, const size_t N)
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
    PowerIteration
    Power Iteration with double Q, assuming the 2x1 Block data structure
    This simplified Multiplication assumes that the Matrix consists of 1x1 submatrices
    Uses Vec4d for vectorization
*/
template <typename MT>
void powerIterationOld(const MT &M, std::shared_ptr<double[]> &Qin, std::shared_ptr<double[]> &Qout, const size_t qCols, const size_t N)
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
        // Should probably do the sqrt before and not inside the loop
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

/*
    PowerIteration
    Power Iteration with double Q, assuming the 2x1 Block data structure
    This simplified Multiplication assumes that the Matrix consists of 1x1 submatrices
    Uses Vec4d for vectorization
*/
template <typename MT>
void powerIterationOld(const MT &M, std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, const size_t qCols, const size_t N)
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
        // Should probably do the sqrt before and not inside the loop
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

/*
    MultQSimpleNaive
    Matrix multiplication with double Q 
    This simplified Multiplication assumes that the Matrix consists of 1x1 submatrices
    No vectorization
*/
template <typename MT>
void MultQSimpleNaive(const MT &M, std::unique_ptr<double[]> &Qin, std::unique_ptr<double[]> &Qout, size_t qCols, size_t N)
{
    double product, entryM;
    size_t qinIndex, qoutIndex;
    auto endRow = M.end();
    for (auto rowIterator = M.begin(); rowIterator != endRow; rowIterator++)
    {
        auto endCol = (*rowIterator).end();
        auto rowMatrix = rowIterator.index();
        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            for (size_t subqCol = 0; subqCol < 8; subqCol++)
            {
                qoutIndex = 8 * N * qCol + rowMatrix * 8 + subqCol;
                product = 0.0;
                for (auto colIterator = (*rowIterator).begin(); colIterator != endCol; colIterator++)
                {
                    auto colMatrix = colIterator.index();
                    qinIndex = 8 * N * qCol + 8 * colMatrix + subqCol;
                    entryM = (*colIterator)[0][0];
                    product += entryM * Qin[qinIndex];
                }
                Qout[qoutIndex] = product;
            }
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