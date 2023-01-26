#ifndef DUNE_MOES_QRCOL_HH
#define DUNE_MOES_QRCOL_HH

#include <cstdlib>
#include <cmath>
#include <dune/moes/vectorclass/vectorclass.h>

void fillMatrixRandom(Vec4d *Q, size_t matrixSize)
{
    for (size_t i = 0; i < matrixSize; i++)
    {
        Q[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
}

void fillMatrixRandom(double *Q, size_t matrixSize)
{
    for (size_t i = 0; i < matrixSize; i++)
    {
        Q[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
}

void fillMatrixRandom(std::unique_ptr<double[]> &Q, size_t matrixSize)
{
    for (size_t i = 0; i < matrixSize; i++)
    {
        Q[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
}
void fillMatrixRandom(std::shared_ptr<double[]> &Q, size_t matrixSize)
{
    for (size_t i = 0; i < matrixSize; i++)
    {
        Q[i] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;
    }
}

void printMatrix(double *Q, const size_t N, const size_t rhsWidth)
{
    std::cout << std::endl;
    for (size_t col = 0; col < rhsWidth / 8; col++)
    {
        std::cout << std::endl
                  << "Col: " << col << std::endl;
        for (size_t row = 0; row < N; row++)
        {
            for (size_t i = 0; i < 8; i++)
            {
                std::cout << Q[8 * N * col + 8 * row + i] << " ";
            }
            std::cout << std::endl;
        }
    }
}

void printVec(Vec4d v)
{
    std::cout << "(" << v[0] << ", " << v[1] << ", " << v[2] << ", " << v[3] << ")";
}

void checkOrthoNormality(const Vec4d *Q, const size_t rows, const size_t cols, const size_t blockSize, const double tolerance)
{
    Vec4d *dotProducts = new Vec4d[blockSize * blockSize * 4];
    Vec4d u0, u1, u2, u3, u;
    size_t vIndex = 0;
    size_t uIndex = 0;
    double norm = 0.0;
    double dotProduct = 0.0;
    // uindex = col*blockSize*numRows + row*blockSize + iBI;
    for (size_t col = 0; col < cols; col++)
    {
        for (size_t folcol = 0; folcol < cols; folcol++)
        {
            for (size_t i = 0; i < blockSize * blockSize * 4; i++)
            {
                dotProducts[i] = 0.0;
            }
            for (size_t row = 0; row < rows; row++)
            {
                // calculate the dotproducts between all vectors in a block with those of another block
                for (size_t uBI = 0; uBI < blockSize; uBI++)
                {
                    uIndex = col * rows * blockSize + row * blockSize + uBI;
                    u = Q[uIndex];
                    u0 = u[0]; // broadcast
                    u1 = u[1];
                    u2 = u[2];
                    u3 = u[3];
                    for (size_t vBI = 0; vBI < blockSize; vBI++)
                    {
                        // calculate dot products for all blocks (4 * (blockSize)) Vec4ds
                        vIndex = folcol * rows * blockSize + row * blockSize + vBI;
                        // this should be one IF vBI=vBI AND we are looking at the ith element of the ith dotProduct
                        // And the col == folcol
                        dotProducts[uBI * blockSize * 4 + vBI * 4 + 0] += u0 * Q[vIndex];
                        dotProducts[uBI * blockSize * 4 + vBI * 4 + 1] += u1 * Q[vIndex];
                        dotProducts[uBI * blockSize * 4 + vBI * 4 + 2] += u2 * Q[vIndex];
                        dotProducts[uBI * blockSize * 4 + vBI * 4 + 3] += u3 * Q[vIndex];
                    }
                }
            }

            // check Orthonormality for the current Block
            for (size_t blockIndex = 0; blockIndex < blockSize; blockIndex++)
            {
                for (size_t dotBI = 0; dotBI < blockSize; dotBI++)
                {
                    for (size_t i = 0; i < 4; i++)
                    {
                        for (size_t j = 0; j < 4; j++)
                        {
                            // normality
                            if ((col == folcol) && (blockIndex == dotBI) && (i == j))
                            {
                                norm = dotProducts[blockIndex * blockSize * 4 + dotBI * 4 + i][j];
                                if (std::abs(norm - 1.0) > tolerance)
                                {
                                    std::cout << "Norm violation!" << std::endl;
                                    std::cout << "Norm is " << norm << " should be: 1" << std::endl;
                                    std::cout << "Column: " << col * blockSize * 4 + blockIndex * 4 + i << std::endl;
                                    return;
                                }
                            }
                            else
                            {
                                // orthogonality
                                if (std::abs(dotProduct) > tolerance)
                                {
                                    std::cout << "Orthogonality violation!" << std::endl;
                                    std::cout << "Dotproduct is " << dotProduct << " should be: 0" << std::endl;
                                    std::cout << "Col 1: " << col * blockSize * 4 + blockIndex * 4 + i << " Col 2:" << folcol * blockSize * 4 + dotBI * 4 + j << std::endl;
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "All tests successfull!" << std::endl;
}

void checkOrthoNormalityNaive(const double *Q, const size_t numRows, const size_t numCols, const double tolerance)
{
    double dP = 0.0; // dotproduct
    size_t vIndex = 0;
    size_t uIndex = 0;
    for (size_t col = 0; col < numCols; col++)
    {
        for (size_t folcol = col; folcol < numCols; folcol++)
        {
            // calculate the dotproducts between column vectors (including the current one)
            dP = 0.0;
            for (size_t row = 0; row < numRows; row++)
            {
                uIndex = col * numRows + row;
                vIndex = folcol * numRows + row;
                dP += Q[uIndex] * Q[vIndex];
            }
            if (col == folcol)
            {
                // norm test
                if (std::abs(dP - 1.0) > tolerance)
                {
                    std::cout << "checkOrthoNormalityNaive: Norm violation!" << std::endl;
                    std::cout << "Norm is " << dP << " should be: 1" << std::endl;
                    std::cout << "Column: " << col << std::endl;
                    return;
                }
            }
            else
            {
                // orthogonality test
                if (std::abs(dP) > tolerance)
                {
                    std::cout << "checkOrthoNormalityNaive: Orthogonality violation!" << std::endl;
                    std::cout << "Dotproduct is " << dP << " should be: 0" << std::endl;
                    std::cout << "Col 1: " << col << " Col 2:" << folcol << std::endl;
                    return;
                }
            }
        }
    }
    std::cout << "checkOrthoNormalityNaive: All tests successfull!" << std::endl;
}

void checkOrthoNormalityFixed(const Vec4d *Q, const size_t rows, const size_t cols, const double tolerance)
{
    Vec4d *dotProducts = new Vec4d[16];
    Vec4d *iBDP = new Vec4d[16]; // intraBlockDotProducts
    Vec4d u0, u1, u2, u3, u4, u5, u6, u7, uFirst, uSecond, vFirst, vSecond;
    size_t vIndex = 0;
    size_t uIndex = 0;
    double norm = 0.0;
    double dotProduct = 0.0;
    int column = 0;
    // uindex = col*blockSize*numRows + row*blockSize + iBI;
    for (size_t col = 0; col < cols; col++)
    {
        for (size_t folcol = col + 1; folcol < cols; folcol++)
        {
            for (size_t i = 0; i < 16; i++)
            {
                dotProducts[i] = 0.0;
                iBDP[i] = 0.0;
            }
            for (size_t row = 0; row < rows; row++)
            {
                uIndex = col * rows * 2 + row * 2;
                vIndex = folcol * rows * 2 + row * 2;

                // intrablock
                uFirst = Q[uIndex];      // first ublock
                uSecond = Q[uIndex + 1]; // second ublock
                u0 = uFirst[0];          // broadcast
                u1 = uFirst[1];
                u2 = uFirst[2];
                u3 = uFirst[3];
                u4 = uSecond[0]; // broadcast
                u5 = uSecond[1];
                u6 = uSecond[2];
                u7 = uSecond[3];
                iBDP[0] += u0 * uFirst; // iBDP[0][0] should be 1
                iBDP[1] += u1 * uFirst; // iBDP[1][1] should be 1 ...
                iBDP[2] += u2 * uFirst;
                iBDP[3] += u3 * uFirst;

                iBDP[4] += u4 * uSecond; // iBDP[8][0] should be 1
                iBDP[5] += u5 * uSecond; // iBDP[9][1] should be 1 ....
                iBDP[6] += u6 * uSecond;
                iBDP[7] += u7 * uSecond;

                iBDP[8] += u0 * uSecond; // should be 0 (first with second block)
                iBDP[9] += u1 * uSecond;
                iBDP[10] += u2 * uSecond;
                iBDP[11] += u3 * uSecond;

                iBDP[12] += u4 * uFirst; // should be 0
                iBDP[13] += u5 * uFirst;
                iBDP[14] += u6 * uFirst;
                iBDP[15] += u7 * uFirst;

                // end intrablock

                // between blocks
                vFirst = Q[vIndex];      // first v Block
                vSecond = Q[vIndex + 1]; // second v Block
                dotProducts[0] = u0 * vFirst;
                dotProducts[1] = u1 * vFirst;
                dotProducts[2] = u2 * vFirst;
                dotProducts[3] = u3 * vFirst;

                dotProducts[4] = u4 * vFirst;
                dotProducts[5] = u5 * vFirst;
                dotProducts[6] = u6 * vFirst;
                dotProducts[7] = u7 * vFirst;

                dotProducts[8] = u0 * vSecond;
                dotProducts[9] = u1 * vSecond;
                dotProducts[10] = u2 * vSecond;
                dotProducts[11] = u3 * vSecond;

                dotProducts[12] = u4 * vSecond;
                dotProducts[13] = u5 * vSecond;
                dotProducts[14] = u6 * vSecond;
                dotProducts[15] = u7 * vSecond;
                // end between blocks
            }

            // check Orthonormality inside block
            for (size_t intradots = 0; intradots < 8; intradots++)
            {
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    column = col * 8 + intradots;
                    if (intradots % 4 == iVI) // comparison with self
                    {
                        if (std::abs(iBDP[intradots][iVI] - 1.0) > tolerance)
                        {
                            std::cout << "Norm violation!" << std::endl;
                            std::cout << "Norm is " << iBDP[intradots][iVI] << " should be: 1" << std::endl;
                            std::cout << "Column (integer from 1..N): " << column << std::endl;
                            return;
                        }
                    }
                    else
                    {
                        if (std::abs(iBDP[intradots][iVI]) > tolerance)
                        {
                            std::cout << "Orthogonality violation inside block!" << std::endl;
                            std::cout << "Dotproduct is " << iBDP[intradots][iVI] << " should be: 0" << std::endl;
                            std::cout << "Col (this is only the col blocks 1..N/8): " << col << std::endl;
                            return;
                        }
                    }
                }
            }

            // Orthogonality between blocks
            for (size_t intradots = 8; intradots < 16; intradots++)
            {
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    if (std::abs(dotProducts[intradots][iVI]) > tolerance)
                    {
                        std::cout << "Orthogonality violation between blocks!" << std::endl;
                        std::cout << "Dotproduct is " << dotProducts[intradots][iVI] << " should be: 0" << std::endl;
                        std::cout << "Col (this is only the col blocks 1..N/8) 1: " << col << " Col 2:" << folcol << std::endl;
                        return;
                    }
                }
            }
        }
    }
    std::cout << "All tests successfull!" << std::endl;
}

void checkOrthoNormalityFixed(const std::unique_ptr<double[]> &Q, const size_t rows, const size_t cols, const double tolerance)
{
    Vec4d *dotProducts = new Vec4d[16];
    Vec4d *iBDP = new Vec4d[16]; // intraBlockDotProducts
    Vec4d u0, u1, u2, u3, u4, u5, u6, u7, uFirst, uSecond, vFirst, vSecond;
    size_t vIndex = 0;
    size_t uIndex = 0;
    double norm = 0.0;
    double dotProduct = 0.0;
    int column = 0;
    // uindex = col*blockSize*numRows + row*blockSize + iBI;
    for (size_t col = 0; col < cols; col++)
    {
        for (size_t folcol = col + 1; folcol < cols; folcol++)
        {
            for (size_t i = 0; i < 16; i++)
            {
                dotProducts[i] = 0.0;
                iBDP[i] = 0.0;
            }
            for (size_t row = 0; row < rows; row++)
            {
                uIndex = col * rows * 8 + row * 8;
                vIndex = folcol * rows * 8 + row * 8;

                // intrablock
                uFirst.load(&Q[uIndex]);      // first ublock
                uSecond.load(&Q[uIndex + 4]); // second ublock
                u0 = uFirst[0];               // broadcast
                u1 = uFirst[1];
                u2 = uFirst[2];
                u3 = uFirst[3];
                u4 = uSecond[0]; // broadcast
                u5 = uSecond[1];
                u6 = uSecond[2];
                u7 = uSecond[3];
                iBDP[0] += u0 * uFirst; // iBDP[0][0] should be 1
                iBDP[1] += u1 * uFirst; // iBDP[1][1] should be 1 ...
                iBDP[2] += u2 * uFirst;
                iBDP[3] += u3 * uFirst;

                iBDP[4] += u4 * uSecond; // iBDP[8][0] should be 1
                iBDP[5] += u5 * uSecond; // iBDP[9][1] should be 1 ....
                iBDP[6] += u6 * uSecond;
                iBDP[7] += u7 * uSecond;

                iBDP[8] += u0 * uSecond; // should be 0 (first with second block)
                iBDP[9] += u1 * uSecond;
                iBDP[10] += u2 * uSecond;
                iBDP[11] += u3 * uSecond;

                iBDP[12] += u4 * uFirst; // should be 0
                iBDP[13] += u5 * uFirst;
                iBDP[14] += u6 * uFirst;
                iBDP[15] += u7 * uFirst;

                // end intrablock

                // between blocks
                vFirst.load(&Q[vIndex]);      // first v Block
                vSecond.load(&Q[vIndex + 4]); // second v Block
                dotProducts[0] += u0 * vFirst;
                dotProducts[1] += u1 * vFirst;
                dotProducts[2] += u2 * vFirst;
                dotProducts[3] += u3 * vFirst;

                dotProducts[4] += u4 * vFirst;
                dotProducts[5] += u5 * vFirst;
                dotProducts[6] += u6 * vFirst;
                dotProducts[7] += u7 * vFirst;

                dotProducts[8] += u0 * vSecond;
                dotProducts[9] += u1 * vSecond;
                dotProducts[10] += u2 * vSecond;
                dotProducts[11] += u3 * vSecond;

                dotProducts[12] += u4 * vSecond;
                dotProducts[13] += u5 * vSecond;
                dotProducts[14] += u6 * vSecond;
                dotProducts[15] += u7 * vSecond;
                // end between blocks
            }

            // check Orthonormality inside block
            for (size_t intradots = 0; intradots < 8; intradots++)
            {
                column = col * 8 + intradots;
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    if (intradots % 4 == iVI) // comparison with self
                    {
                        if (std::abs(iBDP[intradots][iVI] - 1.0) > tolerance)
                        {
                            std::cout << "Norm violation!" << std::endl;
                            std::cout << "Norm is " << iBDP[intradots][iVI] << " should be: 1" << std::endl;
                            std::cout << "Column (integer from 1..N/8): " << column << std::endl;
                            return;
                        }
                    }
                    else
                    {
                        if (std::abs(iBDP[intradots][iVI]) > tolerance)
                        {
                            std::cout << "Orthogonality violation inside block!" << std::endl;
                            std::cout << "Dotproduct is " << iBDP[intradots][iVI] << " should be: 0" << std::endl;
                            std::cout << "Col (this is only the col blocks 1..N/8): " << col << " Intradots/iVI: " << intradots << "/" << iVI << std::endl;
                            return;
                        }
                    }
                }
            }

            // Orthogonality between inside blocks
            for (size_t second = 8; second < 16; second++)
            {
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    if (std::abs(iBDP[second][iVI]) > tolerance)
                    {
                        std::cout << "Orthogonality violation between two intra blocks!" << std::endl;
                        std::cout << "Dotproduct is " << iBDP[second][iVI] << " should be: 0" << std::endl;
                        std::cout << "Col (this is only the col blocks 1..N/8): " << col << std::endl;
                        return;
                    }
                }
            }

            // Orthogonality between blocks
            for (size_t betweenDots = 0; betweenDots < 16; betweenDots++)
            {
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    if (std::abs(dotProducts[betweenDots][iVI]) > tolerance)
                    {
                        std::cout << "Orthogonality violation between blocks!" << std::endl;
                        std::cout << "Dotproduct is " << dotProducts[betweenDots][iVI] << " should be: 0" << std::endl;
                        std::cout << "Col (this is only the col blocks 1..N/8) 1: " << col << " Col 2:" << folcol << std::endl;
                        return;
                    }
                }
            }
        }
    }
    std::cout << "All tests successfull!" << std::endl;
    delete[] dotProducts;
    delete[] iBDP;
}

void checkOrthoNormalityFixed(const std::shared_ptr<double[]> &Q, const size_t rows, const size_t cols, const double tolerance)
{
    Vec4d *dotProducts = new Vec4d[16];
    Vec4d *iBDP = new Vec4d[16]; // intraBlockDotProducts
    Vec4d u0, u1, u2, u3, u4, u5, u6, u7, uFirst, uSecond, vFirst, vSecond;
    size_t vIndex = 0;
    size_t uIndex = 0;
    double norm = 0.0;
    double dotProduct = 0.0;
    int column = 0;
    // uindex = col*blockSize*numRows + row*blockSize + iBI;
    for (size_t col = 0; col < cols; col++)
    {
        for (size_t folcol = col + 1; folcol < cols; folcol++)
        {
            for (size_t i = 0; i < 16; i++)
            {
                dotProducts[i] = 0.0;
                iBDP[i] = 0.0;
            }
            for (size_t row = 0; row < rows; row++)
            {
                uIndex = col * rows * 8 + row * 8;
                vIndex = folcol * rows * 8 + row * 8;

                // intrablock
                uFirst.load(&Q[uIndex]);      // first ublock
                uSecond.load(&Q[uIndex + 4]); // second ublock
                u0 = uFirst[0];               // broadcast
                u1 = uFirst[1];
                u2 = uFirst[2];
                u3 = uFirst[3];
                u4 = uSecond[0]; // broadcast
                u5 = uSecond[1];
                u6 = uSecond[2];
                u7 = uSecond[3];
                iBDP[0] += u0 * uFirst; // iBDP[0][0] should be 1
                iBDP[1] += u1 * uFirst; // iBDP[1][1] should be 1 ...
                iBDP[2] += u2 * uFirst;
                iBDP[3] += u3 * uFirst;

                iBDP[4] += u4 * uSecond; // iBDP[8][0] should be 1
                iBDP[5] += u5 * uSecond; // iBDP[9][1] should be 1 ....
                iBDP[6] += u6 * uSecond;
                iBDP[7] += u7 * uSecond;

                iBDP[8] += u0 * uSecond; // should be 0 (first with second block)
                iBDP[9] += u1 * uSecond;
                iBDP[10] += u2 * uSecond;
                iBDP[11] += u3 * uSecond;

                iBDP[12] += u4 * uFirst; // should be 0
                iBDP[13] += u5 * uFirst;
                iBDP[14] += u6 * uFirst;
                iBDP[15] += u7 * uFirst;

                // end intrablock

                // between blocks
                vFirst.load(&Q[vIndex]);      // first v Block
                vSecond.load(&Q[vIndex + 4]); // second v Block
                dotProducts[0] += u0 * vFirst;
                dotProducts[1] += u1 * vFirst;
                dotProducts[2] += u2 * vFirst;
                dotProducts[3] += u3 * vFirst;

                dotProducts[4] += u4 * vFirst;
                dotProducts[5] += u5 * vFirst;
                dotProducts[6] += u6 * vFirst;
                dotProducts[7] += u7 * vFirst;

                dotProducts[8] += u0 * vSecond;
                dotProducts[9] += u1 * vSecond;
                dotProducts[10] += u2 * vSecond;
                dotProducts[11] += u3 * vSecond;

                dotProducts[12] += u4 * vSecond;
                dotProducts[13] += u5 * vSecond;
                dotProducts[14] += u6 * vSecond;
                dotProducts[15] += u7 * vSecond;
                // end between blocks
            }

            // check Orthonormality inside block
            for (size_t intradots = 0; intradots < 8; intradots++)
            {
                column = col * 8 + intradots;
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    if (intradots % 4 == iVI) // comparison with self
                    {
                        if (std::abs(iBDP[intradots][iVI] - 1.0) > tolerance)
                        {
                            std::cout << "Norm violation!" << std::endl;
                            std::cout << "Norm is " << iBDP[intradots][iVI] << " should be: 1" << std::endl;
                            std::cout << "Column (integer from 1..N/8): " << column << std::endl;
                            return;
                        }
                    }
                    else
                    {
                        if (std::abs(iBDP[intradots][iVI]) > tolerance)
                        {
                            std::cout << "Orthogonality violation inside block!" << std::endl;
                            std::cout << "Dotproduct is " << iBDP[intradots][iVI] << " should be: 0" << std::endl;
                            std::cout << "Col (this is only the col blocks 1..N/8): " << col << " Intradots/iVI: " << intradots << "/" << iVI << std::endl;
                            return;
                        }
                    }
                }
            }

            // Orthogonality between inside blocks
            for (size_t second = 8; second < 16; second++)
            {
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    if (std::abs(iBDP[second][iVI]) > tolerance)
                    {
                        std::cout << "Orthogonality violation between two intra blocks!" << std::endl;
                        std::cout << "Dotproduct is " << iBDP[second][iVI] << " should be: 0" << std::endl;
                        std::cout << "Col (this is only the col blocks 1..N/8): " << col << std::endl;
                        return;
                    }
                }
            }

            // Orthogonality between blocks
            for (size_t betweenDots = 0; betweenDots < 16; betweenDots++)
            {
                for (size_t iVI = 0; iVI < 4; iVI++)
                {
                    if (std::abs(dotProducts[betweenDots][iVI]) > tolerance)
                    {
                        std::cout << "Orthogonality violation between blocks!" << std::endl;
                        std::cout << "Dotproduct is " << dotProducts[betweenDots][iVI] << " should be: 0" << std::endl;
                        std::cout << "Col (this is only the col blocks 1..N/8) 1: " << col << " Col 2:" << folcol << std::endl;
                        return;
                    }
                }
            }
        }
    }
    std::cout << "All tests successfull!" << std::endl;
    delete[] dotProducts;
    delete[] iBDP;
}

/*
    rhsWidth = numCols*blockSize*4 (i.e. blockSize=2 comprises 2 Vec4ds = 8 columns of doubles)
    N = numRows
    Q.size() = numCols * numRows of Vec4ds
    One Block = blockSize*numRows
    uBlockSize <= blockSize: when orthonormalizing a block with an already orthed block, gives the other side of the rectangle for the index range 
*/
void qr(Vec4d *Q, size_t numRows, size_t numCols, size_t blockSize, size_t uBlockSize)
{
    Vec4d *norms = new Vec4d[blockSize];
    Vec4d *dotProducts = new Vec4d[blockSize * blockSize * 4];
    double *udots = new double[blockSize * 4];
    double ui = 0.0;
    Vec4d u = 0.0;
    Vec4d v = 0.0;
    size_t index = 0;
    size_t uindex = 0;
    size_t vindex = 0;
    size_t dotIndex = 0;
    Vec4d normSqrt = 0.0;
    Vec4d uiVec = 0.0;
    Vec4d viVec = 0.0;
    Vec4d uivi = 0.0;
    Vec4d uVec = 0.0;
    Vec4d u0Vec = 0.0;
    Vec4d u1Vec = 0.0;
    Vec4d u2Vec = 0.0;
    Vec4d u3Vec = 0.0;
    Vec4d v0Vec = 0.0;
    Vec4d v1Vec = 0.0;
    Vec4d v2Vec = 0.0;
    Vec4d v3Vec = 0.0;
    for (size_t col = 0; col < numCols; col++)
    {
        // orthogonalize current block
        for (size_t i = 0; i < blockSize; i++)
        {
            norms[i] = 0.0;
        }

        for (size_t iBI = 0; iBI < blockSize; iBI++) // intraBlockIndex
        {
            for (size_t iVI = 0; iVI < 4; iVI++) // intraVectorIndex
            {
                for (size_t i = 0; i < blockSize * 4; i++)
                {
                    udots[i] = 0.0;
                }

                // calculate dot products with current column
                for (size_t row = 0; row < numRows; row++)
                {
                    uindex = col * blockSize * numRows + row * blockSize + iBI;
                    ui = Q[uindex][iVI];
                    norms[iBI].insert(iVI, norms[iBI][iVI] + ui * ui);
                    // dot product with the following columns in the same Vec4d in the same block
                    for (size_t iVIfol = iVI + 1; iVIfol < 4; iVIfol++)
                    {
                        udots[iBI * 4 + iVIfol] += ui * Q[uindex][iVIfol];
                    }
                    // dot product following block
                    for (size_t iBIfol = iBI + 1; iBIfol < blockSize; iBIfol++)
                    {
                        vindex = col * blockSize * numRows + row * blockSize + iBIfol;
                        uiVec = ui;
                        // viVec.load(&Q[vindex]);
                        viVec = Q[vindex];
                        uivi.load(&udots[iBIfol * 4]);
                        uivi += uiVec * viVec;
                        uivi.store(&udots[iBIfol * 4]);
                    }
                }
                // Linear combination with current column
                for (size_t row = 0; row < numRows; row++)
                {
                    uindex = col * blockSize * numRows + row * blockSize + iBI;
                    ui = Q[uindex][iVI];
                    // apply linear combination to following columns in the same Vec4d in the same block
                    for (size_t iVIfol = iVI + 1; iVIfol < 4; iVIfol++)
                    {
                        Q[uindex].insert(iVIfol, Q[uindex][iVIfol] - udots[iBI * 4 + iVIfol] / norms[iBI][iVI] * ui);
                    }
                    // apply linear combinations to following Vec4d-columns in the same block
                    for (size_t iBIfol = iBI + 1; iBIfol < blockSize; iBIfol++)
                    {
                        vindex = col * blockSize * numRows + row * blockSize + iBIfol;
                        uiVec = ui;
                        uivi.load(&udots[iBIfol * 4]);
                        Q[vindex] -= uiVec * uivi / norms[iBI][iVI]; // This line might cause trouble because we are dividing a Vec4d by a scalar
                    }
                }
            }
        }
        // end orthogonalize current Block

        // normalize current block
        for (size_t row = 0; row < numRows; row++)
        {
            for (size_t iBI = 0; iBI < blockSize; iBI++)
            {
                index = col * blockSize * numRows + row * blockSize + iBI;
                Q[index] /= sqrt(norms[iBI]); // God I hope this is fine
            }
        }
        // end normalize current block

        // orthogonalize following blocks
        for (size_t i = 0; i < blockSize * blockSize * 4; i++)
        {
            dotProducts[i] = 0;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // get dotProducts
            for (size_t uBIBlock = 0; uBIBlock < blockSize; uBIBlock += uBlockSize)
            {
                for (size_t row = 0; row < numRows; row++)
                {
                    for (size_t uBI = 0; uBI < uBlockSize; uBI++)
                    {
                        uindex = col * blockSize * numRows + row * blockSize + uBIBlock * uBlockSize + uBI; // nightmare
                        u = Q[uindex];
                        // load registers
                        u0Vec = u[0];
                        u1Vec = u[1];
                        u2Vec = u[2];
                        u3Vec = u[3];
                        for (size_t vBI = 0; vBI < blockSize; vBI++)
                        {
                            vindex = folCol * blockSize * numRows + row * blockSize + vBI;
                            dotIndex = (uBIBlock * uBlockSize + uBI) * blockSize * 4 + vBI * 4;
                            v = Q[vindex];
                            dotProducts[dotIndex + 0] += u0Vec * v;
                            dotProducts[dotIndex + 1] += u1Vec * v;
                            dotProducts[dotIndex + 2] += u2Vec * v;
                            dotProducts[dotIndex + 3] += u3Vec * v;
                        }
                    }
                }
            }
            // linear combination
            for (size_t uBIBlock = 0; uBIBlock < blockSize; uBIBlock += uBlockSize)
            {
                for (size_t row = 0; row < numRows; row++)
                {
                    for (size_t uBI = 0; uBI < uBlockSize; uBI++)
                    {
                        uindex = col * blockSize * numRows + row * blockSize + uBIBlock * uBlockSize + uBI; // nightmare
                        u = Q[uindex];
                        // load registers
                        u0Vec = u[0];
                        u1Vec = u[1];
                        u2Vec = u[2];
                        u3Vec = u[3];
                        for (size_t vBI = 0; vBI < blockSize; vBI++)
                        {
                            vindex = folCol * blockSize * numRows + row * blockSize + vBI;
                            dotIndex = (uBIBlock * uBlockSize + uBI) * blockSize * 4 + vBI * 4;
                            v = Q[vindex];
                            Q[vindex] -= dotProducts[dotIndex + 0] * u0Vec + dotProducts[dotIndex + 1] * u1Vec + dotProducts[dotIndex + 2] * u2Vec + dotProducts[dotIndex + 3] * u3Vec;
                        }
                    }
                }
            }
        }
        // end orthogonalize following blocks
        // maybe easier without setting the blockSize because of the registers
    }
}

// Same as above but with fixed block Sizes (vBlock=2) (uBlock=1)
void qrunblocked(Vec4d *Q, size_t numRows, size_t numCols, size_t blockSize, size_t uBlockSize)
{
    Vec4d *norms = new Vec4d[2];
    double currentNorm = 0.0;
    Vec4d *dotProducts = new Vec4d[16];
    double *udots = new double[8];
    double ui = 0.0;
    Vec4d u = 0.0;
    Vec4d v = 0.0;
    Vec4d v0 = 0.0;
    Vec4d v1 = 0.0;
    size_t index = 0;
    size_t uindex = 0;
    size_t vindex = 0;
    Vec4d uiVec = 0.0;
    Vec4d viVec = 0.0;
    Vec4d uivi = 0.0;
    Vec4d u0Vec = 0.0;
    Vec4d u1Vec = 0.0;
    Vec4d u2Vec = 0.0;
    Vec4d u3Vec = 0.0;
    for (size_t col = 0; col < numCols; col++)
    {
        // orthogonalize current block
        norms[0] = 0.0;
        norms[1] = 0.0;

        for (size_t iBI = 0; iBI < 2; iBI++) // intraBlockIndex
        {
            for (size_t iVI = 0; iVI < 4; iVI++) // intraVectorIndex
            {
                for (size_t i = 0; i < 8; i++)
                {
                    udots[i] = 0.0;
                }
                currentNorm = 0.0;
                // calculate dot products with current column
                uindex = 2 * col * numRows + iBI;
                for (size_t row = 0; row < numRows; row++)
                {
                    // uindex = col*2*numRows + row*2 + iBI;
                    ui = Q[uindex][iVI];
                    currentNorm += ui * ui;
                    // norms[iBI].insert(iVI, norms[iBI][iVI] + ui*ui);
                    // dot product with the following columns in the same Vec4d in the same block
                    for (size_t iVIfol = iVI + 1; iVIfol < 4; iVIfol++)
                    {
                        udots[iBI * 4 + iVIfol] += ui * Q[uindex][iVIfol];
                    }
                    // dot product following block
                    vindex = uindex;
                    for (size_t iBIfol = iBI + 1; iBIfol < 2; iBIfol++)
                    {
                        // vindex = col*2*numRows + row*2 + iBIfol;
                        vindex++;
                        uiVec = ui;
                        // viVec.load(&Q[vindex]);
                        viVec = Q[vindex];
                        uivi.load(&udots[iBIfol * 4]);
                        uivi += uiVec * viVec;
                        uivi.store(&udots[iBIfol * 4]);
                    }
                    uindex += 2;
                }
                norms[iBI].insert(iVI, currentNorm); // don't do this for every row
                // Linear combination with current column
                uindex = 2 * col * numRows + iBI;
                for (size_t row = 0; row < numRows; row++)
                {
                    // uindex = col*2*numRows + row*2 + iBI;
                    ui = Q[uindex][iVI];
                    // apply linear combination to following columns in the same Vec4d in the same block
                    for (size_t iVIfol = iVI + 1; iVIfol < 4; iVIfol++)
                    {
                        Q[uindex].insert(iVIfol, Q[uindex][iVIfol] - udots[iBI * 4 + iVIfol] / norms[iBI][iVI] * ui); // Does this have a big impact? (around 5%)
                    }
                    // apply linear combinations to following Vec4d-columns in the same block
                    vindex = uindex;
                    for (size_t iBIfol = iBI + 1; iBIfol < 2; iBIfol++)
                    {
                        vindex++;
                        // vindex = col*2*numRows + row*2 + iBIfol;
                        uiVec = ui;
                        uivi.load(&udots[iBIfol * 4]);
                        Q[vindex] -= uiVec * uivi / norms[iBI][iVI];
                    }
                    uindex += 2;
                }
            }
        }
        // end orthogonalize current Block

        // normalize current block
        index = col * 2 * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // index = col*2*numRows + 2*row;
            Q[index] /= sqrt(norms[0]);
            Q[index + 1] /= sqrt(norms[1]);
            index += 2;
        }
        // end normalize current block

        // orthogonalize following blocks
        for (size_t i = 0; i < 16; i++)
        {
            dotProducts[i] = 0;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // get dotProducts

            // first uBlock
            uindex = col * 2 * numRows;
            vindex = folCol * 2 * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u = Q[uindex];
                v0 = Q[vindex];
                v1 = Q[vindex + 1]; // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[0] += u0Vec * v0;
                dotProducts[1] += u1Vec * v0;
                dotProducts[2] += u2Vec * v0;
                dotProducts[3] += u3Vec * v0;

                dotProducts[4] += u0Vec * v1;
                dotProducts[5] += u1Vec * v1;
                dotProducts[6] += u2Vec * v1;
                dotProducts[7] += u3Vec * v1;
                uindex += 2;
                vindex += 2;
            }

            uindex = col * 2 * numRows + 1;
            vindex = folCol * 2 * numRows;
            // second uBlock
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u = Q[uindex];
                v0 = Q[vindex];
                v1 = Q[vindex + 1]; // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[8] += u0Vec * v0;
                dotProducts[9] += u1Vec * v0;
                dotProducts[10] += u2Vec * v0;
                dotProducts[11] += u3Vec * v0;

                dotProducts[12] += u0Vec * v1;
                dotProducts[13] += u1Vec * v1;
                dotProducts[14] += u2Vec * v1;
                dotProducts[15] += u3Vec * v1;
                uindex += 2;
                vindex += 2;
            }

            // end dotproducts

            // linear combination

            // first uBlock
            uindex = col * 2 * numRows;
            vindex = folCol * 2 * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u = Q[uindex];
                v0 = Q[vindex];
                v1 = Q[vindex + 1]; // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                Q[vindex] -= dotProducts[0] * u0Vec + dotProducts[1] * u1Vec + dotProducts[2] * u2Vec + dotProducts[3] * u3Vec; // this should have -=
                Q[vindex + 1] -= dotProducts[4] * u0Vec + dotProducts[5] * u1Vec + dotProducts[6] * u2Vec + dotProducts[7] * u3Vec;
                uindex += 2;
                vindex += 2;
            }

            // second uBlock
            uindex = col * 2 * numRows + 1;
            vindex = folCol * 2 * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u = Q[uindex];
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                Q[vindex] -= dotProducts[8] * u0Vec + dotProducts[9] * u1Vec + dotProducts[10] * u2Vec + dotProducts[11] * u3Vec;
                Q[vindex + 1] -= dotProducts[12] * u0Vec + dotProducts[13] * u1Vec + dotProducts[14] * u2Vec + dotProducts[15] * u3Vec;
                uindex += 2;
                vindex += 2;
            }

            // end linear combination
        }
        // end orthogonalize following blocks
    }
}

void qrFixedBlockOptimizedVec4d(Vec4d *Q, size_t numRows, size_t numCols, size_t blockSize, size_t uBlockSize)
{
    Vec4d *norms = new Vec4d[2];
    double currentNorm = 0.0;
    Vec4d *dotProducts = new Vec4d[16];
    double *udots = new double[8];
    double ui = 0.0;
    double lastBlockColNorm = 0.0;
    Vec4d u = 0.0;
    Vec4d uNorm = 0.0;
    Vec4d v = 0.0;
    Vec4d uv = 0.0;
    Vec4d v0 = 0.0;
    Vec4d v1 = 0.0;
    size_t index = 0;
    size_t uindex = 0;
    size_t vindex = 0;
    Vec4d uiVec = 0.0;
    Vec4d viVec = 0.0;
    Vec4d uivi = 0.0;
    Vec4d u0Vec = 0.0;
    Vec4d u1Vec = 0.0;
    Vec4d u2Vec = 0.0;
    Vec4d u3Vec = 0.0;
    for (size_t col = 0; col < numCols; col++)
    {
        // orthogonalize current block
        norms[0] = 0.0;
        norms[1] = 0.0;

        // 0th entry
        udots[0] = 0.0;
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 2 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex][0];
            udots[0] += ui * ui;
            udots[1] += ui * Q[uindex][1];
            udots[2] += ui * Q[uindex][2];
            udots[3] += ui * Q[uindex][3];
            uv += ui * Q[uindex + 1];
            uindex += 2;
        }
        uv.store(&udots[4]);
        norms[0].insert(0, udots[0]);
        // linear combination
        uindex = 2 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex] = {Q[uindex][0], Q[uindex][1] - udots[1] / udots[0] * Q[uindex][0], Q[uindex][2] - udots[2] / udots[0] * Q[uindex][0], Q[uindex][3] - udots[3] / udots[0] * Q[uindex][0]};
            uNorm = udots[0];
            uiVec = Q[uindex][0];
            Q[uindex + 1] -= uv / uNorm * uiVec;
            uindex += 2;
        }

        // 1st entry
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 2 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex][1];
            udots[1] += ui * ui;
            udots[2] += ui * Q[uindex][2];
            udots[3] += ui * Q[uindex][3];
            uv += ui * Q[uindex + 1];
            uindex += 2;
        }
        uv.store(&udots[4]);
        norms[0].insert(1, udots[1]);
        // linear combination
        uindex = 2 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2] - udots[2] / udots[1] * Q[uindex][1], Q[uindex][3] - udots[3] / udots[1] * Q[uindex][1]};
            uNorm = udots[1];
            uiVec = Q[uindex][1];
            Q[uindex + 1] -= uv / uNorm * uiVec;
            uindex += 2;
        }

        // 2nd entry
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        lastBlockColNorm = 0.0;
        uv = 0.0;
        uindex = 2 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex][2];
            udots[2] += ui * ui;
            udots[3] += ui * Q[uindex][3];
            uv += ui * Q[uindex + 1];
            uindex += 2;
        }
        uv.store(&udots[4]);
        norms[0].insert(2, udots[2]);
        // linear combination
        uindex = 2 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            ui = Q[uindex][3] - udots[3] / udots[2] * Q[uindex][2];
            Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2], ui};
            lastBlockColNorm += ui * ui;
            uNorm = udots[2];
            uiVec = Q[uindex][2];
            Q[uindex + 1] -= uv / uNorm * uiVec;
            uindex += 2;
        }
        norms[0].insert(3, lastBlockColNorm);

        // 3rd entry
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 2 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex][3];
            udots[3] += ui * ui;
            uv += ui * Q[uindex + 1];
            uindex += 2;
        }
        uv.store(&udots[4]);
        norms[0].insert(2, udots[2]);
        // linear combination
        uindex = 2 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            // Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2], Q[uindex][3]}; // dont need this anymore
            uNorm = udots[3];
            uiVec = Q[uindex][3];
            Q[uindex + 1] -= uv / uNorm * uiVec;
            uindex += 2;
        }

        // 4th entry
        udots[0] = 0.0;                 // 4
        udots[1] = 0.0;                 // 5
        udots[2] = 0.0;                 // 6
        udots[3] = 0.0;                 // 7
        uindex = 2 * col * numRows + 1; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex][0];
            udots[0] += ui * ui;
            udots[1] += ui * Q[uindex][1];
            udots[2] += ui * Q[uindex][2];
            udots[3] += ui * Q[uindex][3];
            uindex += 2;
        }
        norms[1].insert(0, udots[0]);
        // linear combination
        uindex = 2 * col * numRows + 1;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex] = {Q[uindex][0], Q[uindex][1] - udots[1] / udots[0] * Q[uindex][0], Q[uindex][2] - udots[2] / udots[0] * Q[uindex][0], Q[uindex][3] - udots[3] / udots[0] * Q[uindex][0]};
            uindex += 2;
        }

        // 5th entry
        udots[1] = 0.0;                 // 5
        udots[2] = 0.0;                 // 6
        udots[3] = 0.0;                 // 7
        uindex = 2 * col * numRows + 1; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex][1];
            udots[1] += ui * ui;
            udots[2] += ui * Q[uindex][2];
            udots[3] += ui * Q[uindex][3];
            uindex += 2;
        }
        norms[1].insert(1, udots[1]);
        // linear combination
        uindex = 2 * col * numRows + 1;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2] - udots[2] / udots[1] * Q[uindex][1], Q[uindex][3] - udots[3] / udots[1] * Q[uindex][1]};
            uindex += 2;
        }

        // 6th entry
        udots[2] = 0.0; // 6
        udots[3] = 0.0; // 7
        lastBlockColNorm = 0.0;
        uindex = 2 * col * numRows + 1; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex][2];
            udots[2] += ui * ui;
            udots[3] += ui * Q[uindex][3];
            uindex += 2;
        }
        norms[1].insert(2, udots[2]);
        // linear combination
        uindex = 2 * col * numRows + 1;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            ui = Q[uindex][3] - udots[3] / udots[2] * Q[uindex][2];
            Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2], ui};
            lastBlockColNorm += ui * ui;
            uindex += 2;
        }
        norms[1].insert(3, lastBlockColNorm);

        // normalize current block
        index = col * 2 * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // index = col*2*numRows + 2*row;
            Q[index] /= sqrt(norms[0]);
            Q[index + 1] /= sqrt(norms[1]);
            index += 2;
        }
        // end normalize current block

        // orthogonalize following blocks
        for (size_t i = 0; i < 16; i++)
        {
            dotProducts[i] = 0;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // get dotProducts

            // first uBlock
            uindex = col * 2 * numRows;
            vindex = folCol * 2 * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u = Q[uindex];
                v0 = Q[vindex];
                v1 = Q[vindex + 1]; // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[0] += u0Vec * v0;
                dotProducts[1] += u1Vec * v0;
                dotProducts[2] += u2Vec * v0;
                dotProducts[3] += u3Vec * v0;

                dotProducts[4] += u0Vec * v1;
                dotProducts[5] += u1Vec * v1;
                dotProducts[6] += u2Vec * v1;
                dotProducts[7] += u3Vec * v1;
                uindex += 2;
                vindex += 2;
            }

            uindex = col * 2 * numRows + 1;
            vindex = folCol * 2 * numRows;
            // second uBlock
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u = Q[uindex];
                v0 = Q[vindex];
                v1 = Q[vindex + 1]; // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[8] += u0Vec * v0;
                dotProducts[9] += u1Vec * v0;
                dotProducts[10] += u2Vec * v0;
                dotProducts[11] += u3Vec * v0;

                dotProducts[12] += u0Vec * v1;
                dotProducts[13] += u1Vec * v1;
                dotProducts[14] += u2Vec * v1;
                dotProducts[15] += u3Vec * v1;
                uindex += 2;
                vindex += 2;
            }

            // end dotproducts

            // linear combination

            // first uBlock
            uindex = col * 2 * numRows;
            vindex = folCol * 2 * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u = Q[uindex];
                v0 = Q[vindex];
                v1 = Q[vindex + 1]; // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                Q[vindex] -= dotProducts[0] * u0Vec + dotProducts[1] * u1Vec + dotProducts[2] * u2Vec + dotProducts[3] * u3Vec; // this should have -=
                Q[vindex + 1] -= dotProducts[4] * u0Vec + dotProducts[5] * u1Vec + dotProducts[6] * u2Vec + dotProducts[7] * u3Vec;
                uindex += 2;
                vindex += 2;
            }

            // second uBlock
            uindex = col * 2 * numRows + 1;
            vindex = folCol * 2 * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u = Q[uindex];
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                Q[vindex] -= dotProducts[8] * u0Vec + dotProducts[9] * u1Vec + dotProducts[10] * u2Vec + dotProducts[11] * u3Vec;
                Q[vindex + 1] -= dotProducts[12] * u0Vec + dotProducts[13] * u1Vec + dotProducts[14] * u2Vec + dotProducts[15] * u3Vec;
                uindex += 2;
                vindex += 2;
            }

            // end linear combination
        }
        // end orthogonalize following blocks
    }
}

void qrFixedBlockOptimizedDoubleUnsafe(double *Q, size_t numRows, size_t numCols, size_t blockSize, size_t uBlockSize)
{
    Vec4d *norms = new Vec4d[2];
    double currentNorm = 0.0;
    Vec4d *dotProducts = new Vec4d[16];
    double *udots = new double[8];
    double ui = 0.0;
    Vec4d u = 0.0;
    Vec4d uNorm = 0.0;
    Vec4d v = 0.0;
    Vec4d uv = 0.0;
    Vec4d v0 = 0.0;
    Vec4d v1 = 0.0;
    size_t index = 0;
    size_t uindex = 0;
    size_t vindex = 0;
    Vec4d uiVec = 0.0;
    Vec4d viVec = 0.0;
    Vec4d uivi = 0.0;
    Vec4d u0Vec = 0.0;
    Vec4d u1Vec = 0.0;
    Vec4d u2Vec = 0.0;
    Vec4d u3Vec = 0.0;

    // Here the matrix is still full
    for (size_t col = 0; col < numCols; col++)
    {
        // orthogonalize current block
        norms[0] = 0.0;
        norms[1] = 0.0;

        // 0th entry
        udots[0] = 0.0;
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // 8 because thats the width
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[0] += ui * ui;
            udots[1] += ui * Q[uindex + 1];
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(0, udots[0]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[1] / udots[0] * Q[uindex]; // + 0
            Q[uindex + 2] -= udots[2] / udots[0] * Q[uindex];
            Q[uindex + 3] -= udots[3] / udots[0] * Q[uindex];
            uNorm = udots[0];
            uiVec = Q[uindex];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 1st entry
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[1] += ui * ui;
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(1, udots[1]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[2] / udots[1] * Q[uindex + 1];
            Q[uindex + 3] -= udots[3] / udots[1] * Q[uindex + 1];
            uNorm = udots[1];
            uiVec = Q[uindex + 1];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 2nd entry
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[2] += ui * ui;
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(2, udots[2]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[3] / udots[2] * Q[uindex + 2];
            uNorm = udots[2];
            uiVec = Q[uindex + 2];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 3rd entry
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[3] += ui * ui;
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(3, udots[3]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            // Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2], Q[uindex][3]}; // dont need this anymore
            uNorm = udots[3];
            uiVec = Q[uindex + 3];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 4th entry
        udots[4] = 0.0;                 // 4
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[4] += ui * ui;
            udots[5] += ui * Q[uindex + 1];
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(0, udots[4]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[5] / udots[4] * Q[uindex];
            Q[uindex + 2] -= udots[6] / udots[4] * Q[uindex];
            Q[uindex + 3] -= udots[7] / udots[4] * Q[uindex];
            uindex += 8;
        }

        // 5th entry
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[5] += ui * ui;
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(1, udots[5]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[6] / udots[5] * Q[uindex + 1];
            Q[uindex + 3] -= udots[7] / udots[5] * Q[uindex + 1];
            uindex += 8;
        }

        // 6th entry
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[6] += ui * ui;
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(2, udots[6]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[7] / udots[6] * Q[uindex + 2];
            uindex += 8;
        }

        // 7th entry, only have to do the norm
        uindex = 8 * col * numRows + 4;
        udots[7] = 0.0;
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[7] += ui * ui;
            uindex += 8;
        }
        norms[1].insert(3, udots[7]);

        // normalize current block
        index = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            u0Vec.load(&Q[index]);
            u1Vec.load(&Q[index + 4]);
            // index = col*2*numRows + 2*row;
            u0Vec /= sqrt(norms[0]);
            u1Vec /= sqrt(norms[1]);

            u0Vec.store(&Q[index]);
            u1Vec.store(&Q[index + 4]);
            index += 8;
        }
        // end normalize current block

        // orthogonalize following blocks
        for (size_t i = 0; i < 16; i++)
        {
            dotProducts[i] = 0;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // get dotProducts
            //initialize dotProducts;
            dotProducts[0] = 0.0;
            dotProducts[1] = 0.0;
            dotProducts[2] = 0.0;
            dotProducts[3] = 0.0;
            dotProducts[4] = 0.0;
            dotProducts[5] = 0.0;
            dotProducts[6] = 0.0;
            dotProducts[7] = 0.0;
            dotProducts[8] = 0.0;
            dotProducts[9] = 0.0;
            dotProducts[10] = 0.0;
            dotProducts[11] = 0.0;
            dotProducts[12] = 0.0;
            dotProducts[13] = 0.0;
            dotProducts[14] = 0.0;
            dotProducts[15] = 0.0;

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[0] += u0Vec * v0;
                dotProducts[1] += u1Vec * v0;
                dotProducts[2] += u2Vec * v0;
                dotProducts[3] += u3Vec * v0;

                dotProducts[4] += u0Vec * v1;
                dotProducts[5] += u1Vec * v1;
                dotProducts[6] += u2Vec * v1;
                dotProducts[7] += u3Vec * v1;
                uindex += 8;
                vindex += 8;
            }

            uindex = 8 * col * numRows + 4;
            vindex = 8 * folCol * numRows;
            // second uBlock
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[8] += u0Vec * v0;
                dotProducts[9] += u1Vec * v0;
                dotProducts[10] += u2Vec * v0;
                dotProducts[11] += u3Vec * v0;

                dotProducts[12] += u0Vec * v1;
                dotProducts[13] += u1Vec * v1;
                dotProducts[14] += u2Vec * v1;
                dotProducts[15] += u3Vec * v1;
                uindex += 8;
                vindex += 8;
            }

            // end dotproducts

            // linear combination

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                v0 -= dotProducts[0] * u0Vec + dotProducts[1] * u1Vec + dotProducts[2] * u2Vec + dotProducts[3] * u3Vec; // this should have -=
                v1 -= dotProducts[4] * u0Vec + dotProducts[5] * u1Vec + dotProducts[6] * u2Vec + dotProducts[7] * u3Vec;

                v0.store(&Q[vindex]);
                v1.store(&Q[vindex + 4]);
                uindex += 8;
                vindex += 8;
            }

            // second uBlock
            uindex = 8 * col * numRows + 4;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]);
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                v0 -= dotProducts[8] * u0Vec + dotProducts[9] * u1Vec + dotProducts[10] * u2Vec + dotProducts[11] * u3Vec;
                v1 -= dotProducts[12] * u0Vec + dotProducts[13] * u1Vec + dotProducts[14] * u2Vec + dotProducts[15] * u3Vec;

                v0.store(&Q[vindex]);
                v1.store(&Q[vindex + 4]);
                uindex += 8;
                vindex += 8;
            }

            // end linear combination
        }
        // end orthogonalize following blocks
    }
    delete[] norms;
    delete[] dotProducts;
    delete[] udots;
}

void qrFixedBlockOptimizedDouble(std::unique_ptr<double[]> &Q, size_t numRows, size_t numCols, size_t blockSize, size_t uBlockSize)
{
    Vec4d *norms = new Vec4d[2];
    double currentNorm = 0.0;
    Vec4d *dotProducts = new Vec4d[16];
    double *udots = new double[8];
    double ui = 0.0;
    Vec4d u = 0.0;
    Vec4d uNorm = 0.0;
    Vec4d v = 0.0;
    Vec4d uv = 0.0;
    Vec4d v0 = 0.0;
    Vec4d v1 = 0.0;
    size_t index = 0;
    size_t uindex = 0;
    size_t vindex = 0;
    Vec4d uiVec = 0.0;
    Vec4d viVec = 0.0;
    Vec4d uivi = 0.0;
    Vec4d u0Vec = 0.0;
    Vec4d u1Vec = 0.0;
    Vec4d u2Vec = 0.0;
    Vec4d u3Vec = 0.0;

    // Here the matrix is still full
    for (size_t col = 0; col < numCols; col++)
    {
        // orthogonalize current block
        norms[0] = 0.0;
        norms[1] = 0.0;

        // 0th entry
        udots[0] = 0.0;
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // 8 because thats the width
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[0] += ui * ui;
            udots[1] += ui * Q[uindex + 1];
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(0, udots[0]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[1] / udots[0] * Q[uindex]; // + 0
            Q[uindex + 2] -= udots[2] / udots[0] * Q[uindex];
            Q[uindex + 3] -= udots[3] / udots[0] * Q[uindex];
            uNorm = udots[0];
            uiVec = Q[uindex];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 1st entry
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[1] += ui * ui;
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(1, udots[1]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[2] / udots[1] * Q[uindex + 1];
            Q[uindex + 3] -= udots[3] / udots[1] * Q[uindex + 1];
            uNorm = udots[1];
            uiVec = Q[uindex + 1];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 2nd entry
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[2] += ui * ui;
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(2, udots[2]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[3] / udots[2] * Q[uindex + 2];
            uNorm = udots[2];
            uiVec = Q[uindex + 2];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 3rd entry
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[3] += ui * ui;
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(3, udots[3]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            // Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2], Q[uindex][3]}; // dont need this anymore
            uNorm = udots[3];
            uiVec = Q[uindex + 3];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 4th entry
        udots[4] = 0.0;                 // 4
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[4] += ui * ui;
            udots[5] += ui * Q[uindex + 1];
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(0, udots[4]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[5] / udots[4] * Q[uindex];
            Q[uindex + 2] -= udots[6] / udots[4] * Q[uindex];
            Q[uindex + 3] -= udots[7] / udots[4] * Q[uindex];
            uindex += 8;
        }

        // 5th entry
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[5] += ui * ui;
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(1, udots[5]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[6] / udots[5] * Q[uindex + 1];
            Q[uindex + 3] -= udots[7] / udots[5] * Q[uindex + 1];
            uindex += 8;
        }

        // 6th entry
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[6] += ui * ui;
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(2, udots[6]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[7] / udots[6] * Q[uindex + 2];
            uindex += 8;
        }

        // 7th entry, only have to do the norm
        uindex = 8 * col * numRows + 4;
        udots[7] = 0.0;
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[7] += ui * ui;
            uindex += 8;
        }
        norms[1].insert(3, udots[7]);

        // normalize current block
        index = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            u0Vec.load(&Q[index]);
            u1Vec.load(&Q[index + 4]);
            // index = col*2*numRows + 2*row;
            u0Vec /= sqrt(norms[0]);
            u1Vec /= sqrt(norms[1]);

            u0Vec.store(&Q[index]);
            u1Vec.store(&Q[index + 4]);
            index += 8;
        }
        // end normalize current block

        // orthogonalize following blocks
        for (size_t i = 0; i < 16; i++)
        {
            dotProducts[i] = 0;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // get dotProducts
            //initialize dotProducts;
            dotProducts[0] = 0.0;
            dotProducts[1] = 0.0;
            dotProducts[2] = 0.0;
            dotProducts[3] = 0.0;
            dotProducts[4] = 0.0;
            dotProducts[5] = 0.0;
            dotProducts[6] = 0.0;
            dotProducts[7] = 0.0;
            dotProducts[8] = 0.0;
            dotProducts[9] = 0.0;
            dotProducts[10] = 0.0;
            dotProducts[11] = 0.0;
            dotProducts[12] = 0.0;
            dotProducts[13] = 0.0;
            dotProducts[14] = 0.0;
            dotProducts[15] = 0.0;

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[0] += u0Vec * v0;
                dotProducts[1] += u1Vec * v0;
                dotProducts[2] += u2Vec * v0;
                dotProducts[3] += u3Vec * v0;

                dotProducts[4] += u0Vec * v1;
                dotProducts[5] += u1Vec * v1;
                dotProducts[6] += u2Vec * v1;
                dotProducts[7] += u3Vec * v1;
                uindex += 8;
                vindex += 8;
            }

            uindex = 8 * col * numRows + 4;
            vindex = 8 * folCol * numRows;
            // second uBlock
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[8] += u0Vec * v0;
                dotProducts[9] += u1Vec * v0;
                dotProducts[10] += u2Vec * v0;
                dotProducts[11] += u3Vec * v0;

                dotProducts[12] += u0Vec * v1;
                dotProducts[13] += u1Vec * v1;
                dotProducts[14] += u2Vec * v1;
                dotProducts[15] += u3Vec * v1;
                uindex += 8;
                vindex += 8;
            }

            // end dotproducts

            // linear combination

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                v0 -= dotProducts[0] * u0Vec + dotProducts[1] * u1Vec + dotProducts[2] * u2Vec + dotProducts[3] * u3Vec; // this should have -=
                v1 -= dotProducts[4] * u0Vec + dotProducts[5] * u1Vec + dotProducts[6] * u2Vec + dotProducts[7] * u3Vec;

                v0.store(&Q[vindex]);
                v1.store(&Q[vindex + 4]);
                uindex += 8;
                vindex += 8;
            }

            // second uBlock
            uindex = 8 * col * numRows + 4;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]);
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                v0 -= dotProducts[8] * u0Vec + dotProducts[9] * u1Vec + dotProducts[10] * u2Vec + dotProducts[11] * u3Vec;
                v1 -= dotProducts[12] * u0Vec + dotProducts[13] * u1Vec + dotProducts[14] * u2Vec + dotProducts[15] * u3Vec;

                v0.store(&Q[vindex]);
                v1.store(&Q[vindex + 4]);
                uindex += 8;
                vindex += 8;
            }

            // end linear combination
        }
        // end orthogonalize following blocks
    }
    delete[] norms;
    delete[] dotProducts;
    delete[] udots;
}

void qrFixedBlockOptimizedDoubleUnique(std::unique_ptr<double[]> &Q, size_t numRows, size_t numCols, size_t blockSize, size_t uBlockSize)
{
    Vec4d *norms = new Vec4d[2];
    double currentNorm = 0.0;
    Vec4d *dotProducts = new Vec4d[16];
    double *udots = new double[8];
    double ui = 0.0;
    Vec4d u = 0.0;
    Vec4d uNorm = 0.0;
    Vec4d v = 0.0;
    Vec4d uv = 0.0;
    Vec4d v0 = 0.0;
    Vec4d v1 = 0.0;
    size_t index = 0;
    size_t uindex = 0;
    size_t vindex = 0;
    Vec4d uiVec = 0.0;
    Vec4d viVec = 0.0;
    Vec4d uivi = 0.0;
    Vec4d u0Vec = 0.0;
    Vec4d u1Vec = 0.0;
    Vec4d u2Vec = 0.0;
    Vec4d u3Vec = 0.0;

    // Here the matrix is still full
    for (size_t col = 0; col < numCols; col++)
    {
        // orthogonalize current block
        norms[0] = 0.0;
        norms[1] = 0.0;

        // 0th entry
        udots[0] = 0.0;
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // 8 because thats the width
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[0] += ui * ui;
            udots[1] += ui * Q[uindex + 1];
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(0, udots[0]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[1] / udots[0] * Q[uindex]; // + 0
            Q[uindex + 2] -= udots[2] / udots[0] * Q[uindex];
            Q[uindex + 3] -= udots[3] / udots[0] * Q[uindex];
            uNorm = udots[0];
            uiVec = Q[uindex];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 1st entry
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[1] += ui * ui;
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(1, udots[1]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[2] / udots[1] * Q[uindex + 1];
            Q[uindex + 3] -= udots[3] / udots[1] * Q[uindex + 1];
            uNorm = udots[1];
            uiVec = Q[uindex + 1];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 2nd entry
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[2] += ui * ui;
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(2, udots[2]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[3] / udots[2] * Q[uindex + 2];
            uNorm = udots[2];
            uiVec = Q[uindex + 2];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 3rd entry
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[3] += ui * ui;
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(3, udots[3]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            // Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2], Q[uindex][3]}; // dont need this anymore
            uNorm = udots[3];
            uiVec = Q[uindex + 3];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 4th entry
        udots[4] = 0.0;                 // 4
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[4] += ui * ui;
            udots[5] += ui * Q[uindex + 1];
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(0, udots[4]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[5] / udots[4] * Q[uindex];
            Q[uindex + 2] -= udots[6] / udots[4] * Q[uindex];
            Q[uindex + 3] -= udots[7] / udots[4] * Q[uindex];
            uindex += 8;
        }

        // 5th entry
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[5] += ui * ui;
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(1, udots[5]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[6] / udots[5] * Q[uindex + 1];
            Q[uindex + 3] -= udots[7] / udots[5] * Q[uindex + 1];
            uindex += 8;
        }

        // 6th entry
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[6] += ui * ui;
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(2, udots[6]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[7] / udots[6] * Q[uindex + 2];
            uindex += 8;
        }

        // 7th entry, only have to do the norm
        uindex = 8 * col * numRows + 4;
        udots[7] = 0.0;
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[7] += ui * ui;
            uindex += 8;
        }
        norms[1].insert(3, udots[7]);

        // normalize current block
        index = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            u0Vec.load(&Q[index]);
            u1Vec.load(&Q[index + 4]);
            // index = col*2*numRows + 2*row;
            u0Vec /= sqrt(norms[0]);
            u1Vec /= sqrt(norms[1]);

            u0Vec.store(&Q[index]);
            u1Vec.store(&Q[index + 4]);
            index += 8;
        }
        // end normalize current block

        // orthogonalize following blocks
        for (size_t i = 0; i < 16; i++)
        {
            dotProducts[i] = 0;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // get dotProducts
            //initialize dotProducts;
            dotProducts[0] = 0.0;
            dotProducts[1] = 0.0;
            dotProducts[2] = 0.0;
            dotProducts[3] = 0.0;
            dotProducts[4] = 0.0;
            dotProducts[5] = 0.0;
            dotProducts[6] = 0.0;
            dotProducts[7] = 0.0;
            dotProducts[8] = 0.0;
            dotProducts[9] = 0.0;
            dotProducts[10] = 0.0;
            dotProducts[11] = 0.0;
            dotProducts[12] = 0.0;
            dotProducts[13] = 0.0;
            dotProducts[14] = 0.0;
            dotProducts[15] = 0.0;

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[0] += u0Vec * v0;
                dotProducts[1] += u1Vec * v0;
                dotProducts[2] += u2Vec * v0;
                dotProducts[3] += u3Vec * v0;

                dotProducts[4] += u0Vec * v1;
                dotProducts[5] += u1Vec * v1;
                dotProducts[6] += u2Vec * v1;
                dotProducts[7] += u3Vec * v1;
                uindex += 8;
                vindex += 8;
            }

            uindex = 8 * col * numRows + 4;
            vindex = 8 * folCol * numRows;
            // second uBlock
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                dotProducts[8] += u0Vec * v0;
                dotProducts[9] += u1Vec * v0;
                dotProducts[10] += u2Vec * v0;
                dotProducts[11] += u3Vec * v0;

                dotProducts[12] += u0Vec * v1;
                dotProducts[13] += u1Vec * v1;
                dotProducts[14] += u2Vec * v1;
                dotProducts[15] += u3Vec * v1;
                uindex += 8;
                vindex += 8;
            }

            // end dotproducts

            // linear combination

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                v0 -= dotProducts[0] * u0Vec + dotProducts[1] * u1Vec + dotProducts[2] * u2Vec + dotProducts[3] * u3Vec; // this should have -=
                v1 -= dotProducts[4] * u0Vec + dotProducts[5] * u1Vec + dotProducts[6] * u2Vec + dotProducts[7] * u3Vec;

                v0.store(&Q[vindex]);
                v1.store(&Q[vindex + 4]);
                uindex += 8;
                vindex += 8;
            }

            // second uBlock
            uindex = 8 * col * numRows + 4;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 1; // second u block
                // vindex = folCol*2*numRows + row*2; // first v block
                u.load(&Q[uindex]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]);
                // broadcast
                u0Vec = u[0];
                u1Vec = u[1];
                u2Vec = u[2];
                u3Vec = u[3];

                v0 -= dotProducts[8] * u0Vec + dotProducts[9] * u1Vec + dotProducts[10] * u2Vec + dotProducts[11] * u3Vec;
                v1 -= dotProducts[12] * u0Vec + dotProducts[13] * u1Vec + dotProducts[14] * u2Vec + dotProducts[15] * u3Vec;

                v0.store(&Q[vindex]);
                v1.store(&Q[vindex + 4]);
                uindex += 8;
                vindex += 8;
            }

            // end linear combination
        }
        // end orthogonalize following blocks
    }
    delete[] norms;
    delete[] dotProducts;
    delete[] udots;
}

void qrFixedBlockOptimizedDouble(std::shared_ptr<double[]> &Q, size_t numRows, size_t numCols, size_t blockSize = 2, size_t uBlockSize = 1)
{
    std::unique_ptr<Vec4d[]> norms(new Vec4d[2]);
    double currentNorm = 0.0;
    std::unique_ptr<Vec4d[]> dotProducts(new Vec4d[16]);
    std::unique_ptr<double[]> udots(new double[8]);
    double ui = 0.0;
    Vec4d u = 0.0;
    Vec4d u0 = 0.0;
    Vec4d u1 = 0.0;
    Vec4d uNorm = 0.0;
    Vec4d v = 0.0;
    Vec4d uv = 0.0;
    Vec4d v0 = 0.0;
    Vec4d v1 = 0.0;
    size_t index = 0;
    size_t uindex = 0;
    size_t vindex = 0;
    Vec4d uiVec = 0.0;
    Vec4d viVec = 0.0;
    Vec4d uivi = 0.0;
    Vec4d u0Vec = 0.0;
    Vec4d u1Vec = 0.0;
    Vec4d u2Vec = 0.0;
    Vec4d u3Vec = 0.0;
    Vec4d u00Vec = 0.0;
    Vec4d u01Vec = 0.0;
    Vec4d u02Vec = 0.0;
    Vec4d u03Vec = 0.0;
    Vec4d u10Vec = 0.0;
    Vec4d u11Vec = 0.0;
    Vec4d u12Vec = 0.0;
    Vec4d u13Vec = 0.0;

    // Here the matrix is still full
    for (size_t col = 0; col < numCols; col++)
    {
        // orthogonalize current block
        norms[0] = 0.0;
        norms[1] = 0.0;

        // 0th entry
        udots[0] = 0.0;
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // 8 because thats the width
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[0] += ui * ui;
            udots[1] += ui * Q[uindex + 1];
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(0, udots[0]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[1] / udots[0] * Q[uindex]; // + 0
            Q[uindex + 2] -= udots[2] / udots[0] * Q[uindex];
            Q[uindex + 3] -= udots[3] / udots[0] * Q[uindex];
            uNorm = udots[0];
            uiVec = Q[uindex];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 1st entry
        udots[1] = 0.0;
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[1] += ui * ui;
            udots[2] += ui * Q[uindex + 2];
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(1, udots[1]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[2] / udots[1] * Q[uindex + 1];
            Q[uindex + 3] -= udots[3] / udots[1] * Q[uindex + 1];
            uNorm = udots[1];
            uiVec = Q[uindex + 1];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 2nd entry
        udots[2] = 0.0;
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows + 0; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[2] += ui * ui;
            udots[3] += ui * Q[uindex + 3];
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(2, udots[2]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[3] / udots[2] * Q[uindex + 2];
            uNorm = udots[2];
            uiVec = Q[uindex + 2];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 3rd entry
        udots[3] = 0.0;
        udots[4] = 0.0;
        udots[5] = 0.0;
        udots[6] = 0.0;
        udots[7] = 0.0;
        uv = 0.0;
        uindex = 8 * col * numRows; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[3] += ui * ui;
            v.load(&Q[uindex + 4]);
            uv += ui * v;
            uv.store(&udots[4]);
            uindex += 8;
        }
        norms[0].insert(3, udots[3]);
        // linear combination
        uindex = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            // Q[uindex] = {Q[uindex][0], Q[uindex][1], Q[uindex][2], Q[uindex][3]}; // dont need this anymore
            uNorm = udots[3];
            uiVec = Q[uindex + 3];
            uv.load(&udots[4]);
            u.load(&Q[uindex + 4]);
            u -= uv / uNorm * uiVec;
            u.store(&Q[uindex + 4]);
            uindex += 8;
        }

        // 4th entry
        udots[4] = 0.0;                 // 4
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex];
            udots[4] += ui * ui;
            udots[5] += ui * Q[uindex + 1];
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(0, udots[4]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 1] -= udots[5] / udots[4] * Q[uindex];
            Q[uindex + 2] -= udots[6] / udots[4] * Q[uindex];
            Q[uindex + 3] -= udots[7] / udots[4] * Q[uindex];
            uindex += 8;
        }

        // 5th entry
        udots[5] = 0.0;                 // 5
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +4 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 1];
            udots[5] += ui * ui;
            udots[6] += ui * Q[uindex + 2];
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(1, udots[5]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 2] -= udots[6] / udots[5] * Q[uindex + 1];
            Q[uindex + 3] -= udots[7] / udots[5] * Q[uindex + 1];
            uindex += 8;
        }

        // 6th entry
        udots[6] = 0.0;                 // 6
        udots[7] = 0.0;                 // 7
        uindex = 8 * col * numRows + 4; // +1 would be the second 4-Block
        // dot products
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 2];
            udots[6] += ui * ui;
            udots[7] += ui * Q[uindex + 3];
            uindex += 8;
        }
        norms[1].insert(2, udots[6]);
        // linear combination
        uindex = 8 * col * numRows + 4;
        for (size_t row = 0; row < numRows; row++)
        {
            // avoid the insert
            Q[uindex + 3] -= udots[7] / udots[6] * Q[uindex + 2];
            uindex += 8;
        }

        // 7th entry, only have to do the norm
        uindex = 8 * col * numRows + 4;
        udots[7] = 0.0;
        for (size_t row = 0; row < numRows; row++)
        {
            ui = Q[uindex + 3];
            udots[7] += ui * ui;
            uindex += 8;
        }
        norms[1].insert(3, udots[7]);

        // normalize current block
        index = 8 * col * numRows;
        for (size_t row = 0; row < numRows; row++)
        {
            u0Vec.load(&Q[index]);
            u1Vec.load(&Q[index + 4]);
            // index = col*2*numRows + 2*row;
            u0Vec /= sqrt(norms[0]);
            u1Vec /= sqrt(norms[1]);

            u0Vec.store(&Q[index]);
            u1Vec.store(&Q[index + 4]);
            index += 8;
        }
        // end normalize current block

        // orthogonalize following blocks
        for (size_t i = 0; i < 16; i++)
        {
            dotProducts[i] = 0;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // get dotProducts
            //initialize dotProducts;
            dotProducts[0] = 0.0;
            dotProducts[1] = 0.0;
            dotProducts[2] = 0.0;
            dotProducts[3] = 0.0;
            dotProducts[4] = 0.0;
            dotProducts[5] = 0.0;
            dotProducts[6] = 0.0;
            dotProducts[7] = 0.0;
            dotProducts[8] = 0.0;
            dotProducts[9] = 0.0;
            dotProducts[10] = 0.0;
            dotProducts[11] = 0.0;
            dotProducts[12] = 0.0;
            dotProducts[13] = 0.0;
            dotProducts[14] = 0.0;
            dotProducts[15] = 0.0;

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u0.load(&Q[uindex]);
                u1.load(&Q[uindex + 4]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u00Vec = u0[0];
                u01Vec = u0[1];
                u02Vec = u0[2];
                u03Vec = u0[3];

                u10Vec = u1[0];
                u11Vec = u1[1];
                u12Vec = u1[2];
                u13Vec = u1[3];

                dotProducts[0] += u00Vec * v0;
                dotProducts[1] += u01Vec * v0;
                dotProducts[2] += u02Vec * v0;
                dotProducts[3] += u03Vec * v0;

                dotProducts[4] += u00Vec * v1;
                dotProducts[5] += u01Vec * v1;
                dotProducts[6] += u02Vec * v1;
                dotProducts[7] += u03Vec * v1;

                dotProducts[8] += u10Vec * v0;
                dotProducts[9] += u11Vec * v0;
                dotProducts[10] += u12Vec * v0;
                dotProducts[11] += u13Vec * v0;

                dotProducts[12] += u10Vec * v1;
                dotProducts[13] += u11Vec * v1;
                dotProducts[14] += u12Vec * v1;
                dotProducts[15] += u13Vec * v1;
                uindex += 8;
                vindex += 8;
            }

            // end dotproducts

            // linear combination

            // first uBlock
            uindex = 8 * col * numRows;
            vindex = 8 * folCol * numRows;
            for (size_t row = 0; row < numRows; row++)
            {
                // uindex = col*2*numRows + row*2 + 0; // first u block
                // vindex = folCol*2*numRows + row*2 + 0; // first v block
                u0.load(&Q[uindex]);
                u1.load(&Q[uindex + 4]);
                v0.load(&Q[vindex]);
                v1.load(&Q[vindex + 4]); // second vBlock
                // broadcast
                u00Vec = u0[0];
                u01Vec = u0[1];
                u02Vec = u0[2];
                u03Vec = u0[3];

                u10Vec = u1[0];
                u11Vec = u1[1];
                u12Vec = u1[2];
                u13Vec = u1[3];

                v0 -= dotProducts[0] * u00Vec + dotProducts[1] * u01Vec + dotProducts[2] * u02Vec + dotProducts[3] * u03Vec + dotProducts[8] * u10Vec + dotProducts[9] * u11Vec + dotProducts[10] * u12Vec + dotProducts[11] * u13Vec; // this should have -=
                v1 -= dotProducts[4] * u00Vec + dotProducts[5] * u01Vec + dotProducts[6] * u02Vec + dotProducts[7] * u03Vec + dotProducts[12] * u10Vec + dotProducts[13] * u11Vec + dotProducts[14] * u12Vec + dotProducts[15] * u13Vec;

                v0.store(&Q[vindex]);
                v1.store(&Q[vindex + 4]);
                uindex += 8;
                vindex += 8;
            }
            // end linear combination
        }
        // end orthogonalize following blocks
    }
}

void qrNaiveQNaive(std::shared_ptr<double[]> &Q, size_t numRows, size_t numCols)
{
    double u, v, unorm;
    double uv = 0.0;
    size_t uIndex = 0;
    size_t vIndex = 0;

    for (size_t col = 0; col < numCols; col++)
    {
        unorm = 0.0;
        for (size_t row = 0; row < numRows; row++)
        {
            uIndex = col * numRows + row;
            u = Q[uIndex];
            unorm += u * u;
        }
        for (size_t folCol = col + 1; folCol < numCols; folCol++)
        {
            // Dot product
            uv = 0.0;
            for (size_t row = 0; row < numRows; row++)
            {
                uIndex = col * numRows + row;
                vIndex = folCol * numRows + row;
                u = Q[uIndex];
                v = Q[vIndex];
                uv += u * v;
            }

            // linear combination
            for (size_t row = 0; row < numRows; row++)
            {
                uIndex = col * numRows + row;
                vIndex = folCol * numRows + row;
                u = Q[uIndex];
                Q[vIndex] -= uv / unorm * u;
            }
        }
        // normalize current column
        for (size_t row = 0; row < numRows; row++)
        {
            uIndex = col * numRows + row;
            Q[uIndex] /= std::sqrt(unorm);
        }
    }
}

#endif // DUNE_MOES_QRCOL_HH