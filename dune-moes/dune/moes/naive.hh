#ifndef DUNE_MOES_NAIVE_HH
#define DUNE_MOES_NAIVE_HH
#include <vector>
#include <cstdlib>
#include <cmath>
#include <dune/moes/vectorclass/vectorclass.h>

void printMatrix(const std::vector<std::vector<double>> &Q)
{
    std::cout << std::endl;
    for (size_t row = 0; row < Q[0].size(); row++)
    {
        for (size_t col = 0; col < Q.size(); col++)
        {
            std::cout << Q[col][row] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printVector(const std::vector<double> &v)
{
    std::cout << std::endl;
    for (size_t i = 0; i < v.size(); i++)
    {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

void fillMatrixRandom(std::vector<std::vector<double>> &Q)
{
    for (size_t col = 0; col < Q.size(); col++)
    {
        for (size_t row = 0; row < Q[0].size(); row++)
        {
            Q[col][row] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }
}

template <int N, int rhsWidth>
void fillMatrixRandomNative(double **Q)
{
    for (int col = 0; col < rhsWidth; col++)
    {
        for (int row = 0; row < N; row++)
        {
            Q[col][row] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }
}

void twoNorm(const std::vector<double> &v, double &norm)
{
    norm = 0.0;
    for (size_t i = 0; i < v.size(); i++)
    {
        norm += v[i] * v[i]; // fused mutliply add
    }
}

void euclidNorm(const std::vector<double> &v, double &norm)
{
    twoNorm(v, norm);
    norm = std::sqrt(norm);
}

void vecMul(const std::vector<double> &u, const std::vector<double> &v, double &uv)
{
    uv = 0.0;
    for (size_t i = 0; i < u.size(); i++)
    {
        uv += u[i] * v[i]; //fma
    }
}

void vecSub(const double &factor, const std::vector<double> &x, std::vector<double> &y)
{
    for (size_t i = 0; i < x.size(); i++)
    {
        y[i] -= factor * x[i]; //fma
    }
}

void getFactors(const std::vector<std::vector<double>> &Q, std::vector<double> &factors, const size_t &orthIndex)
{
    double uv = 0.0;
    double unorm = 0.0;
    twoNorm(Q[orthIndex], unorm);
    for (size_t col = orthIndex + 1; col < Q.size(); col++)
    {
        vecMul(Q[col], Q[orthIndex], uv);
        factors[col] = uv / unorm;
    }
}

void qr(std::vector<std::vector<double>> &Q, int tiling = 1000)
{
    size_t cols = Q.size();
    size_t rows = Q[0].size();
    double dotproduct = 0.0;
    double unorm = 0.0;
    for (size_t ucol = 0; ucol < cols; ucol++)
    {
        unorm = 0.0;
        for (size_t i = 0; i < rows; i++)
        {
            unorm += Q[ucol][i] * Q[ucol][i];
        }
        for (size_t vcol = ucol + 1; vcol < cols; vcol++)
        {
            dotproduct = 0.0;
            for (size_t row = 0; row < rows; row++)
            {
                dotproduct += Q[ucol][row] * Q[vcol][row];
            }
            dotproduct /= unorm;
            for (size_t row = 0; row < rows; row++)
            {
                Q[vcol][row] -= dotproduct * Q[ucol][row];
            }
        }
        for (size_t i = 0; i < rows; i++)
        {
            Q[ucol][i] /= std::sqrt(unorm);
        }
    }
}

void qrVN(std::vector<std::vector<double>> &Q, int tiling = 1000)
{
    size_t cols = Q.size();
    size_t rows = Q[0].size();
    if (cols % 4 != 0 || rows % 4 != 0)
    {
        std::cout << "qrV Error: Vector length and rhsWidth needs to be divisible by 4!" << std::endl;
        return;
    }
    double dotproduct = 0.0;
    double unorm = 0.0;
    Vec4d dotproductV = 0.0;
    Vec4d unormV = 0.0;
    Vec4d ui = 0.0;
    Vec4d vi = 0.0;

    // This has lots of strided access, which is probably why it is slower.
    // Runtime has great dependency on N (rows). Maybe use tiling to improve cache reuse.
    // Want to keep a big as possible part of u in cache for as long as possible and prevent striding.
    // cache = 8MB. double = 8B, therefore should load as close to 1million values as possible (each Vec4d) has 4 of those.
    for (size_t ucol = 0; ucol < cols; ucol++)
    {
        // zero the result of the dotproducts
        unormV = 0.0;
        for (size_t row = 0; row < rows; row += 4)
        {
            ui.load(&Q[ucol][row]);
            unormV += square(ui);
        }
        unorm = horizontal_add(unormV);
        unormV = unorm;
        // get norm of current u column and calculate u*v
        // This is where most calculation happen
        // in cache are: Q[ucol][row - row + x], cols * Q[vcol][row - row+x], (dotproductV, ui, vi) the last have very little impact
        // Basically cache usage = x + cols*x doubles.
        // Tiling: rows and columns should be restricted to fit in cache
        // u column should fill roughly half of the cache (to keep it between v columns)
        for (size_t vcol = ucol + 1; vcol < cols; vcol++)
        {
            dotproductV = 0.0;
            for (size_t row = 0; row < rows; row += 4)
            {
                ui.load(&Q[ucol][row]);
                vi.load(&Q[vcol][row]);
                //dotproductV[vcol] = mul_add(vi, ui, dotproductV[vcol]);
                dotproductV += vi * ui;
            }

            // horizontal add and broadcasting dot product
            // also create factor uv/|u|
            dotproduct = horizontal_add(dotproductV) / unorm;
            dotproductV = dotproduct; // broadcast
            // v = v - uv/|u| * u
            for (size_t row = 0; row < rows; row += 4)
            {
                ui.load(&Q[ucol][row]);
                vi.load(&Q[vcol][row]);
                vi = nmul_add(dotproductV[vcol], ui, vi);
                vi.store(&Q[vcol][row]);
            }
        }
        // normalize
        for (size_t row = 0; row < rows; row += 4)
        {
            ui.load(&Q[ucol][row]);
            ui /= sqrt(unormV);
            ui.store(&Q[ucol][row]);
        }
    }
}

void qrV(std::vector<std::vector<double>> &Q, int tiling = 1000)
{
    size_t cols = Q.size();
    size_t rows = Q[0].size();
    size_t colstep = 16;
    size_t rowstep = 2000;
    if (cols % 4 != 0 || rows % 4 != 0)
    {
        std::cout << "qrV Error: Vector length and rhsWidth needs to be divisible by 4!" << std::endl;
        return;
    }
    double dotproduct = 0.0;
    double unorm = 0.0;
    Vec4d dotproductV[cols];
    Vec4d unormV = 0.0;
    Vec4d ui = 0.0;
    Vec4d vi = 0.0;

    // This has lots of strided access, which is probably why it is slower.
    // Runtime has great dependency on N (rows). Maybe use tiling to improve cache reuse.
    // Want to keep a big as possible part of u in cache for as long as possible and prevent striding.
    // cache = 8MB. double = 8B, therefore should load as close to 1million values as possible (each Vec4d) has 4 of those.
    for (size_t ucol = 0; ucol < cols; ucol++)
    {
        // zero the result of the dotproducts
        unormV = 0.0;
        for (size_t vcol = ucol + 1; vcol < cols; vcol++)
        {
            dotproductV[vcol] = 0.0;
        }
        for (size_t row = 0; row < rows; row += 4)
        {
            ui.load(&Q[ucol][row]);
            unormV += square(ui);
        }
        unorm = horizontal_add(unormV);
        unormV = unorm;
        // get norm of current u column and calculate u*v
        // This is where most calculation happen
        // in cache are: Q[ucol][row - row + x], cols * Q[vcol][row - row+x], (dotproductV, ui, vi) the last have very little impact
        // Basically cache usage = x + cols*x doubles.
        // Tiling: rows and columns should be restricted to fit in cache
        // u column should fill roughly half of the cache (to keep it between v columns)
        for (size_t rowTile = 0; rowTile < rows; rowTile += tiling)
        {
            for (size_t vcol = ucol + 1; vcol < cols; vcol++)
            {
                for (size_t row = 0; row < tiling; row += 4)
                {
                    ui.load(&Q[ucol][rowTile + row]);
                    vi.load(&Q[vcol][rowTile + row]);
                    dotproductV[vcol] = mul_add(vi, ui, dotproductV[vcol]);
                }
            }
        }

        // horizontal add and broadcasting dot product
        // also create factor uv/|u|
        for (size_t vcol = ucol + 1; vcol < cols; vcol++)
        {
            dotproduct = horizontal_add(dotproductV[vcol]) / unorm;
            dotproductV[vcol] = dotproduct; // broadcast
        }

        // v = v - uv/|u| * u
        for (size_t row = 0; row < rows; row += 4)
        {
            ui.load(&Q[ucol][row]);
            // go through the columns
            for (size_t vcol = ucol + 1; vcol < cols; vcol++)
            {
                vi.load(&Q[vcol][row]);
                vi = nmul_add(dotproductV[vcol], ui, vi);
                vi.store(&Q[vcol][row]);
            }
            ui /= sqrt(unormV);
            ui.store(&Q[ucol][row]);
        }
    }
}

void qrPar(std::vector<std::vector<double>> &Q, int tiling = 1000)
{
    size_t cols = Q.size();
    size_t rows = Q[0].size();
    size_t vcolRoundStart = 0;
    double dotproduct = 0.0;
    std::vector<double> dotproductV(4, 0.0);
    double unorm = 0.0;
    for (size_t ucol = 0; ucol < cols; ucol++)
    {
        unorm = 0.0;
        for (size_t i = 0; i < rows; i++)
        {
            unorm += Q[ucol][i] * Q[ucol][i];
        }
        // treat the columns until the next 4th seperately
        for (size_t vcol = ucol + 1; (vcol % 4) != 0; vcol++)
        {
            dotproduct = 0.0;
            for (size_t row = 0; row < rows; row++)
            {
                dotproduct += Q[ucol][row] * Q[vcol][row];
            }
            dotproduct /= unorm;
            for (size_t row = 0; row < rows; row++)
            {
                Q[vcol][row] -= dotproduct * Q[ucol][row];
            }
            vcolRoundStart = vcol + 1;
        }

        for (size_t vcol = vcolRoundStart; vcol < cols; vcol += 4) // parallel problem
        {
            dotproductV[0] = 0.0;
            dotproductV[1] = 0.0;
            dotproductV[2] = 0.0;
            dotproductV[3] = 0.0;
            for (size_t row = 0; row < rows; row++)
            {
                dotproductV[0] += Q[ucol][row] * Q[vcol][row];
                dotproductV[1] += Q[ucol][row] * Q[vcol + 1][row];
                dotproductV[2] += Q[ucol][row] * Q[vcol + 2][row];
                dotproductV[3] += Q[ucol][row] * Q[vcol + 3][row];
            }
            dotproductV[0] /= unorm;
            dotproductV[1] /= unorm;
            dotproductV[2] /= unorm;
            dotproductV[3] /= unorm;
            for (size_t row = 0; row < rows; row++)
            {
                Q[vcol][row] -= dotproductV[0] * Q[ucol][row];
                Q[vcol + 1][row] -= dotproductV[1] * Q[ucol][row];
                Q[vcol + 2][row] -= dotproductV[2] * Q[ucol][row];
                Q[vcol + 3][row] -= dotproductV[3] * Q[ucol][row];
            }
        }
        for (size_t i = 0; i < rows; i++)
        {
            Q[ucol][i] /= std::sqrt(unorm);
        }
    }
}

void qrParV(std::vector<std::vector<double>> &Q, int tiling = 1000)
{
    const size_t loadwidth = 1;
    size_t cols = Q.size();
    size_t rows = Q[0].size();
    size_t vcolRoundStart = 0;
    Vec4d dotproduct = 0.0;
    double dotproductS = 0.0;
    std::vector<Vec4d> dotproductV(loadwidth, 0.0);
    double unorm = 0.0;
    Vec4d unormV = 0.0;
    Vec4d ui = 0.0;
    std::vector<Vec4d> viv(loadwidth, 0.0);
    Vec4d vi = 0.0;

    for (size_t ucol = 0; ucol < cols; ucol++)
    {
        unormV = 0.0;
        for (size_t row = 0; row < rows; row += 4)
        {
            ui.load(&Q[ucol][row]);
            unormV += square(ui);
        }
        unorm = horizontal_add(unormV);
        unormV = unorm;
        // treat the columns until the next 4th seperately
        vcolRoundStart = ucol; // this is just for the loadwith = 1 case
        for (size_t vcol = ucol + 1; (vcol % loadwidth) != 0; vcol++)
        {
            dotproduct = 0.0;
            for (size_t row = 0; row < rows; row += 4)
            {
                ui.load(&Q[ucol][row]);
                vi.load(&Q[vcol][row]);
                dotproduct += ui * vi;
            }
            dotproductS = horizontal_add(dotproduct) / unorm;
            dotproduct = dotproductS; // broadcast
            for (size_t row = 0; row < rows; row += 4)
            {
                ui.load(&Q[ucol][row]);
                vi.load(&Q[vcol][row]);
                vi = nmul_add(dotproduct, ui, vi);
                vi.store(&Q[vcol][row]);
            }
            vcolRoundStart = vcol + 1;
        }

        // reduce loading of ui by using it for more columns.
        for (size_t vcol = vcolRoundStart; vcol < cols; vcol += loadwidth)
        {
            for (size_t i = 0; i < loadwidth; i++)
            {
                dotproductV[i] = 0.0;
            }
            for (size_t row = 0; row < rows; row += 4)
            {
                ui.load(&Q[ucol][row]);
                for (size_t i = 0; i < loadwidth; i++)
                {
                    viv[i].load(&Q[vcol + i][row]);
                }
                for (size_t i = 0; i < loadwidth; i++)
                {
                    dotproductV[i] += ui * viv[i];
                }
            }
            for (size_t i = 0; i < loadwidth; i++)
            {
                dotproductV[i] = horizontal_add(dotproductV[i]) / unorm;
            }
            for (size_t row = 0; row < rows; row += 4)
            {
                ui.load(&Q[ucol][row]);
                for (size_t i = 0; i < loadwidth; i++)
                {
                    viv[i].load(&Q[vcol + i][row]);
                }
                for (size_t i = 0; i < loadwidth; i++)
                {
                    viv[i] = nmul_add(dotproductV[i], ui, viv[i]);
                }
                for (size_t i = 0; i < loadwidth; i++)
                {
                    viv[i].store(&Q[vcol + i][row]);
                }
            }
        }
        for (size_t row = 0; row < rows; row += 4)
        {
            ui.load(&Q[ucol][row]);
            ui /= sqrt(unormV);
            ui.store(&Q[ucol][row]);
        }
    }
}

/* 
    Inefficient naive implementation of the modified Gram Schmidt algorithm, only for comparison.
    in Blocks it works
    between blocks it does not work (Problem with the test or with the algorithm?)
*/
void qrNaive(double *Q, size_t numRows, size_t numCols, size_t blockSize, size_t ublockSize)
{
    double u, v, unorm;
    double uv = 0.0;
    size_t uIndex = 0;
    size_t vIndex = 0;

    for (size_t col = 0; col < numCols; col++)
    {
        for (size_t iBI = 0; iBI < 8; iBI++)
        {
            // calculate unorm
            unorm = 0.0;
            for (size_t row = 0; row < numRows; row++)
            {
                uIndex = 8 * col * numRows + 8 * row + iBI;
                u = Q[uIndex];
                unorm += u * u;
            }

            for (size_t iBIfol = iBI + 1; iBIfol < 8; iBIfol++)
            {
                /* orthogonalize rest of block */
                uv = 0.0;

                // dot product
                for (size_t row = 0; row < numRows; row++)
                {
                    uIndex = 8 * col * numRows + 8 * row + iBI;
                    vIndex = 8 * col * numRows + 8 * row + iBIfol;
                    u = Q[uIndex];
                    v = Q[vIndex];
                    uv += u * v;
                }

                // linear combination
                for (size_t row = 0; row < numRows; row++)
                {
                    uIndex = 8 * col * numRows + 8 * row + iBI;
                    vIndex = 8 * col * numRows + 8 * row + iBIfol;
                    u = Q[uIndex];
                    Q[vIndex] -= uv / unorm * u;
                }
            }
            for (size_t folCol = col + 1; folCol < numCols; folCol++)
            {
                /* orthogonalize other blocks */
                for (size_t iBIfolCol = 0; iBIfolCol < 8; iBIfolCol++)
                {
                    /* orthogonalize columns in other blocks */
                    uv = 0.0;
                    // dot product
                    for (size_t row = 0; row < numRows; row++)
                    {
                        uIndex = 8 * col * numRows + 8 * row + iBI;
                        vIndex = 8 * folCol * numRows + 8 * row + iBIfolCol;
                        u = Q[uIndex];
                        v = Q[vIndex];
                        uv += u * v;
                    }

                    // linear combination
                    for (size_t row = 0; row < numRows; row++)
                    {
                        uIndex = 8 * col * numRows + 8 * row + iBI;
                        vIndex = 8 * folCol * numRows + 8 * row + iBIfolCol;
                        u = Q[uIndex];
                        Q[vIndex] -= uv / unorm * u;
                    }
                }
            }
            // normalize current column
            for (size_t row = 0; row < numRows; row++)
            {
                uIndex = 8 * col * numRows + 8 * row + iBI;
                Q[uIndex] /= std::sqrt(unorm);
            }
        }
    }
}

/* 
    Inefficient naive implementation of the modified Gram Schmidt algorithm, only for comparison.
    no blocks
*/
void gramSchmidtNaive(double *Q, size_t numRows, size_t numCols)
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

void getFactorsSingleFunction(const std::vector<std::vector<double>> &Q, std::vector<double> &factors, const size_t &orthIndex)
{
    double uv = 0.0;
    double unorm = 0.0;
    const std::vector<double> *u = &Q[orthIndex]; // saves on lookup (I think)
    const std::vector<double> *v = &Q[orthIndex + 1];
    for (size_t i = 0; i < (*u).size(); i++)
    {
        unorm += (*u)[i] * (*u)[i]; // fused mutliply add
    }
    for (size_t col = orthIndex + 1; col < Q.size(); col++)
    {
        uv = 0.0;
        v = &Q[col];
        for (size_t i = 0; i < (*u).size(); i++) // this block is responsible for basically all runtime, why so slow?
        {
            uv += (*u)[i] * (*v)[i]; //fma // eher fml
        }
        factors[col] = uv / unorm;
    }
}

template <size_t N, size_t rhsWidth>
void getFactorsSingleFunctionNative(double **Q, double *factors, const size_t &orthIndex)
{
    double uv = 0.0;
    double unorm = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        unorm += Q[orthIndex][i] * Q[orthIndex][i]; // fused mutliply add
    }
    for (size_t col = orthIndex + 1; col < rhsWidth; col++)
    {
        uv = 0.0;
        for (size_t i = 0; i < N; i++) // this block is responsible for basically all runtime, why so slow?
        {
            uv += Q[orthIndex][i] * Q[col][i]; //fma // eher fml
        }
        factors[col] = uv / unorm;
    }
}

bool checkOrthogonality(const std::vector<std::vector<double>> &Q, const double &tolerance)
{
    double uv = 0.0;
    for (size_t col = 0; col < Q.size(); col++)
    {
        for (size_t folcol = col + 1; folcol < Q.size(); folcol++)
        {
            vecMul(Q[col], Q[folcol], uv);
            if (std::abs(uv) > tolerance)
            {
                std::cout << "Orthogonality violation!" << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool checkNorm(const std::vector<std::vector<double>> &Q, const double &tolerance)
{
    double unorm = 0.0;
    for (size_t col = 0; col < Q.size(); col++)
    {
        euclidNorm(Q[col], unorm);
        if (std::abs(unorm - 1.0) > tolerance)
        {
            std::cout << "Norm violation!" << std::endl;
            std::cout << "Norm is " << unorm << " should be: 1" << std::endl;
            return false;
        }
    }
    return true;
}

void checkOrthoNormality(const std::vector<std::vector<double>> &Q, const double &tolerance)
{
    bool success = true;
    if (!checkNorm(Q, tolerance))
    {
        success = false;
    }
    if (!checkOrthogonality(Q, tolerance))
    {
        success = false;
    }
    if (success)
    {
        std::cout << "Successfully passed checks!" << std::endl;
    }
    else
    {
        std::cout << "Checks failed!" << std::endl;
    }
}

#endif // DUNE_MOES_NAIVE_HH
