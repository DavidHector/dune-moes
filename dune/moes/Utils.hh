#ifndef DUNE_MOES_UTILS_HH
#define DUNE_MOES_UTILS_HH

#include <dune/moes/vectorclass/vectorclass.h>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/common/scalarmatrixview.hh>
/**
 * @brief Solves L x = b
 * 
 * @param b Multivector to go in (Qin)
 * @param x Multivector to go out (Qout)
 * @param Lp 
 * @param Lj 
 * @param Lx 
 */
void solveL(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const int64_t *Lp, const int64_t *Lj, const double *Lx, const int64_t n_row, const size_t qCols)
{
    // Dont have to initialize x, because every value will be written
    const size_t qSize = n_row * qCols * 8;
    Vec4d LEntry, bFirst, bSecond, xFirst, xSecond;
    size_t bIndex, xIndex, LCol;
    // iterate over the rows of L
    for (int64_t row = 0; row < n_row; row++)
    {
        // Not really sure what is in Lp -> Index ranges for the values in the row
        // std::cout << "Lp[" << row << "] = " << Lp[row] << std::endl;
        // Multiply matrix rows with columns of Q
        for (size_t qCol = 0; qCol < qCols; qCol++)
        {
            xFirst = 0.0;
            xSecond = 0.0;
            bIndex = 8 * qCol * n_row + 8 * row;
            // Go through the previous (solved entries) until col = row, where the right side comes into play
            // If we can trust (have to check) that columns are in order, then we can do the back insertion via right-side writing
            for (int64_t j = Lp[row]; j < Lp[row + 1] - 1; j++) // have to check if index starts with 1 or 0
            {
                LCol = Lj[j];
                LEntry = Lx[j]; // broadcast
                xIndex = 8 * qCol * n_row + 8 * LCol;
                bFirst.load(&x[xIndex]);
                bSecond.load(&x[xIndex + 4]);
                xFirst -= LEntry * bFirst; // This here is for the non-diagonal elements that have to use the previous solutions
                xSecond -= LEntry * bSecond;
            }
            // Diagonal entry
            LCol = Lj[Lp[row + 1] - 1];   // this is for the entry corresponding right side b (diagonal element)
            LEntry = Lx[Lp[row + 1] - 1]; // This should always be 1 (Since L is a unit lower triangular matrix)
            // Checked this: It does work as intended, no non-1 diagonals were found that could cause the nans
            /*
            if (LEntry[0] != 1.0)
            {
                std::cout << "solveL: Encountered non-1 diagonal: " << LEntry[0] << std::endl;
            }
            */
            bFirst.load(&b[bIndex]);
            bSecond.load(&b[bIndex + 4]);
            xFirst += bFirst;
            xFirst /= LEntry;
            xSecond += bSecond;
            xSecond /= LEntry;
            xFirst.store(&x[bIndex]);
            xSecond.store(&x[bIndex + 4]);
        }
    }
}

/**
 * @brief Solves U x = b, should basically be like solveL, just in reverse, U is in CCS (Compressed Column Storage) format.
 * 
 * @param b Multivector to go in 
 * @param x Multivector to go out
 * @param Up 
 * @param Ui 
 * @param Ux 
 */
void solveUQ(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const int64_t *Q, const int64_t *Up, const int64_t *Ui, const double *Ux, const int64_t n_col, const size_t qCols)
{
    // initialize x
    for (size_t i = 0; i < 8 * n_col * qCols; i++)
    {
        x[i] = 0.0;
    }
    Vec4d xFirst, xSecond, UEntry, bFirst, bSecond, tmpFirst, tmpSecond;
    size_t bIndex, bQIndex, xIndex, URow;
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        // Go from right to left and bottom to top
        for (int64_t col = n_col - 1; col >= 0; col--)
        {
            // Go from bottom to top
            // Diagonal Entry
            URow = Ui[Up[col + 1] - 1];
            UEntry = Ux[Up[col + 1] - 1];
            // TODO: First solveU then apply PivotQ
            // Checked this: It does work as intended, no small diagonals were found that could cause the nans
            /*
            if (std::abs(UEntry[0]) < 0.00005)
            {
                std::cout << "solveU: Encountered very small diagonal: " << UEntry[0] << std::endl;
            }
            */

            bIndex = 8 * n_col * qCol + 8 * URow;
            bQIndex = 8 * n_col * qCol + 8 * Q[URow];
            bFirst.load(&b[bIndex]);
            bSecond.load(&b[bIndex + 4]);
            xFirst.load(&x[bQIndex]);
            xSecond.load(&x[bQIndex + 4]);
            xFirst += bFirst;
            xSecond += bSecond;
            xFirst /= UEntry;
            xSecond /= UEntry;
            xFirst.store(&x[bQIndex]);
            xSecond.store(&x[bQIndex + 4]);
            for (int64_t i = Up[col + 1] - 2; i >= Up[col]; i--)
            {
                URow = Ui[i];
                UEntry = Ux[i];
                xIndex = 8 * n_col * qCol + 8 * Q[URow];
                tmpFirst.load(&x[xIndex]);
                tmpSecond.load(&x[xIndex + 4]);
                tmpFirst -= UEntry * xFirst;
                tmpSecond -= UEntry * xSecond;
                tmpFirst.store(&x[xIndex]);
                tmpSecond.store(&x[xIndex + 4]);
            }
        }
    }
}

/**
 * @brief Applies the pivot x = Pb with P being a vector of row indices as provided by Umfpack
 * 
 * @param b Mutlivector to go in
 * @param x Multivector to go out
 * @param P Pivot vector
 * @param N Length of the Pivot vector/Multivectors
 * @param qCols Width of the Multivectors
 */
void applyPivotP(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const int64_t *P, const size_t N, const size_t qCols)
{
    Vec4d bFirst, bSecond;
    size_t bIndex, xIndex;
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        for (size_t i = 0; i < N; i++)
        {
            /* load old vector at correct indices and write into new vector at i */
            bIndex = 8 * qCol * N + 8 * P[i];
            xIndex = 8 * qCol * N + 8 * i; // This can be much optimized
            bFirst.load(&b[bIndex]);
            bSecond.load(&b[bIndex + 4]);
            bFirst.store(&x[xIndex]);
            bSecond.store(&x[xIndex + 4]);
        }
    }
}

/**
 * @brief Applies the pivot x = Qb with Q being a vector of column indices as provided by Umfpack
 * 
 * @param b Mutlivector to go in
 * @param x Multivector to go out
 * @param Q Pivot vector
 * @param N Length of the Pivot vector/Multivectors
 * @param qCols Width of the Multivectors
 */

void applyPivotQ(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const int64_t *Q, const size_t N, const size_t qCols)
{
    Vec4d bFirst, bSecond;
    size_t bIndex, xIndex;
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        for (size_t i = 0; i < N; i++)
        {
            xIndex = 8 * qCol * N + 8 * Q[i];
            bIndex = 8 * qCol * N + 8 * i; // This can be much optimized
            bFirst.load(&b[bIndex]);
            bSecond.load(&b[bIndex + 4]);
            bFirst.store(&x[xIndex]);
            bSecond.store(&x[xIndex + 4]);
        }
    }
}

/**
 * @brief Normalizes x
 * 
 * @param x Multivector to normalize
 * @param N Length of the Multivector
 * @param qCols rhsWidth/8 (number of columns)
 */
void normalize(std::shared_ptr<double[]> &x, const size_t N, const size_t qCols)
{
    Vec4d xFirst, xSecond, normFirst, normSecond, oneVec;
    size_t xIndex;
    oneVec = 1.0; // broadcast for efficent division later
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        normFirst = 0.0;
        normSecond = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            xIndex = 8 * qCol * N + 8 * i;
            xFirst.load(&x[xIndex]);
            xSecond.load(&x[xIndex + 4]);
            normFirst += square(xFirst);
            normSecond += square(xSecond);
        }
        normFirst = oneVec / sqrt(normFirst);
        normSecond = oneVec / sqrt(normSecond);
        for (size_t i = 0; i < N; i++)
        {
            xIndex = 8 * qCol * N + 8 * i;
            xFirst.load(&x[xIndex]);
            xSecond.load(&x[xIndex + 4]);
            xFirst *= normFirst;
            xSecond *= normSecond;
            xFirst.store(&x[xIndex]);
            xSecond.store(&x[xIndex + 4]);
        }
    }
}
void applyScalingMult(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const double *Rs, const int64_t *P, const size_t N, const size_t qCols)
{
    Vec4d REntry, xFirst, xSecond;
    size_t xIndex = 0;
    size_t bIndex;
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        for (size_t row = 0; row < N; row++)
        {
            REntry = Rs[P[row]]; // broadcast
            // xIndex = 8 * qCol * N + 8 * row;
            bIndex = 8 * qCol * N + 8 * P[row];
            xFirst.load(&b[bIndex]);
            xSecond.load(&b[bIndex + 4]);
            xFirst *= REntry;
            xSecond *= REntry;
            xFirst.store(&x[xIndex]);
            xSecond.store(&x[xIndex + 4]);
            xIndex += 8;
        }
    }
};
void applyScalingInverse(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const double *Rs, const int64_t *P, const size_t N, const size_t qCols)
{
    Vec4d REntry, xFirst, xSecond;
    size_t xIndex = 0;
    size_t bIndex;
    for (size_t qCol = 0; qCol < qCols; qCol++)
    {
        for (size_t row = 0; row < N; row++)
        {
            REntry = Rs[P[row]]; // broadcast
            // xIndex = 8 * qCol * N + 8 * row;
            bIndex = 8 * qCol * N + 8 * P[row];
            xFirst.load(&b[bIndex]);
            xSecond.load(&b[bIndex + 4]);
            xFirst /= REntry;
            xSecond /= REntry;
            xFirst.store(&x[xIndex]);
            xSecond.store(&x[xIndex + 4]);
            xIndex += 8;
        }
    }
};

/**
 * @brief Applies the scaling vector R, which represents a diagonal matrix
 * 
 * @param b 
 * @param x 
 * @param Rs 
 * @param N 
 * @param qCols 
 * @param do_recip 
 */
void applyScalingP(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const double *Rs, const int64_t *P, const size_t N, const size_t qCols, const int do_recip)
{
    if (do_recip == 1)
    {
        applyScalingMult(b, x, Rs, P, N, qCols);
    }
    else
    {
        applyScalingInverse(b, x, Rs, P, N, qCols);
    }
}

template <class B, class Alloc>
void setupNeumannSparsityPattern(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc> Matrix;
    A.setSize(N, N, N);
    A.setBuildMode(Matrix::row_wise);
    for (typename Dune::BCRSMatrix<B, Alloc>::CreateIterator i = A.createbegin(); i != A.createend(); ++i)
    {
        i.insert(i.index());
    }
}

template <class B, class Alloc>
void setupNeumannMat(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc>::field_type FieldType;
    setupNeumannSparsityPattern(A, N);
    auto startrows = A.begin()->operator[](A.begin().index()).rows;
    for (size_t startrow = 0; startrow < startrows; startrow++)
    {
        A.begin()->operator[](A.begin().index())[startrow][startrow] = 0.0;
    }

    for (typename Dune::BCRSMatrix<B, Alloc>::RowIterator i = A.begin()++; i != A.end()--; ++i)
    {
        auto bRows = i->operator[](i.index()).rows;
        for (size_t brow = 0; brow < bRows; brow++)
        {
            i->operator[](i.index())[brow][brow] = 1.0;
        }

        //i->operator[](i.index()) = 1.0; // cant just assign 1 to the entire block, only the diagonal
    }
    auto stoprows = A.end()->operator[](A.end().index()).rows;
    for (size_t stoprow = 0; stoprow < stoprows; stoprow++)
    {
        A.end()->operator[](A.end().index())[stoprow][stoprow] = 0.0;
    }
}

template <class B, class Alloc>
void setupLaplacianSparsityPattern(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc> Matrix;
    A.setSize(N * N, N * N, N * N * 5);
    A.setBuildMode(Matrix::row_wise);

    for (typename Dune::BCRSMatrix<B, Alloc>::CreateIterator i = A.createbegin(); i != A.createend(); ++i)
    {
        int x = i.index() % N; // x coordinate in the 2d field
        int y = i.index() / N; // y coordinate in the 2d field

        if (y > 0)
            // insert lower neighbour
            i.insert(i.index() - N);
        if (x > 0)
            // insert left neighbour
            i.insert(i.index() - 1);

        // insert diagonal value
        i.insert(i.index());

        if (x < N - 1)
            //insert right neighbour
            i.insert(i.index() + 1);
        if (y < N - 1)
            // insert upper neighbour
            i.insert(i.index() + N);
    }
}

template <class B, class Alloc>
void setupIdentityWithLaplacianSparsityPattern(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc>::field_type FieldType;
    setupLaplacianSparsityPattern(A, std::sqrt(N));
    for (typename Dune::BCRSMatrix<B, Alloc>::RowIterator i = A.begin(); i != A.end(); ++i)
    {
        auto bRows = i->operator[](i.index()).rows;
        for (size_t brow = 0; brow < bRows; brow++)
        {
            i->operator[](i.index())[brow][brow] = 1.0;
        }

        //i->operator[](i.index()) = 1.0; // cant just assign 1 to the entire block, only the diagonal
    }
}

template <class B, class Alloc>
void setupLaplacianWithBoundary(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc>::field_type FieldType;
    size_t neighborCount;

    setupLaplacianSparsityPattern(A, N);

    B diagonal(static_cast<FieldType>(0)), bone(static_cast<FieldType>(0));

    auto setDiagonal = [](auto &&scalarOrMatrix, const auto &value) {
        auto &&matrix = Dune::Impl::asMatrix(scalarOrMatrix);
        for (auto rowIt = matrix.begin(); rowIt != matrix.end(); ++rowIt)
            (*rowIt)[rowIt.index()] = value;
    };

    setDiagonal(diagonal, 4.0);
    setDiagonal(bone, -1.0);

    for (typename Dune::BCRSMatrix<B, Alloc>::RowIterator i = A.begin(); i != A.end(); ++i)
    {
        int x = i.index() % N; // x coordinate in the 2d field
        int y = i.index() / N; // y coordinate in the 2d field

        neighborCount = 0;

        if (y > 0)
        {
            i->operator[](i.index() - N) = bone;
            neighborCount++;
        }
        if (y < N - 1)
        {
            i->operator[](i.index() + N) = bone;
            neighborCount++;
        }

        if (x > 0)
        {
            i->operator[](i.index() - 1) = bone;
            neighborCount++;
        }

        if (x < N - 1)
        {
            i->operator[](i.index() + 1) = bone;
            neighborCount++;
        }
        setDiagonal(diagonal, neighborCount);
        i->operator[](i.index()) = diagonal;
    }
}

// Laplacian without the boundary, i.e. A = DLD, if L is the Laplacian from above and D are diagonal matrices, with 0 on the boundary
template <class B, class Alloc>
void setupLaplacianWithoutBoundary(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc>::field_type FieldType;

    setupLaplacianSparsityPattern(A, N);

    B diagonal(static_cast<FieldType>(0)), bone(static_cast<FieldType>(0));

    auto setDiagonal = [](auto &&scalarOrMatrix, const auto &value) {
        auto &&matrix = Dune::Impl::asMatrix(scalarOrMatrix);
        for (auto rowIt = matrix.begin(); rowIt != matrix.end(); ++rowIt)
            (*rowIt)[rowIt.index()] = value;
    };

    setDiagonal(diagonal, 4.0);
    setDiagonal(bone, -1.0);
    // Have to construct it differently, right now, there are too many -1 entries left over
    // Probably first make a pass over the entire matrix with the all neighbor condition
    // Then another pass with the zeroes
    // Or maybe do the multiplication DAD^T = DAD
    for (typename Dune::BCRSMatrix<B, Alloc>::RowIterator i = A.begin(); i != A.end(); ++i)
    {
        int x = i.index() % N; // x coordinate in the 2d field
        int y = i.index() / N; // y coordinate in the 2d field

        if ((y > 0) && (y < N - 1) && (x > 0) && (x < N - 1))
        {
            setDiagonal(diagonal, 4.0);
            i->operator[](i.index() - N) = bone;
            i->operator[](i.index() + N) = bone;
            i->operator[](i.index() - 1) = bone;
            i->operator[](i.index()) = diagonal;
            i->operator[](i.index() + 1) = bone;
        }
    }

    bool boundary;
    setDiagonal(diagonal, 0.0);
    // There are too many -1 left
    for (typename Dune::BCRSMatrix<B, Alloc>::RowIterator i = A.begin(); i != A.end(); ++i)
    {
        int x = i.index() % N; // x coordinate in the 2d field
        int y = i.index() / N; // y coordinate in the 2d field
        boundary = !((y > 0) && (y < N - 1) && (x > 0) && (x < N - 1));
        if (boundary)
        {
            // Set entire row to zero
            if (y > 0)
            {
                i->operator[](i.index() - N) = diagonal;
            }
            if (y < N - 1)
            {
                i->operator[](i.index() + N) = diagonal;
            }

            if (x > 0)
            {
                i->operator[](i.index() - 1) = diagonal;
            }

            if (x < N - 1)
            {
                i->operator[](i.index() + 1) = diagonal;
            }
            i->operator[](i.index()) = diagonal;
            // Set entire column to Zero
            // i.index() is both the column and the row
            // I have to go through every row, then iterate through the columns, if they have index i.index(), then set it to 0
            for (auto row = A.begin(); row != A.end(); row++)
            {
                for (auto col = (*row).begin(); col != (*row).end(); col++)
                {
                    if (col.index() == i.index())
                    {
                        row->operator[](i.index()) = diagonal;
                    }
                }
            }
        }
    }
}

template <typename MAT>
void printBCRS(const MAT &A)
{
    for (auto rowIterator = A.begin(); rowIterator != A.end(); rowIterator++)
    {
        for (auto colIterator = (*rowIterator).begin(); colIterator != (*rowIterator).end(); colIterator++)
        {
            std::cout << *colIterator << "\t";
        }
        std::cout << std::endl;
    }
}

template <typename VEC>
double printNormVec(const VEC &v)
{
    double norm = 0.0;
    for (size_t i = 0; i < v.N(); i++)
    {
        norm += v[i][0] * v[i][0];
    }
    return std::sqrt(norm);
}

template <typename VEC>
void normalizeVec(VEC &v)
{
    double norm = 0.0;
    for (size_t i = 0; i < v.N(); i++)
    {
        norm += v[i][0] * v[i][0];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < v.N(); i++)
    {
        v[i] = v[i][0] / norm;
    }
}

template <class B, class Alloc>
void setupSparsityPattern(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc> Matrix;
    A.setSize(N * N, N * N, N * N * 5);
    A.setBuildMode(Matrix::row_wise);

    for (typename Dune::BCRSMatrix<B, Alloc>::CreateIterator i = A.createbegin(); i != A.createend(); ++i)
    {
        int x = i.index() % N; // x coordinate in the 2d field
        int y = i.index() / N; // y coordinate in the 2d field

        if (y > 0)
            // insert lower neighbour
            i.insert(i.index() - N);
        if (x > 0)
            // insert left neighbour
            i.insert(i.index() - 1);

        // insert diagonal value
        i.insert(i.index());

        if (x < N - 1)
            //insert right neighbour
            i.insert(i.index() + 1);
        if (y < N - 1)
            // insert upper neighbour
            i.insert(i.index() + N);
    }
}

template <class B, class Alloc>
void setupLaplacian(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc>::field_type FieldType;

    setupSparsityPattern(A, N);

    B diagonal(static_cast<FieldType>(0)), bone(static_cast<FieldType>(0));

    auto setDiagonal = [](auto &&scalarOrMatrix, const auto &value) {
        auto &&matrix = Dune::Impl::asMatrix(scalarOrMatrix);
        for (auto rowIt = matrix.begin(); rowIt != matrix.end(); ++rowIt)
            (*rowIt)[rowIt.index()] = value;
    };

    setDiagonal(diagonal, 4.0);
    setDiagonal(bone, -1.0);

    for (typename Dune::BCRSMatrix<B, Alloc>::RowIterator i = A.begin(); i != A.end(); ++i)
    {
        int x = i.index() % N; // x coordinate in the 2d field
        int y = i.index() / N; // y coordinate in the 2d field

        /*    if(x==0 || x==N-1 || y==0||y==N-1){

       i->operator[](i.index())=1.0;

       if(y>0)
       i->operator[](i.index()-N)=0;

       if(y<N-1)
       i->operator[](i.index()+N)=0.0;

       if(x>0)
       i->operator[](i.index()-1)=0.0;

       if(x < N-1)
       i->operator[](i.index()+1)=0.0;

       }else*/
        {

            i->operator[](i.index()) = diagonal;

            if (y > 0)
                i->operator[](i.index() - N) = bone;

            if (y < N - 1)
                i->operator[](i.index() + N) = bone;

            if (x > 0)
                i->operator[](i.index() - 1) = bone;

            if (x < N - 1)
                i->operator[](i.index() + 1) = bone;
        }
    }
}

template <class B, class Alloc>
void setupIdentitySparsityPattern(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc> Matrix;
    A.setSize(N, N, N);
    A.setBuildMode(Matrix::row_wise);
    for (typename Dune::BCRSMatrix<B, Alloc>::CreateIterator i = A.createbegin(); i != A.createend(); ++i)
    {
        i.insert(i.index());
    }
}

template <class B, class Alloc>
void setupIdentity(Dune::BCRSMatrix<B, Alloc> &A, int N)
{
    typedef typename Dune::BCRSMatrix<B, Alloc>::field_type FieldType;
    setupIdentitySparsityPattern(A, N);
    for (typename Dune::BCRSMatrix<B, Alloc>::RowIterator i = A.begin(); i != A.end(); ++i)
    {
        auto bRows = i->operator[](i.index()).rows;
        for (size_t brow = 0; brow < bRows; brow++)
        {
            i->operator[](i.index())[brow][brow] = 1.0;
        }

        //i->operator[](i.index()) = 1.0; // cant just assign 1 to the entire block, only the diagonal
    }
}
#endif