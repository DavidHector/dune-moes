#ifndef DUNE_MOES_QRVEC_HH
#define DUNE_MOES_QRVEC_HH
#include <vector>
#include <cstdlib>
#include <cmath>
#include <dune/moes/vectorclass/vectorclass.h>

void fillMatrixRandom(std::vector<std::vector<double>>& Q){
    for (size_t row = 0; row < Q.size(); row++)
    {
        for (size_t col = 0; col < Q[0].size(); col++)
        {
            Q[row][col] = static_cast<double> (std::rand()) / RAND_MAX;
        }
    }
}

void fillMatrixRandom(double** Q, const size_t rows, const size_t cols){
    for (size_t row = 0; row < rows; row++)
    {
        for (size_t col = 0; col < cols; col++)
        {
            Q[row][col] = static_cast<double> (std::rand()) / RAND_MAX;
        }
    }
}



bool checkOrthogonality(const std::vector<std::vector<double>>& Q, const double& tolerance){
    double uv = 0.0;
    for (size_t col = 0; col < Q[0].size(); col++)
    {
        for (size_t folcol = col+1; folcol < Q[0].size(); folcol++)
        {
            uv = 0.0;
            for (size_t row = 0; row < Q.size(); row++)
            {
                uv += Q[row][col] * Q[row][folcol];
            }
            if (std::abs(uv) > tolerance)
            {
                std::cout << "Orthogonality violation!" << std::endl;
                std::cout << "Dotproduct is " << uv << " should be: 0" << std::endl;
                std::cout << "Col 1: " << col << " Col 2:" << folcol << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool checkNorm(const std::vector<std::vector<double>>& Q, const double& tolerance){
    double unorm = 0.0;
    for (size_t col = 0; col < Q[0].size(); col++)
    {
        unorm = 0.0;
        for (size_t row = 0; row < Q.size(); row++)
        {
            unorm += Q[row][col] * Q[row][col];
        }
        unorm = std::sqrt(unorm);
        if (std::abs(unorm - 1.0) > tolerance)
        {
            std::cout << "Norm violation!" << std::endl;
            std::cout << "Norm is " << unorm << " should be: 1" << std::endl;
            std::cout << "Column: " << col << std::endl;
            return false;
        }   
    }
    return true;
}

void checkOrthoNormality(const std::vector<std::vector<double>>& Q, const double& tolerance){
    bool success = true;
    if(!checkNorm(Q, tolerance)){
        success = false;
    }
    if(!checkOrthogonality(Q, tolerance)){
        success = false;
    }
    if (success)
    {
        std::cout << "Successfully passed checks!" << std::endl;
    } else {
        std::cout << "Checks failed!" << std::endl;
    }
}

bool checkOrthogonality(double** Q, const double& tolerance, const size_t rows, const size_t cols){
    double uv = 0.0;
    for (size_t col = 0; col < cols; col++)
    {
        for (size_t folcol = col+1; folcol < cols; folcol++)
        {
            uv = 0.0;
            for (size_t row = 0; row < rows; row++)
            {
                uv += Q[row][col] * Q[row][folcol];
            }
            if (std::abs(uv) > tolerance)
            {
                std::cout << "Orthogonality violation!" << std::endl;
                std::cout << "Dotproduct is " << uv << " should be: 0" << std::endl;
                std::cout << "Col 1: " << col << " Col 2:" << folcol << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool checkNorm(double** Q, const double& tolerance, const size_t rows, const size_t cols){
    double unorm = 0.0;
    for (size_t col = 0; col < cols; col++)
    {
        unorm = 0.0;
        for (size_t row = 0; row < rows; row++)
        {
            unorm += Q[row][col] * Q[row][col];
        }
        unorm = std::sqrt(unorm);
        if (std::abs(unorm - 1.0) > tolerance)
        {
            std::cout << "Norm violation!" << std::endl;
            std::cout << "Norm is " << unorm << " should be: 1" << std::endl;
            std::cout << "Column: " << col << std::endl;
            return false;
        }   
    }
    return true;
}

void checkOrthoNormality(double** Q, const double& tolerance, const size_t rows, const size_t cols){
    bool success = true;
    if(!checkNorm(Q, tolerance, rows, cols)){
        success = false;
    }
    if(!checkOrthogonality(Q, tolerance, rows, cols)){
        success = false;
    }
    if (success)
    {
        std::cout << "Successfully passed checks!" << std::endl;
    } else {
        std::cout << "Checks failed!" << std::endl;
    }
}

template<size_t blockSize>
void qr(std::vector<std::vector<double>>& Q){
    size_t cols = Q[0].size();
    size_t rows = Q.size();
    std::vector<std::vector<double>> factors(cols, std::vector<double>(blockSize, 0.0));
    std::vector<double> intraBlockFactors(blockSize, 0.0);
    double unorm = 0.0;

    for (size_t ucols = 0; ucols < cols; ucols+=blockSize)
    {   
        
        // Orthonormalize the current block
        for (size_t uucol = 0; uucol < blockSize; uucol++)
        {
            // reset intraBlockFactors
            for (size_t i = 0; i < blockSize; i++)
            {
                intraBlockFactors[i] = 0.0;
            }
            unorm = 0.0;
            // get norm of uucol
            for (size_t row = 0; row < rows; row++)
            {
                unorm += Q[row][ucols + uucol] * Q[row][ucols + uucol]; 
            }
            // normalize ucol to 1
            for (size_t row = 0; row < rows; row++)
            {
                Q[row][ucols + uucol] /= std::sqrt(unorm);
            }
            
            // get intraBlockfactors
            // u1*u2, u1*u3,...
            for (size_t row = 0; row < rows; row++)
            {
                for (size_t uvcol = uucol+1; uvcol < blockSize; uvcol++)
                {
                    intraBlockFactors[uvcol] += Q[row][ucols + uucol] * Q[row][ucols + uvcol];
                } 
            }

            // linear combination
            for (size_t row = 0; row < rows; row++)
            {
                for (size_t uvcol = uucol+1; uvcol < blockSize; uvcol++)
                {
                    Q[row][ucols+uvcol] -= intraBlockFactors[uvcol] * Q[row][ucols + uucol];
                }
            }
        }

        // Orthogonalize following blocks

        // reset factors
        for (size_t i = 0; i < factors.size(); i++)
        {
            for (size_t j = 0; j < factors[0].size(); j++)
            {
                factors[i][j] = 0.0;
            }
        }
        
        // calculate factors (can vectorize a lot here), should also be able to keep ucols in cache here
        for (size_t row = 0; row < rows; row++)
        {
            for (size_t vcols = ucols+blockSize; vcols < cols; vcols+=blockSize)
            {
                for (size_t vvcol = 0; vvcol < blockSize; vvcol++)
                {
                    for (size_t uucol = 0; uucol < blockSize; uucol++)
                    {
                        factors[vcols+vvcol][uucol] += Q[row][vcols+vvcol] * Q[row][ucols+uucol];
                    }
                }
            }
        }

        for (size_t row = 0; row < rows; row++)
        {
            for (size_t vcols = ucols+blockSize; vcols < cols; vcols+=blockSize)
            {
                for (size_t vvcol = 0; vvcol < blockSize; vvcol++)
                {
                    for (size_t uucol = 0; uucol < blockSize; uucol++)
                    {
                        Q[row][vcols+vvcol] -= factors[vcols+vvcol][uucol] * Q[row][ucols+uucol];
                    }
                }
            }
        }        
    }
    
}

template<size_t blockSize>
void qrV(std::vector<std::vector<double>>& Q){
    size_t cols = Q[0].size();
    size_t rows = Q.size();
    std::vector<std::vector<Vec4d>> factors(cols/4, std::vector<Vec4d>(blockSize, 0.0)); // maybe this is accessed in a wrong spot?
    std::vector<double> intraBlockFactors(blockSize, 0.0);
    double unorm = 0.0;
    size_t factorIndex = 0;

    Vec4d ui = 0.0;
    Vec4d vi = 0.0;

    // used later in broadcasting
    Vec4d uZero = 0.0;
    Vec4d uOne = 0.0;
    Vec4d uTwo = 0.0;
    Vec4d uThree = 0.0;
    

    for (size_t ucols = 0; ucols < cols; ucols+=blockSize)
    {   
        // Orthonormalize the current block
        for (size_t uucol = 0; uucol < blockSize; uucol++)
        {
            // reset intraBlockFactors
            for (size_t i = 0; i < blockSize; i++)
            {
                intraBlockFactors[i] = 0.0;
            }
            unorm = 0.0;
            // get norm of uucol
            for (size_t row = 0; row < rows; row++)
            {
                unorm += Q[row][ucols + uucol] * Q[row][ucols + uucol]; 
            }
            // normalize ucol to 1
            for (size_t row = 0; row < rows; row++)
            {
                Q[row][ucols + uucol] /= std::sqrt(unorm);
            }
            
            // get intraBlockfactors
            // u1*u2, u1*u3,...
            for (size_t row = 0; row < rows; row++)
            {
                for (size_t uvcol = uucol+1; uvcol < blockSize; uvcol++)
                {
                    intraBlockFactors[uvcol] += Q[row][ucols + uucol] * Q[row][ucols + uvcol];
                } 
            }

            // linear combination
            for (size_t row = 0; row < rows; row++)
            {
                for (size_t uvcol = uucol+1; uvcol < blockSize; uvcol++)
                {
                    Q[row][ucols+uvcol] -= intraBlockFactors[uvcol] * Q[row][ucols + uucol];
                }
            }
        }

        // Orthogonalize following blocks

        // reset factors
        for (size_t i = 0; i < cols/4; i++)
        {
            for (size_t j = 0; j < blockSize; j++)
            {
                factors[i][j] = 0.0;
            }
        }
        
        // calculate factors (can vectorize a lot here), should also be able to keep ucols in cache here
        for (size_t row = 0; row < rows; row++)
        {
            for (size_t ucolvec = 0; ucolvec < blockSize; ucolvec+=4)
            {
                uZero = Q[row][ucols + ucolvec]; // broadcast
                uOne = Q[row][ucols + ucolvec + 1];
                uTwo = Q[row][ucols + ucolvec + 2];
                uThree = Q[row][ucols + ucolvec + 3];
                for (size_t vcols = ucols+blockSize; vcols < cols; vcols+=blockSize)
                {
                    for (size_t vcolvec = 0; vcolvec < blockSize; vcolvec+=4)
                    {
                        vi.load(&Q[row][vcols + vcolvec]);
                        factorIndex = (vcols + vcolvec)/4; // Either don't use this or find more efficient method (like shifting bits)
                        /* */
                        factors[factorIndex][ucolvec + 0] += vi * uZero; // this contains [v0*u0, v1*u0, v2*u0, v3*u0]
                        factors[factorIndex][ucolvec + 1] += vi * uOne;
                        factors[factorIndex][ucolvec + 2] += vi * uTwo;
                        factors[factorIndex][ucolvec + 3] += vi * uThree;
                        /**/
                        /*
                        factors[factorIndex][ucolvec + 0] = mul_add(vi, uZero, factors[factorIndex][ucolvec + 0]);
                        factors[factorIndex][ucolvec + 1] = mul_add(vi, uOne, factors[factorIndex][ucolvec + 1]);
                        factors[factorIndex][ucolvec + 2] = mul_add(vi, uTwo, factors[factorIndex][ucolvec + 2]);
                        factors[factorIndex][ucolvec + 3] = mul_add(vi, uThree, factors[factorIndex][ucolvec + 3]);
                        */
                    }
                }
            }
        }

        // Linear combination
        for (size_t row = 0; row < rows; row++)
        {
            for (size_t ucolvec = 0; ucolvec < blockSize; ucolvec+=4)
            {
                uZero = Q[row][ucols + ucolvec];
                uOne = Q[row][ucols + ucolvec + 1];
                uTwo = Q[row][ucols + ucolvec + 2];
                uThree = Q[row][ucols + ucolvec + 3];
                for (size_t vcols = ucols+blockSize; vcols < cols; vcols+=blockSize)
                {
                    for (size_t vcolvec = 0; vcolvec < blockSize; vcolvec+=4)
                    {
                        vi.load(&Q[row][vcols + vcolvec]);
                        factorIndex = (vcols + vcolvec)/4;
                    
                        vi = nmul_add(factors[factorIndex][ucolvec + 0], uZero, vi);
                        vi = nmul_add(factors[factorIndex][ucolvec + 1], uOne, vi);
                        vi = nmul_add(factors[factorIndex][ucolvec + 2], uTwo, vi);
                        vi = nmul_add(factors[factorIndex][ucolvec + 3], uThree, vi);
                        
                        /*
                        vi -= factors[factorIndex][ucolvec + 0]*uZero + factors[factorIndex][ucolvec + 1]*uOne
                            + factors[factorIndex][ucolvec + 2]*uTwo  + factors[factorIndex][ucolvec + 3]*uThree;
                        */
                        vi.store(&Q[row][vcols + vcolvec]);   
                    }
                }
            }
        }
    }
}
template<size_t blockSize>
void qrVArr(double** Q, const size_t rows, const size_t cols){
    Vec4d** factors = new Vec4d*[cols/4];
    for (size_t i = 0; i < cols/4; i++)
    {
        factors[i] = new Vec4d[blockSize];
    }
    double* intraBlockFactors = new double[blockSize];
    double unorm = 0.0;
    size_t factorIndex = 0;

    Vec4d ui = 0.0;
    Vec4d vi = 0.0;

    // used later in broadcasting
    Vec4d uZero = 0.0;
    Vec4d uOne = 0.0;
    Vec4d uTwo = 0.0;
    Vec4d uThree = 0.0;
    

    for (size_t ucols = 0; ucols < cols; ucols+=blockSize)
    {   
        // Orthonormalize the current block
        for (size_t uucol = 0; uucol < blockSize; uucol++)
        {
            // reset intraBlockFactors
            for (size_t i = 0; i < blockSize; i++)
            {
                intraBlockFactors[i] = 0.0;
            }
            unorm = 0.0;
            // get norm of uucol
            for (size_t row = 0; row < rows; row++)
            {
                unorm += Q[row][ucols + uucol] * Q[row][ucols + uucol]; 
            }
            // normalize ucol to 1
            for (size_t row = 0; row < rows; row++)
            {
                Q[row][ucols + uucol] /= std::sqrt(unorm);
            }
            
            // get intraBlockfactors
            // u1*u2, u1*u3,...
            for (size_t row = 0; row < rows; row++)
            {
                for (size_t uvcol = uucol+1; uvcol < blockSize; uvcol++)
                {
                    intraBlockFactors[uvcol] += Q[row][ucols + uucol] * Q[row][ucols + uvcol];
                } 
            }

            // linear combination
            for (size_t row = 0; row < rows; row++)
            {
                for (size_t uvcol = uucol+1; uvcol < blockSize; uvcol++)
                {
                    Q[row][ucols+uvcol] -= intraBlockFactors[uvcol] * Q[row][ucols + uucol];
                }
            }
        }

        // Orthogonalize following blocks

        // reset factors
        for (size_t i = 0; i < cols/4; i++)
        {
            for (size_t j = 0; j < blockSize; j++)
            {
                factors[i][j] = 0.0;
            }
        }
        
        // calculate factors (can vectorize a lot here), should also be able to keep ucols in cache here
        for (size_t row = 0; row < rows; row++)
        {
            for (size_t ucolvec = 0; ucolvec < blockSize; ucolvec+=4)
            {
                uZero = Q[row][ucols + ucolvec]; // broadcast
                uOne = Q[row][ucols + ucolvec + 1];
                uTwo = Q[row][ucols + ucolvec + 2];
                uThree = Q[row][ucols + ucolvec + 3];
                for (size_t vcols = ucols+blockSize; vcols < cols; vcols+=blockSize)
                {
                    for (size_t vcolvec = 0; vcolvec < blockSize; vcolvec+=4)
                    {
                        vi.load(&Q[row][vcols + vcolvec]);
                        factorIndex = (vcols + vcolvec)/4; // Either don't use this or find more efficient method (like shifting bits)
                        /* */
                        factors[factorIndex][ucolvec + 0] += vi * uZero; // this contains [v0*u0, v1*u0, v2*u0, v3*u0]
                        factors[factorIndex][ucolvec + 1] += vi * uOne;
                        factors[factorIndex][ucolvec + 2] += vi * uTwo;
                        factors[factorIndex][ucolvec + 3] += vi * uThree;
                        /**/
                        /*
                        factors[factorIndex][ucolvec + 0] = mul_add(vi, uZero, factors[factorIndex][ucolvec + 0]);
                        factors[factorIndex][ucolvec + 1] = mul_add(vi, uOne, factors[factorIndex][ucolvec + 1]);
                        factors[factorIndex][ucolvec + 2] = mul_add(vi, uTwo, factors[factorIndex][ucolvec + 2]);
                        factors[factorIndex][ucolvec + 3] = mul_add(vi, uThree, factors[factorIndex][ucolvec + 3]);
                        */
                    }
                }
            }
        }

        // Linear combination
        for (size_t row = 0; row < rows; row++)
        {
            for (size_t ucolvec = 0; ucolvec < blockSize; ucolvec+=4)
            {
                uZero = Q[row][ucols + ucolvec];
                uOne = Q[row][ucols + ucolvec + 1];
                uTwo = Q[row][ucols + ucolvec + 2];
                uThree = Q[row][ucols + ucolvec + 3];
                for (size_t vcols = ucols+blockSize; vcols < cols; vcols+=blockSize)
                {
                    for (size_t vcolvec = 0; vcolvec < blockSize; vcolvec+=4)
                    {
                        vi.load(&Q[row][vcols + vcolvec]);
                        factorIndex = (vcols + vcolvec)/4;
                    
                        vi = nmul_add(factors[factorIndex][ucolvec + 0], uZero, vi);
                        vi = nmul_add(factors[factorIndex][ucolvec + 1], uOne, vi);
                        vi = nmul_add(factors[factorIndex][ucolvec + 2], uTwo, vi);
                        vi = nmul_add(factors[factorIndex][ucolvec + 3], uThree, vi);
                        
                        /*
                        vi -= factors[factorIndex][ucolvec + 0]*uZero + factors[factorIndex][ucolvec + 1]*uOne
                            + factors[factorIndex][ucolvec + 2]*uTwo  + factors[factorIndex][ucolvec + 3]*uThree;
                        */
                        vi.store(&Q[row][vcols + vcolvec]);   
                    }
                }
            }
        }
    }
}


#endif // DUNE_MOES_QRVEC_HH