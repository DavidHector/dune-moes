#ifndef DUNE_MOES_QRDECOMPOSITION_HH
#define DUNE_MOES_QRDECOMPOSITION_HH

#include <vector>
#include <iostream>
#include <stdexcept>

template<typename VT, typename SVT>
void subtraction(VT& u, SVT v){
    for (size_t blockIndex = 0; blockIndex < u.N(); blockIndex++)
    {
        for (size_t entryIndex = 0; entryIndex < u[0].N(); entryIndex++)
        {
            u[blockIndex][entryIndex] -= v[blockIndex][entryIndex];
        } 
    }
}

template<typename VT, typename SVT>
void broadcast(VT& u, SVT& v){
    typedef typename VT::field_type FT;
    for (size_t blockIndex = 0; blockIndex < u.N(); blockIndex++)
    {
        for (size_t entryIndex = 0; entryIndex < u[0].N(); entryIndex++)
        {
            u[blockIndex][entryIndex] = FT(v[blockIndex][entryIndex]);
        } 
    }
}

template<typename SIMDTYPE, typename SCALARTYPE>
void broadcastScalar(SIMDTYPE& a, SCALARTYPE& b){
    a = SIMDTYPE(b);
}

// input: Dune BlockVector u
// output: normalized (|u| = 1.0) BlockVector u
template<typename VT>
void normalize(VT& u, double tolerance = 0.0000001){
    typename VT::field_type norm = u*u;
    for (size_t i = 0; i < Dune::Simd::lanes(norm); i++)
    {
        if (std::abs(Dune::Simd::lane(i, norm) - 0.0) < tolerance)
        {
            throw std::invalid_argument("Column vector is zero!");
        }
        Dune::Simd::lane(i, norm) = std::sqrt(Dune::Simd::lane(i, norm));   
    }
    u /= norm;
}

// input: BlockVectors: u, v  |  Length of vector u: unorm
// output: BlockVector: projection = projection of u onto v divided by unorm
template<typename VT>
void projectionV(const VT& u, const VT&v, const typename VT::field_type& unorm, VT& projection){
    projection = u; // copy
    typename VT::field_type zero(0.0);
    if ((unorm==zero).isFull())
    {
        std::cout << "Projection is zero!" << std::endl;
    }
    typename VT::field_type factor = (projection*v) / unorm;
    projection *= factor;
}

// input: BlockVectors: u, v  |  Length of vector u: unorm
// output: BlockVector: ucopy = projection of u onto v divided by unorm
template<typename SVT>
void projection(const SVT& u, const SVT&v, const typename SVT::field_type& unorm, SVT& ucopy){
    ucopy = u; // initialize with correct size
    if (unorm == 0)
    {
        std::cout << "Projection is zero!" << std::endl;
    }
    typename SVT::field_type factor = (u*v) / unorm;
    ucopy *= factor;
}

// input: std::vector<Dune::Blockvector> in (no SIMD)
// output: std::vector<Dune::Blockvector> Q (with SIMD)
template<typename MV, typename MVScalar>
void transformScalarToSimd(MV& Q, MVScalar& in){
    const size_t oldVectorNumber = Q.size();
    const size_t simdLanes = Dune::Simd::lanes(Q[0][0][0]);
    const size_t newVectorNumber = oldVectorNumber*simdLanes;
    const size_t blockNumber = Q[0].N();
    const size_t blockSize = Q[0][0].N();
    for (size_t vectorIndex = 0; vectorIndex < Q.size(); vectorIndex++)
    {
        for (size_t blockIndex = 0; blockIndex < blockNumber; blockIndex++)
        {
            for (size_t entryIndex = 0; entryIndex < blockSize; entryIndex++)
            {
                for (size_t simdIndex = 0; simdIndex < simdLanes; simdIndex++)
                {
                    Dune::Simd::lane(simdIndex, Q[vectorIndex][blockIndex][entryIndex]) = in[vectorIndex*simdLanes + simdIndex][blockIndex][entryIndex];
                }   
            }
        }
    }
}

// input: std::vector<Dune::Blockvector> Q (with SIMD)
// output: std::vector<Dune::Blockvector> out (no SIMD)
template<typename MV, typename MVScalar, int BS>
void transformSimdToScalar(MV& Q, MVScalar& out){
    for (size_t vectorIndex = 0; vectorIndex < Q.size(); vectorIndex++)
    {
        for (size_t blockIndex = 0; blockIndex < Q[0].N(); blockIndex++)
        {
            for (size_t entryIndex = 0; entryIndex < BS; entryIndex++)
            {
                for (size_t simdIndex = 0; simdIndex < Dune::Simd::lanes(Q[0][0][0]); simdIndex++)
                {
                    out[vectorIndex*Dune::Simd::lanes(Q[0][0][0]) + simdIndex][blockIndex][entryIndex] = Dune::Simd::lane(simdIndex, Q[vectorIndex][blockIndex][entryIndex]);
                }   
            }
        }
    }
}

// input: Simd BlockVector Qi
// output: std::vector of scalar BlockVectors scalarVec
template<typename VT, typename SVT>
void transformSimdBVToScalarStdVector(VT& Qi, std::vector<SVT>& scalarVec){
    for (size_t blockIndex = 0; blockIndex < Qi.N(); blockIndex++)
    {
        for (size_t entryIndex = 0; entryIndex < Qi[0].N(); entryIndex++)
        {
            for (size_t simdIndex = 0; simdIndex < Dune::Simd::lanes(Qi[0][0]); simdIndex++)
            {
                scalarVec[simdIndex][blockIndex][entryIndex] = Dune::Simd::lane(simdIndex, Qi[blockIndex][entryIndex]);
            }
        }
    }
} 

// input: std::vector of scalar BlockVectors scalarVec
// output: Simd BlockVector Qi
template<typename VT, typename SVT>
void writeScalarStdVectorToSimdBVT(VT& Qi, std::vector<SVT>& scalarVec){
    for (size_t blockIndex = 0; blockIndex < Qi.N(); blockIndex++)
    {
        for (size_t entryIndex = 0; entryIndex < Qi[0].N(); entryIndex++)
        {
            for (size_t simdIndex = 0; simdIndex < Dune::Simd::lanes(Qi[0][0]); simdIndex++)
            {
                Dune::Simd::lane(simdIndex, Qi[blockIndex][entryIndex]) = scalarVec[simdIndex][blockIndex][entryIndex];
            }
        }
    }
} 



// getProjectionVectorAndOrthogonalizeCurrentBlock: gets a single column vector from a scalar BlockVector and 
// orthogonalizes the other column vectors in the same simdBlock
// input: std::vector<Dune::Blockvector> scalarVec (no SIMD), size_t simdIndex this is the index of the vector we want to extract
// output: Dune::Blockvector orthoVec (no SIMD), currentVectorNorm
template<typename SVT, typename SIMDTYPE>
void getProjectionVectorAndOrthogonalizeCurrentBlock(std::vector<SVT>& scalarVec, SVT& orthoVec, size_t& simdIndex, SIMDTYPE& cv, SVT& projectedVector, typename SVT::field_type& currentVectorNorm){
    orthoVec = scalarVec[simdIndex];
    currentVectorNorm = orthoVec * orthoVec;
    broadcastScalar(cv, currentVectorNorm);
    for (size_t col = simdIndex+1; col < scalarVec.size(); col++)
    {
        projection(orthoVec, scalarVec[col], currentVectorNorm, projectedVector);
        scalarVec[col] -= projectedVector;
    }
}

// orthogonalizeBroadcastCurrentBlock: orthogonalizes and broadcasts the column vectors of a simd block
// input:   Q_i: simdBlock to orthogonalize, 
//        
// output:  out: vector of orthogonalized and broadcasted simd blocks for later use
//          cvnVector: (simd norms of out)
template<typename VT, typename SVT>
void orthogonalizeBroadcastCurrentBlock(
    VT& Qi,
    std::vector<VT>& out,
    std::vector<SVT>& scalarVec,
    std::vector<typename VT::field_type>& cvnVector,
    typename SVT::field_type& cvnScalar,
    SVT& projectedVector)
    {
    // simdWidth = out.size()
    transformSimdBVToScalarStdVector(Qi, scalarVec);
    for (size_t simdIndex = 0; simdIndex < out.size(); simdIndex++)
    {
        cvnScalar = scalarVec[simdIndex]*scalarVec[simdIndex];
        cvnVector[simdIndex] = typename VT::field_type(cvnScalar);
        for (size_t simdIndexf = simdIndex+1; simdIndexf < out.size(); simdIndexf++)
        {
            projection(scalarVec[simdIndex], scalarVec[simdIndexf], cvnScalar, projectedVector);
            scalarVec[simdIndexf] -= projectedVector;
        }
        broadcast(out[simdIndex], scalarVec[simdIndex]);
    }
    writeScalarStdVectorToSimdBVT(Qi, scalarVec);
}



template<typename SVT> // Only with double support in mind currently
void qDecompositionGramSchmidt(std::vector<SVT>& Q, SVT& projectedVector){
    typename SVT::field_type currentVectorNorm = Q[0][0][0]; // without SIMD this is double
    // iterate over all the columns (vectors)
    for (size_t col = 0; col < Q.size()-1; col++)
    {
        currentVectorNorm = Q[col] * Q[col];
        // orthogonalize all following vectors with that vector
        for (size_t colProj = col+1; colProj < Q.size(); colProj++)
        {
            projection(Q[col], Q[colProj], currentVectorNorm, projectedVector);
            Q[colProj] -= projectedVector;
        }
        // normalize
        normalize(Q[col]);
    }
    // normalize the last vector
    normalize(Q[Q.size()-1]);
}

template<typename VT, int BS, typename ET = double> // Only with double support in mind currently
void qScalarized(std::vector<VT>& Q){
    typedef Dune::FieldVector<ET, BS> BlockVectorBlock;
    typedef Dune::BlockVector<BlockVectorBlock> BlockVector;
    const size_t oldVectorNumber = Q.size();
    const size_t simdLanes = Dune::Simd::lanes(Q[0][0][0]);
    const size_t newVectorNumber = oldVectorNumber*simdLanes;
    const size_t blockNumber = Q[0].N();
    BlockVector projectedVector(blockNumber);

    std::vector<BlockVector> tmp(newVectorNumber, BlockVector(blockNumber));
    transformSimdToScalar<std::vector<VT>, std::vector<BlockVector>,  BS>(Q, tmp);
    qDecompositionGramSchmidt(tmp, projectedVector);
    transformScalarToSimd(Q, tmp);
}

// qVectorized: Vectorized Gram-Schmidt to orthogonalize the matrix Q
// input: std::vector<Dune::BlockVector> (with Simd entries) Q
// output: Q, but orthogonalized
template<typename VT, int BS, typename ET=double>
void qVectorized(std::vector<VT>& Q){
    // Initialize all the things, so no initialization has to happen inside the functions
    const size_t simdWidth = Dune::Simd::lanes(Q[0][0][0]);
    const size_t blockNumber = Q[0].N();
    typedef Dune::FieldVector<ET, BS> BlockVectorBlock;
    typedef Dune::BlockVector<BlockVectorBlock> BlockVector; // Scalarized Vector type
    std::vector<BlockVector> scalarVectors(simdWidth, BlockVector(blockNumber));
    std::vector<VT> bOSB(simdWidth, VT(blockNumber)); //broadcasted Orthogonalized Simd Block
    std::vector<typename VT::field_type> cvnVector(simdWidth); // norms of bOSB
    BlockVector projectedVector(blockNumber);
    VT projection(blockNumber);
    ET currentVectorNorm;
    typename VT::field_type cv = Q[0][0][0];
    
    // Iterate over all simd Blocks (column vectors with simd width)
    for (size_t simdBlock = 0; simdBlock < Q.size(); simdBlock++)
    {
        orthogonalizeBroadcastCurrentBlock(Q[simdBlock], bOSB, scalarVectors, cvnVector, currentVectorNorm, projectedVector);
        for (size_t simdIndex = 0; simdIndex < simdWidth; simdIndex++)
        {
            for (size_t followingBlockIndex = simdBlock + 1; followingBlockIndex < Q.size(); followingBlockIndex++)
            {
                projectionV(bOSB[simdIndex], Q[followingBlockIndex], cvnVector[simdIndex], projection);
                Q[followingBlockIndex] -= projection;
            }
        }
        normalize(Q[simdBlock]);  
    }
    normalize(Q[Q.size()-1]);
}

template<typename VT, int BS, typename ET = double>
void checkOrthoNormality(std::vector<VT>& Q, ET tolerance, bool& error){
    typedef Dune::FieldVector<ET, BS> BlockVectorBlock;
    typedef Dune::BlockVector<BlockVectorBlock> BlockVector;
    const size_t oldVectorNumber = Q.size();
    const size_t simdLanes = Dune::Simd::lanes(Q[0][0][0]);
    const size_t newVectorNumber = oldVectorNumber*simdLanes;
    const size_t blockNumber = Q[0].N();
    const size_t entryNumber = Q[0].N() * Q[0][0].N();
    if (newVectorNumber > entryNumber)
    {
        std::cout << "Error: Width of the Multivector is greater than the length of each vector, no linear independence possible" <<  std::endl;
        error = true;
        return;
    }
    
    
    std::vector<BlockVector> tmp(newVectorNumber, BlockVector(blockNumber));
    transformSimdToScalar<std::vector<VT>, std::vector<BlockVector>,  BS>(Q, tmp);
    for (size_t i = 0; i < tmp.size(); i++)
    {
        for (size_t j = 0; j < tmp.size(); j++)
        {
            if (i == j)
            {
                if (std::abs(tmp[i]*tmp[j] - 1.0) > tolerance)
                {
                    std::cout << "Norm violation! Calculated norm with self: " << tmp[i]*tmp[j] << std::endl;
                    error = true;
                }
            } else {
                if (std::abs(tmp[i]*tmp[j] - 0.0) > tolerance)
                {
                    std::cout << "Orthogonality violation! Calculated norm with other: " << tmp[i]*tmp[j] << std::endl;
                    error = true;
                }
            }  
        }
    }
}


#endif // DUNE_MOES_QRDECOMPOSITION_HH
