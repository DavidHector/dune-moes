#ifndef MOES_TIMING_HH
#define MOES_TIMING_HH
#include <config.h>
#include <utility>
#include <vector>
#include <chrono>
#include <cstdlib>

#include <dune/common/fmatrix.hh>
#include <dune/common/classname.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/rangeutilities.hh>
#include <dune/common/simd/io.hh>
#include <dune/common/simd/loop.hh>
#include <dune/common/simd/simd.hh>
#include <dune/common/simd/vc.hh>
#include <dune/common/std/type_traits.hh>
#include <dune/common/typelist.hh>
#include <dune/common/typetraits.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/test/identity.hh>
#include <dune/istl/test/laplacian.hh>
#include <dune/moes/vectorclass/vectorclass.h>

template <typename VO, typename FT = double> // Vc Object
void fillSimd(VO& vo, FT filler = 1.0){
    for (size_t i = 0; i < Dune::Simd::lanes(vo); i++)
    {
        Dune::Simd::lane(i, vo) = filler;
    } 
}

template <typename FT = double> // Template Specialization for Agner Fog's Vec4d vectorclass Object
void fillSimd(Vec4d& vo, FT filler = 1.0){
    double a[4] = {filler, filler, filler, filler};
    vo.load(a);
}

template <typename BV, typename FT = double> // Block Vector object
void fillBV(BV& bv, FT filler = 1.0){
    for (typename BV::size_type i = 0; i < bv.N(); i++)
    {
        for (typename BV::size_type j = 0; j < bv[i].N(); j++)
        {
            fillSimd(bv[i][j], filler);
        }
    }
    
}

template <typename VO, typename FT = double> // Vc Object
void fillSimdRandom(VO& vo){
    for (size_t i = 0; i < Dune::Simd::lanes(vo); i++)
    {
        Dune::Simd::lane(i, vo) = static_cast<FT>(std::rand()) / RAND_MAX;
    } 
}

template <typename FT = double> // Template Specialization for Agner Fog's Vec4d vectorclass Object
void fillSimdRandom(Vec4d& vo){
    double a[4] = {static_cast<FT>(std::rand()) / RAND_MAX, static_cast<FT>(std::rand()) / RAND_MAX, static_cast<FT>(std::rand()) / RAND_MAX, static_cast<FT>(std::rand()) / RAND_MAX};
    vo.load(a);
}

template <typename BV, typename FT = double> // Block Vector object
void fillBVRandom(BV& bv){
    for (typename BV::size_type i = 0; i < bv.N(); i++)
    {
        for (typename BV::size_type j = 0; j < bv[i].N(); j++)
        {
            fillSimdRandom(bv[i][j]);
        }
    }
    
}

template <typename FT, typename ET> // Field Type, (Vectorized) Entry Type, Duration Type
auto timeMatrixMultiplication(size_t N){
    // Set up the matrix
    static const int BS = 1;
    typedef Dune::FieldMatrix<FT,BS,BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;

    BCRSMat laplacian;
    setupLaplacian(laplacian, N);

    // Set up Vectors 
    typedef Dune::FieldVector<ET,1> BlockVectorBlock; // 1 = SIZE of the fieldvector
    typedef Dune::BlockVector<BlockVectorBlock> BlockVector; 
    
    BlockVector bv(N*N);
    BlockVector y(N*N);
    fillBV(bv);

    // Measure time for the matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    laplacian.mv(bv, y);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    return duration.count();
}

template <typename FT, typename ET> // Field Type, (Vectorized) Entry Type, Duration Type
auto timeMatrixMultiplication(size_t N, const size_t simdArrayLength, const size_t simdLength=4, const size_t sampleCount = 10){
    // Set up the matrix
    static const int BS = 1;
    typedef Dune::FieldMatrix<FT,BS,BS> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> BCRSMat;

    BCRSMat laplacian;
    setupLaplacian(laplacian, N); // Note that it will have size: N^2xN^2

    // Set up Vectors 
    typedef Dune::FieldVector<ET,1> BlockVectorBlock; // 1 = SIZE of the fieldvector
    typedef Dune::BlockVector<BlockVectorBlock> BlockVector; 

    std::vector<BlockVector> emulatedSimdArr(simdArrayLength/simdLength, BlockVector(N*N));
    std::vector<BlockVector> y_emulatedSimdArr(simdArrayLength/simdLength, BlockVector(N*N));
    FT fill_value = 1.0;
    for (auto& bv : emulatedSimdArr)
    {
        fillBV(bv, fill_value);
        fill_value += 1.0;
    }

    // Measure time for matrix multiplications
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t t = 0; t < sampleCount; t++)
    {
        for (size_t i = 0; i < emulatedSimdArr.size(); i++)
        {
            laplacian.mv(emulatedSimdArr[i], y_emulatedSimdArr[i]);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    return duration.count()/sampleCount;
}



#endif // MOES_TIMING_HH