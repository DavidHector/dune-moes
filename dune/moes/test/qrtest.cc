#include <config.h>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>
#include <chrono>

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
// #include <dune/moes/vectorclass/vectorclass.h>
#include <dune/moes/qrdecomposition.hh>
#include "timing.hh"

template<typename T>
bool checkNormTolerance(T a, T b, double tolerance){
    typename T::field_type c = a*b;
    for (size_t l = 0; l < Dune::Simd::lanes(c); l++)
    {
        if (std::abs(Dune::Simd::lane(l, c) - 1.0) > tolerance)
        {
            return false;
        }   
    }   
    return true;
}

template<typename T>
bool checkOrthogonalityTolerance(T a, T b, double tolerance){
    typename T::field_type c = a*b;
    for (size_t l = 0; l < Dune::Simd::lanes(c); l++)
    {
        if (std::abs(Dune::Simd::lane(l, c) - 0.0) > tolerance) // the 0.0 is important for type inference
        {
            return false;
        }   
    }   
    return true;
}

int main(int argc, char const *argv[])
{
    std::size_t N = 5;
    size_t simdArrayLength = 256;
    if (argc > 1)
    {
        if (1 == std::sscanf(argv[1], "%zu", &N))
        {
        } else {
            std::cout << "Please enter an unsigned integer!" << std::endl;
            return -1;
        }
    }
    if (argc > 2)
    {
        if (1 == std::sscanf(argv[2], "%zu", &simdArrayLength))
        {
        } else {
            std::cout << "Please enter a power of 32!" << std::endl;
            return -1;
        }
    }
    
    // Try out matrix multiplication with simd block vector
    const size_t simdLength = 2; // 1 = Scalar, 32 optimal for Matmul
    if ((simdLength > simdArrayLength) && (simdArrayLength%simdLength != 0))
    {
        std::cout << "simdArrayLength needs to be a multiple of the simdLength!" << std::endl;
        return 1;
    }
    
    const size_t repetitions = 10;
    typedef double FIELD_TYPE;
    typedef typename Vc::SimdArray<FIELD_TYPE, simdLength> vcArray;
    static const int BS = 1;
    const double tolerance = 1e-10;

    // Set up Vectors 
    typedef Dune::FieldVector<vcArray, BS> BlockVectorBlock; // 1 = SIZE of the fieldvector
    typedef Dune::BlockVector<BlockVectorBlock> BlockVector; 

    // Vectorized Algorithm
    std::vector<BlockVector> Multivector(simdArrayLength/simdLength, BlockVector(N));
    for (auto & bv : Multivector)
    {
        fillBVRandom(bv);
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        auto startT = std::chrono::high_resolution_clock::now();
        qVectorized<BlockVector, BS>(Multivector);
        auto stopT = std::chrono::high_resolution_clock::now();
        auto durationT = std::chrono::duration_cast<std::chrono::milliseconds>(stopT - startT);
        std::cout << "Vectorized algorithm: " << durationT.count() << "ms" << std::endl;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // Scalarized Algorithm
    std::vector<BlockVector> MultivectorOld(simdArrayLength/simdLength, BlockVector(N));
    for (auto & bv : MultivectorOld)
    {
        fillBVRandom(bv);
    }
    auto startOld = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < repetitions; i++)
    {
        auto startT = std::chrono::high_resolution_clock::now();
        qScalarized<BlockVector, BS>(MultivectorOld);
        auto stopT = std::chrono::high_resolution_clock::now();
        auto durationT = std::chrono::duration_cast<std::chrono::milliseconds>(stopT - startT);
        std::cout << "Scalarized algorithm: " << durationT.count() << "ms" << std::endl;
    }
    auto stopOld = std::chrono::high_resolution_clock::now();
    auto durationOld = std::chrono::duration_cast<std::chrono::milliseconds>(stopOld - startOld);

    std::cout << "Multivector Width = " << simdArrayLength << " | Individual vector size = " << N << std::endl;
    std::cout << "SimdArray Length = " << simdLength << std::endl;
    std::cout << "Average duration for " << repetitions << " repetitions of the vectorized algorithm: " << duration.count()/repetitions << "ms" << std::endl;
    std::cout << "Average duration for " << repetitions << " repetitions of the old algorithm: " << durationOld.count()/repetitions << "ms" << std::endl;

    std::cout << "Checking orthogonality and norms..." << std::endl;
    bool error = false;

    checkOrthoNormality<BlockVector, BS, FIELD_TYPE>(Multivector, 1e-10, error);

    if (error)
    {
        std::cout << "Encountered Errors!" << std::endl;
    } else {
        std::cout << "No errors were found :)" << std::endl;
    }
    
    return 0;
}
