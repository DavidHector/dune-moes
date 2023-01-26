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
#include <dune/moes/vectorclass/vectorclass.h>
#include "timing.hh"

int main(int argc, char const *argv[])
{
    std::size_t N = 3;
    if (argc > 1)
    {
        if (1 == std::sscanf(argv[1], "%zu", &N))
        {
        } else {
            std::cout << "Please enter an unsigned integer!" << std::endl;
            return -1;
        }
    }
    
    // Try out matrix multiplication with simd block vector
    const size_t simdArrayLength = 256;
    const size_t simdLength = 4;
    typedef double FIELD_TYPE;
    typedef typename Vc::Vector<FIELD_TYPE> vcvec; // Has length 4
    typedef typename Vc::SimdArray<FIELD_TYPE, simdArrayLength> vcArray;
    static const int BS = 1;

    std::cout << "Tests with laplacian matrix: " << std::endl << std::endl;

    std::cout << "BlockVector length = " << N*N << std::endl;
    std::cout << "Field-Type: uses doubles" << std::endl;
    std::cout << "single: simdVector length = " << simdLength << std::endl;
    std::cout << "single (Agner Fog's Vectorclass): simdVector length = 4" << std::endl;
    std::cout << "native: simdArray length = " << simdArrayLength << std::endl;
    std::cout << "emulated: " << simdArrayLength/simdLength << " std::vectors that contain elements with simdVector length = " << simdLength << std::endl;
    std::cout << "emulated (Agner Fog's Vectorclass): " << simdArrayLength/simdLength << " std::vectors that contain elements with simdVector length = 4" << std::endl;
    std::cout << "Duration Field-Type: " << timeMatrixMultiplication<FIELD_TYPE, FIELD_TYPE>(N) << "ms" << std::endl;
    std::cout << "Duration single: " << timeMatrixMultiplication<FIELD_TYPE, vcvec>(N) << "ms" << std::endl;
    std::cout << "Duration single (Agner Fog's Vectorclass): " << timeMatrixMultiplication<FIELD_TYPE, Vec4d>(N) << "ms" << std::endl;
    std::cout << "Duration emulated: " << timeMatrixMultiplication<FIELD_TYPE, vcvec>(N, simdArrayLength) << "ms" << std::endl;
    std::cout << "Duration emulated (Agner Fogs Vectorclass): " << timeMatrixMultiplication<FIELD_TYPE, Vec4d>(N, simdArrayLength) << "ms" << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, vcArray>(N, simdArrayLength, simdArrayLength) << "ms" << std::endl;

    std::cout << std::endl << std::endl;

    std::cout << "BlockVector length = " << N*N << std::endl;
    std::cout << "Multivector size = " << simdArrayLength << std::endl;
    std::cout << "Duration single (Vec4d): " << timeMatrixMultiplication<FIELD_TYPE, Vec4d>(N, simdArrayLength) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 1 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 1>>(N, simdArrayLength, 1) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 2 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 2>>(N, simdArrayLength, 2) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 4 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 4>>(N, simdArrayLength, 4) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 8 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 8>>(N, simdArrayLength, 8) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 16 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 16>>(N, simdArrayLength, 16) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 32 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 32>>(N, simdArrayLength, 32) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 64 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 64>>(N, simdArrayLength, 64) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 128 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 128>>(N, simdArrayLength, 128) << "ms" << std::endl;

    std::cout << "SimdArray length = " << 256 << std::endl;
    std::cout << "Duration native: " << timeMatrixMultiplication<FIELD_TYPE, Vc::SimdArray<FIELD_TYPE, 256>>(N, simdArrayLength, 256) << "ms" << std::endl;


    /*
    std::cout << "vec: " << std::endl;
    std::cout << vec << std::endl;
    
    std::cout << "arr: " << std::endl;
    std::cout << arr << std::endl;

    std::cout << "emulatedSimdArr: " << std::endl;
    for (auto& v: emulatedSimdArr)
    {
        std::cout << v << std::endl;
    }

    std::cout << "y_emulatedSimdArr: " << std::endl;
    for (auto& v: y_emulatedSimdArr)
    {
        std::cout << v << std::endl;
    }

    std::cout << "bvA: " << std::endl;
    std::cout << bvA << std::endl;

    std::cout << "bvA[0]: " << std::endl;
    std::cout << bvA[0] << std::endl;

    std::cout << "yA: " << std::endl;
    std::cout << yA << std::endl;
    */
    return 0;
}
