#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>

#include <dune/moes/gramSchmidt.hh>
#include <dune/moes/vectorclass/vectorclass.h>
#include <dune/moes/flopUtils.hh>

int main(int argc, char const *argv[])
{
    std::string gsSTFile = "gsST.csv";
    std::string gsMTFile = "gsMT.csv";
    std::string MatMulSTFile = "MatMulST.csv";
    std::string MatMulMTFile = "MatMulMT.csv";

    std::cout << "Benchmarking the Gram-Schmidt-Process (single Thread)..." << std::endl;
    // flopsGSAutoST(gsSTFile);
    std::cout << "Finished." << std::endl;

    std::cout << "Benchmarking the Gram-Schmidt-Process (multithreaded)..." << std::endl;
    // flopsGSAutoMT(gsMTFile);
    std::cout << "Finished." << std::endl;

    std::cout << "Benchmarking the Matrix Multiplication (single Thread)..." << std::endl;
    flopsMatmulST(MatMulSTFile);
    std::cout << "Finished." << std::endl;

    std::cout << "Benchmarking the Matrix Multiplication (multithreaded)..." << std::endl;
    flopsMatmulMT(MatMulMTFile);
    std::cout << "Finished." << std::endl;
    return 0;
}
