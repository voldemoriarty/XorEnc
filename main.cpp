#ifndef BUFFSIZE
#define BUFFSIZE (8192 * 32)
#endif

#include <immintrin.h>  // AVX2 Intrinsics
#include <iostream>     // std::cout
#include <fstream>      // std::fstream
#include <chrono>       // timing
#include <cstdint>      // uint64_t

using ivec_t = __m256i; // SIMD Vector type

static void ShowUsage()
{
  std::cout << "Usage: xor inputFile outputFile" << std::endl;
}

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    ShowUsage();
    return 1;
  }

  // make sure the buffer size is a multiple of
  // 32 bytes (256bit, SIMD register type)
  constexpr int vectorsPerBuff = BUFFSIZE / 32;
  static_assert(BUFFSIZE % 32 == 0, "Buffer size should be a multiple of 32 (simd register size)");

  std::ifstream iFile(argv[1]);
  std::ofstream oFile(argv[2]);

  // the buffer needs to be 256bit aligned
  alignas(32) char buffer[BUFFSIZE];

  // fill the vector with 0xdeadbeefs
  ivec_t key = _mm256_set1_epi32(0xdeadbeef);
  uint64_t totalBytes = 0;
  auto time = std::chrono::high_resolution_clock::now;
  auto then = time();

  // read BUFFSIZE bytes at a time, xor and write
  while (iFile) {
    iFile.read(buffer, BUFFSIZE);
    // the number of bytes read
    auto bytesRead = iFile.gcount();

    for (int i = 0; i < vectorsPerBuff; ++i) {
      auto pVector = reinterpret_cast<ivec_t *>(buffer) + i;  // address in the buffer
      auto vVector = _mm256_load_si256(pVector);              // load in register
      auto vXorred = _mm256_xor_si256(key, vVector);          // xor
      _mm256_store_si256(pVector, vXorred);                   // store back
    }

    oFile.write(buffer, bytesRead);
    totalBytes += bytesRead;
  }

  iFile.close();
  oFile.close();

  auto now = time();

  {
    using namespace std::chrono;
    auto duration = duration_cast<microseconds>(now - then);
    std::cout << "Time: " << duration.count() << " us" << std::endl;
    std::cout << "Data: " << totalBytes << " bytes" << std::endl;
    std::cout << "Average speed: " << totalBytes / duration.count() * 1e6 / 1024 / 1024 << " MBps" << std::endl;
  }
  return 0;
}
