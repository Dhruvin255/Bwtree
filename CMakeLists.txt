cmake_minimum_required(VERSION 3.10)
project(bwtree CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

# All targets compile the same two translation units.
set(BWTREE_SOURCES
    src/bwtree.cpp
    tests/concurrency_stress.cpp)

find_package(Threads REQUIRED)

# ---- plain optimized build ------------------------------------------------
add_executable(bwtree_test ${BWTREE_SOURCES})
target_compile_options(bwtree_test PRIVATE -O2 -g -Wall -Wextra)
target_link_libraries(bwtree_test PRIVATE Threads::Threads)

# ---- ThreadSanitizer build (data races, use-after-free on the read path) ---
add_executable(bwtree_test_tsan ${BWTREE_SOURCES})
target_compile_options(bwtree_test_tsan PRIVATE
    -O1 -g -fsanitize=thread -fno-omit-frame-pointer -Wall -Wextra)
target_link_options(bwtree_test_tsan PRIVATE -fsanitize=thread)
target_link_libraries(bwtree_test_tsan PRIVATE Threads::Threads)

# ---- AddressSanitizer build (heap-use-after-free, leaks, OOB) --------------
add_executable(bwtree_test_asan ${BWTREE_SOURCES})
target_compile_options(bwtree_test_asan PRIVATE
    -O1 -g -fsanitize=address -fno-omit-frame-pointer -Wall -Wextra)
target_link_options(bwtree_test_asan PRIVATE -fsanitize=address)
target_link_libraries(bwtree_test_asan PRIVATE Threads::Threads)
