cmake_minimum_required(VERSION 3.15)
project(Nebula CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ARM / Raspberry Pi optimisations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|armv[78]")
    add_compile_options(-O3 -march=native -mfloat-abi=hard -ffast-math)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(SNDFILE REQUIRED sndfile)

set(LIB_SOURCES
    src/audio_utils.cpp
    src/fft_utils.cpp
    src/filterbank.cpp
    src/features/time_domain.cpp
    src/features/spectral.cpp
    src/features/time_frequency.cpp
    src/features/cepstral.cpp
    src/features/pitch.cpp
    src/features/chroma.cpp
)

add_library(nebula STATIC ${LIB_SOURCES})
target_include_directories(nebula PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${FFTW3F_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)
target_link_directories(nebula PUBLIC
    ${FFTW3F_LIBRARY_DIRS}
    ${SNDFILE_LIBRARY_DIRS}
)
target_link_libraries(nebula PUBLIC
    ${FFTW3F_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    m
)

add_executable(benchmark main.cpp)
target_link_libraries(benchmark nebula)
target_include_directories(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)
