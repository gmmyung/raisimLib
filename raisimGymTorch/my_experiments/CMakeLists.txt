cmake_minimum_required(VERSION 3.10)
project(raisim_examples LANGUAGES CXX)

list(APPEND CMAKE_PREFIX_PATH /home/audrud/raisimlib/raisim/linux/)
find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(MY_EXPERIMENT raisim_gym.cpp)
target_link_libraries(MY_EXPERIMENT PUBLIC raisim::raisim pthread)
target_link_libraries(MY_EXPERIMENT PUBLIC ${EIGEN3_LIBRARIES})
target_include_directories(MY_EXPERIMENT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})