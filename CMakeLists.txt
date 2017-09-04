cmake_minimum_required(VERSION 2.8)

project(milkbox)

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /opt/freenect2/)

find_package(PCL 1.7 REQUIRED PATHS /usr NO_DEFAULT_PATH)
find_package(VTK REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(freenect2 REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include(${VTK_USE_FILE})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${freenect2_INCLUDE_DIR})
include_directories("/usr/local/include")
link_directories(${PCL_LIBRARY_DIRS})
link_directories("/opt/freenect2/lib/")

set(SRC_LIST
    main.cpp
    kinect2grabber.cpp
    )

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} freenect2)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})
