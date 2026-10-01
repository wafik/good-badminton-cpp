cmake_minimum_required(VERSION 3.20)
project(gb_cpp CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GB_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(OpenCV_DIR ${GB_ROOT}/third_party/opencv/build)
find_package(OpenCV REQUIRED)

set(ORT_DIR ${GB_ROOT}/third_party/onnxruntime-win-x64-1.19.2)
add_library(ort INTERFACE)
target_include_directories(ort INTERFACE ${ORT_DIR}/include)
target_link_libraries(ort INTERFACE ${ORT_DIR}/lib/onnxruntime.lib)

add_library(gbnlohmann INTERFACE)
target_include_directories(gbnlohmann INTERFACE ${GB_ROOT}/third_party)

function(gb_obj name dir)
  file(GLOB srcs ${GB_ROOT}/src/${dir}/*.cpp)
  if(NOT srcs)
    return()  # module not ported yet — skip target
  endif()
  add_library(${name} OBJECT ${srcs})
  target_include_directories(${name} PUBLIC ${GB_ROOT}/include PRIVATE ${GB_ROOT}/third_party ${ORT_DIR}/include)
  target_link_libraries(${name} PUBLIC ${OpenCV_LIBS} ort gbnlohmann)
  if(MSVC)
    target_compile_options(${name} PRIVATE /utf-8 /W3)
  endif()
endfunction()

gb_obj(gb_court   court)
gb_obj(gb_detect  detect)
gb_obj(gb_track   track)
gb_obj(gb_viz     viz)
gb_obj(gb_io      io)

file(GLOB CORE_SRC ${GB_ROOT}/src/core/*.cpp)
add_library(gb_core OBJECT ${CORE_SRC})
target_include_directories(gb_core PUBLIC ${GB_ROOT}/include)
if(MSVC)
  target_compile_options(gb_core PRIVATE /utf-8 /W3)
endif()

# gb_cpp only once integration sources exist (agents land files over time)
set(APP_SRC src/main.cpp src/system.cpp)
set(APP_PRESENT TRUE)
foreach(f ${APP_SRC})
  if(NOT EXISTS ${GB_ROOT}/${f})
    set(APP_PRESENT FALSE)
  endif()
endforeach()
if(APP_PRESENT AND TARGET gb_court AND TARGET gb_detect AND TARGET gb_track
   AND TARGET gb_viz AND TARGET gb_io)
  add_executable(gb_cpp
    ${APP_SRC}
    $<TARGET_OBJECTS:gb_court>
    $<TARGET_OBJECTS:gb_detect>
    $<TARGET_OBJECTS:gb_track>
    $<TARGET_OBJECTS:gb_viz>
    $<TARGET_OBJECTS:gb_io>
    $<TARGET_OBJECTS:gb_core>
  )
  target_include_directories(gb_cpp PRIVATE ${GB_ROOT}/include ${GB_ROOT}/third_party ${ORT_DIR}/include)
  target_link_libraries(gb_cpp PRIVATE ${OpenCV_LIBS} ort gbnlohmann)
  if(MSVC)
    target_compile_options(gb_cpp PRIVATE /utf-8 /W3)
  endif()
endif()

# rally state self-check: build/test_rally.exe must print ok
add_executable(test_rally tests/test_rally.cpp src/core/rally_state.cpp)
target_include_directories(test_rally PRIVATE ${GB_ROOT}/include)
if(MSVC)
  target_compile_options(test_rally PRIVATE /utf-8)
endif()
