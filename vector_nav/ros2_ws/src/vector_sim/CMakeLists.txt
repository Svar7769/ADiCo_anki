cmake_minimum_required(VERSION 3.8)
project(vector_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY worlds models launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
