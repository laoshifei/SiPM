cmake_minimum_required(VERSION 3.16)
project(CsISourceWaveforms LANGUAGES CXX)
find_package(Geant4 11.2 REQUIRED ui_all vis_all)
find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp HINTS ${ROOT_INCLUDE_DIRS} REQUIRED)
include(${Geant4_USE_FILE})
add_library(sipm_response vendor/SiPMResponse.cc)
target_compile_features(sipm_response PUBLIC cxx_std_17)
target_include_directories(sipm_response PUBLIC vendor ${JSON_INCLUDE_DIR})
add_library(readout_noise src/ReadoutNoise.cc src/ReadoutResponse.cc)
target_include_directories(readout_noise PUBLIC include)
target_link_libraries(readout_noise PUBLIC sipm_response)
add_executable(csi_sources
  main.cc
  src/DetectorConstruction.cc
  src/PhysicsList.cc
  src/ActionInitialization.cc
  src/PrimaryGeneratorAction.cc
  src/RunAction.cc
  src/EventAction.cc
  src/StackingAction.cc
  src/TrackingAction.cc
  src/TrackInformation.cc
  src/SteppingAction.cc
  src/SiPMDigitizer.cc
  src/SiPMSD.cc
  src/SiPMHit.cc
  src/RootIO.cc
  src/RunConfig.cc
  src/DataStructure.cc
)
target_include_directories(csi_sources PRIVATE include)
target_compile_definitions(csi_sources PRIVATE CSI_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(csi_sources PRIVATE readout_noise ${Geant4_LIBRARIES} ROOT::Core ROOT::RIO ROOT::Tree)
add_executable(replay_response replay_response.cc)
target_compile_features(replay_response PRIVATE cxx_std_17)
target_link_libraries(replay_response PRIVATE readout_noise ROOT::Core ROOT::RIO ROOT::Tree)
include(CTest)
if(BUILD_TESTING)
  add_executable(test_sipm tests/test_vendor_sipm.cc)
  target_link_libraries(test_sipm PRIVATE sipm_response)
  add_test(NAME inherited_sensor_charge_recovery_rng COMMAND test_sipm ${CMAKE_CURRENT_SOURCE_DIR}/tests/vendor_fixture.json)
  add_executable(test_noise tests/test_readout_noise.cc)
  target_link_libraries(test_noise PRIVATE readout_noise)
  add_test(NAME stationary_readout_noise COMMAND test_noise ${CMAKE_CURRENT_SOURCE_DIR}/tests/vendor_fixture.json)
  add_executable(test_response tests/test_readout_response.cc)
  target_link_libraries(test_response PRIVATE readout_noise)
  add_test(NAME parallel_readout_response COMMAND test_response ${CMAKE_CURRENT_SOURCE_DIR}/tests/vendor_fixture.json)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    add_test(NAME csi_application_workflow
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_workflow.py)
  endif()
endif()
