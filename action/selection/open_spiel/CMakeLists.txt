cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
        open_spiel
        GIT_REPOSITORY https://github.com/deepmind/open_spiel.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)

FetchContent_GetProperties(open_spiel)
if (NOT open_spiel_POPULATED)
    FetchContent_Populate(open_spiel)
    # Execute OpenSpiel's one-time setup. This pulls in absl, sets up Python, etc.
    execute_process(
            COMMAND bash install.sh
            COMMAND_ECHO STDOUT
            WORKING_DIRECTORY ${open_spiel_SOURCE_DIR}
    )

    # Set environment variables for building OpenSpiel as a release-mode shared-library.
    # Specifically, the open_spiel target is required for linking.
    set(BUILD_SHARED_LIB_OLD $ENV{BUILD_SHARED_LIB})
    set(BUILD_TYPE_OLD $ENV{BUILD_TYPE_OLD})
    set(ENV{BUILD_SHARED_LIB} ON)
    set(ENV{BUILD_TYPE} Release)

    # TODO(WAN): EXCLUDE_FROM_ALL targets should be "excluded from IDE project files",
    #   but CLion misbehaves: https://youtrack.jetbrains.com/issue/CPP-1688
    #   As a usability workaround, we ship a default .idea/runConfigurations.
    add_subdirectory(${open_spiel_SOURCE_DIR}/open_spiel ${open_spiel_BINARY_DIR} EXCLUDE_FROM_ALL)

    # Reset environment variables.
    # TODO(WAN): Technically, these should be unset() if necessary.
    set(ENV{BUILD_SHARED_LIB} ${BUILD_SHARED_LIB_OLD})
    set(ENV{BUILD_TYPE} ${BUILD_TYPE_OLD})
endif ()

add_executable(database_game database.h database.cc database_game.cc)
target_include_directories(database_game SYSTEM PRIVATE ${open_spiel_SOURCE_DIR})
target_include_directories(database_game SYSTEM PRIVATE ${open_spiel_SOURCE_DIR}/open_spiel/abseil-cpp)
target_link_libraries(
        database_game           # The database game solver.
        absl::flags             # Command line flags.
        absl::flags_parse       # Command line flags.
        absl::random_random     # Probability distributions.
        open_spiel              # OpenSpiel shared library.
        pqxx                    # Connecting to PostgreSQL.
)