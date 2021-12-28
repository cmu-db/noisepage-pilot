cmake_minimum_required(VERSION 3.16)

project(
        NoisePage_Pilot
        VERSION 0.1.0.0
        DESCRIPTION "Pilot components for the NoisePage project."
        HOMEPAGE_URL "https://noise.page/"
        LANGUAGES CXX
)

add_subdirectory(action/selection/open_spiel)
