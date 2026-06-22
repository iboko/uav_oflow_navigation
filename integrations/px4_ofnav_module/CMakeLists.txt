# PX4 in-tree module CMakeLists.txt
# Install into PX4-Autopilot/src/modules/ofnav using tools/install_px4_ofnav_module.sh

px4_add_module(
    MODULE modules__ofnav
    MAIN ofnav
    STACK_MAIN 4096
    SRCS
        ofnav_px4.cpp
        ofnav_core/src/ofnav.cpp
    INCLUDES
        ofnav_core/include
    DEPENDS
)
