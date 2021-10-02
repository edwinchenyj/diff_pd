#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


#include <fem/tet_deformable.h>
#include <Eigen/Dense>

TEST_CASE("Initialize deformable with tet mesh"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    undeformed_vertices << 0, 0, 0, 0,
                        0, 0, 1, 1, 
                        0, 1, 0, 1;
    Eigen::Matrix<int, 4, 1> elements;
    elements << 0,
                1,
                2,
                3;
    TetDeformable tet;
    tet.Initialize(undeformed_vertices, elements, 1000, "linear", 1e6, 0.45);
}