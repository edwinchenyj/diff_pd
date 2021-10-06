#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


#include <fem/tet_deformable.h>
#include <Eigen/Dense>

TEST_CASE("Initialize deformable with single tet"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    undeformed_vertices <<  0, 0, 0, 1,
                            0, 0, 1, 0, 
                            0, 1, 0, 1;
    Eigen::Matrix<int, 4, 1> elements;
    elements << 0,
                1,
                2,
                3;
    TetDeformable tet;

    real youngs_modulus = 1e6;
    real poissons_ratio = 0.45;
    tet.Initialize(undeformed_vertices, elements, 1000, "linear", youngs_modulus, poissons_ratio);
    
    std::map<std::string, real> options;
    options["max_newton_iter"] = 500;
    options["max_ls_iter"] = 10;
    options["abs_tol"] = 1e-9;
    options["rel_tol"] = 1e-4;
    options["verbose"] = 2;
    options["thread_ct"] = 1;
    VectorXr v(12), f_ext(12), q_next(12), v_next(12);
    Eigen::Map<VectorXr> q(undeformed_vertices.data(), undeformed_vertices.size());
    v.setZero();
    f_ext.setZero();
    q_next.setZero();
    v_next.setZero();
    VectorXr a(0);
    std::vector<int> contact;
    real dt = 0.005;

    tet.Forward("newton_pardiso",q,v,a,f_ext,dt,options,q_next,v_next,contact);
}

TEST_CASE("Initialize deformable with single tet with pd energy and gravity"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    undeformed_vertices <<  0, 0, 0, 1,
                            0, 0, 1, 0, 
                            0, 1, 0, 1;
    Eigen::Matrix<int, 4, 1> elements;
    elements << 0,
                1,
                2,
                3;
    TetDeformable tet;

    real youngs_modulus = 1e6;
    real poissons_ratio = 0.45;
    tet.Initialize(undeformed_vertices, elements, 1000, "linear", youngs_modulus, poissons_ratio);
    real la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio));
    real mu = youngs_modulus / (2 * (1 + poissons_ratio));


    std::vector<real> state_force_params{0,0,-9.81};
    tet.AddStateForce("gravity",state_force_params);
    std::vector<real> pd_corotated_stiffness{2*mu};
    std::vector<real> pd_volume_stiffness{la};
    std::vector<int> pd_apply_to_all_elements;
    pd_apply_to_all_elements.clear();
    tet.AddPdEnergy("corotated",pd_corotated_stiffness,pd_apply_to_all_elements);
    tet.AddPdEnergy("volume",pd_volume_stiffness,pd_apply_to_all_elements);

    std::map<std::string, real> options;
    options["max_newton_iter"] = 500;
    options["max_ls_iter"] = 10;
    options["abs_tol"] = 1e-9;
    options["rel_tol"] = 1e-4;
    options["verbose"] = 2;
    options["thread_ct"] = 4;
    VectorXr v(12), f_ext(12), q_next(12), v_next(12);
    Eigen::Map<VectorXr> q(undeformed_vertices.data(), undeformed_vertices.size());
    v.setZero();
    f_ext.setZero();
    q_next.setZero();
    v_next.setZero();
    VectorXr a(0);
    std::vector<int> contact;
    real dt = 0.005;

    tet.Forward("sibe",q,v,a,f_ext,dt,options,q_next,v_next,contact);
}