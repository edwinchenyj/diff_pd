#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>


#include <fem/tet_deformable.h>
#include <fem/hex_deformable.h>
#include <Eigen/Dense>


class TetTest: public TetDeformable {
    public:
    TetTest(){
        
    }

    ~TetTest(){
    }

    int GetLumpedMassMatrixSize(){
        std::map<int, real> fake_dirichlet;
        return LumpedMassMatrix(fake_dirichlet).rows();
    }
    int GetLumpedMassSum(){
        real sum = std::accumulate(lumped_mass.begin(), lumped_mass.end(),0.0);
        return sum;
    }

    VectorXr GetQNextStepForward(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx){
    
        Forward("newton_pardiso",q,v,a,f_ext,dt,options,q_next,v_next,active_contact_idx);

        return q_next;
    }

};

class HexTest: public HexDeformable {
    public:
    HexTest(){
        
    }

    ~HexTest(){
    }

    int GetLumpedMassMatrixSize(){
        std::map<int, real> fake_dirichlet;
        return LumpedMassMatrix(fake_dirichlet).rows();
    }
    int GetLumpedMassSum(){
        int sum = std::accumulate(lumped_mass.begin(), lumped_mass.end(),0);
        return sum;
    }

};


TEST_CASE("Initialize single hex"){
    Eigen::Matrix<real, 3, 8> undeformed_vertices;
    Eigen::Matrix<int, 8, 1> elements;
    real density = 1000;
    real youngs_modulus = 1e5;
    real poissons_ratio = 0.45;
    undeformed_vertices <<  0, 1, 1, 0, 0, 1, 1, 0,
                            0, 0, 1, 1, 0, 0, 1, 1,
                            0, 0, 0, 0, 1, 1, 1, 1;
    elements << 7,
                6,
                5,
                4,
                3,
                2,
                1,
                0;
    HexTest hex;
    hex.Initialize(undeformed_vertices, elements, density, "neohookean", youngs_modulus, poissons_ratio);
    REQUIRE(hex.GetLumpedMassMatrixSize() == 24);
    REQUIRE(hex.GetLumpedMassSum() == 3 * 1 * density);

}


TEST_CASE("Initialize single tet"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    Eigen::Matrix<int, 4, 1> elements;
    real density = 1000;
    real youngs_modulus = 1e5;
    real poissons_ratio = 0.45;
    undeformed_vertices <<  0, 0, 0, 1,
                            0, 0, 1, 0, 
                            0, 1, 0, 0;
    elements << 0,
                1,
                2,
                3;
    TetTest tet;
    tet.Initialize(undeformed_vertices, elements, density, "neohookean", youngs_modulus, poissons_ratio);
    REQUIRE(tet.GetLumpedMassMatrixSize() == 12);
    REQUIRE(tet.GetLumpedMassSum() == 3.0 * 1/6 * density);

}

TEST_CASE("Forward sim for a single tet with free bc with gravity"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    Eigen::Matrix<int, 4, 1> elements;
    real density = 1000;
    real youngs_modulus = 1e5;
    real poissons_ratio = 0.45;
    undeformed_vertices <<  0, 0, 0, 1,
                            0, 0, 1, 0, 
                            0, 1, 0, 0;
    elements << 0,
                1,
                2,
                3;
    TetTest tet;
    tet.Initialize(undeformed_vertices, elements, density, "neohookean", youngs_modulus, poissons_ratio);
    
    std::vector<real> state_force_params{0,0,-9.81};
    tet.AddStateForce("gravity",state_force_params);
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
    REQUIRE(tet.GetQNextStepForward("newton_pardiso",q,v,a,f_ext,dt,options,q_next,v_next,contact).size() == 12);
    REQUIRE(q_next[2] != q[2]);
    REQUIRE(q_next[5] != q[5]);
    REQUIRE(q_next[8] != q[8]);
}

TEST_CASE("Forward sim for a single tet with free bc no gravity"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    Eigen::Matrix<int, 4, 1> elements;
    real density = 1000;
    real youngs_modulus = 1e5;
    real poissons_ratio = 0.45;
    undeformed_vertices <<  0, 0, 0, 1,
                            0, 0, 1, 0, 
                            0, 1, 0, 0;
    elements << 0,
                1,
                2,
                3;
    TetTest tet;
    tet.Initialize(undeformed_vertices, elements, density, "neohookean", youngs_modulus, poissons_ratio);
    
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
    REQUIRE(tet.GetQNextStepForward("newton_pardiso",q,v,a,f_ext,dt,options,q_next,v_next,contact).size() == 12);
    REQUIRE(q_next[2] == q[2]);
    REQUIRE(q_next[5] == q[5]);
    REQUIRE(q_next[8] == q[8]);
}

TEST_CASE("Forward sim for a single tet with one fixed bc no gravity"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    Eigen::Matrix<int, 4, 1> elements;
    real density = 1000;
    real youngs_modulus = 1e5;
    real poissons_ratio = 0.45;
    undeformed_vertices <<  0, 0, 0, 1,
                            0, 0, 1, 0, 
                            0, 1, 0, 0;
    elements << 0,
                1,
                2,
                3;
    TetTest tet;
    tet.Initialize(undeformed_vertices, elements, density, "neohookean", youngs_modulus, poissons_ratio);
    
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
    std::vector<int> contact{0,1,2};
    real dt = 0.005;
    tet.GetQNextStepForward("newton_pardiso",q,v,a,f_ext,dt,options,q_next,v_next,contact);
    REQUIRE(q_next.size() == 12);
    REQUIRE(q_next[0] == q[0]);
    REQUIRE(q_next[1] == q[1]);
    REQUIRE(q_next[2] == q[2]);
}

TEST_CASE("Forward sim for a single tet with one fixed bc with gravity"){
    Eigen::Matrix<real, 3, 4> undeformed_vertices;
    Eigen::Matrix<int, 4, 1> elements;
    real density = 1000;
    real youngs_modulus = 1e5;
    real poissons_ratio = 0.45;
    undeformed_vertices <<  0, 0, 0, 1,
                            0, 0, 1, 0, 
                            0, 1, 0, 0;
    elements << 0,
                1,
                2,
                3;
    TetTest tet;
    tet.Initialize(undeformed_vertices, elements, density, "neohookean", youngs_modulus, poissons_ratio);
    
    std::vector<real> state_force_params{0,0,-9.81};
    tet.AddStateForce("gravity",state_force_params);

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
    std::vector<int> contact{0};
    real dt = 0.005;
    tet.GetQNextStepForward("newton_pardiso",q,v,a,f_ext,dt,options,q_next,v_next,contact);
    REQUIRE(q_next.size() == 12);
    REQUIRE(q_next[2] == q[2]);
    REQUIRE(q_next[5] != q[5]);
    REQUIRE(q_next[8] != q[8]);
}

// TEST_CASE("Initialize deformable with single tet"){
//     Eigen::Matrix<real, 3, 4> undeformed_vertices;
//     undeformed_vertices <<  0, 0, 0, 1,
//                             0, 0, 1, 0, 
//                             0, 1, 0, 1;
//     Eigen::Matrix<int, 4, 1> elements;
//     elements << 0,
//                 1,
//                 2,
//                 3;
//     TetDeformable tet;

//     real youngs_modulus = 1e5;
//     real poissons_ratio = 0.45;
//     tet.Initialize(undeformed_vertices, elements, 1000, "neohookean", youngs_modulus, poissons_ratio);
//     REQUIRE(tet.LumpedMassMatrix().rows() == 12);
//     std::map<std::string, real> options;
//     options["max_newton_iter"] = 500;
//     options["max_ls_iter"] = 10;
//     options["abs_tol"] = 1e-9;
//     options["rel_tol"] = 1e-4;
//     options["verbose"] = 2;
//     options["thread_ct"] = 1;
//     VectorXr v(12), f_ext(12), q_next(12), v_next(12);
//     Eigen::Map<VectorXr> q(undeformed_vertices.data(), undeformed_vertices.size());
//     v.setZero();
//     f_ext.setZero();
//     q_next.setZero();
//     v_next.setZero();
//     VectorXr a(0);
//     std::vector<int> contact;
//     real dt = 0.005;

//     tet.Forward("newton_pardiso",q,v,a,f_ext,dt,options,q_next,v_next,contact);
// }

// TEST_CASE("Initialize deformable with single tet with pd energy and gravity"){
//     Eigen::Matrix<real, 3, 4> undeformed_vertices;
//     undeformed_vertices <<  0, 0, 0, 1,
//                             0, 0, 1, 0, 
//                             0, 1, 0, 1;
//     Eigen::Matrix<int, 4, 1> elements;
//     elements << 0,
//                 1,
//                 2,
//                 3;
//     TetDeformable tet;

//     real youngs_modulus = 1e5;
//     real poissons_ratio = 0.45;
//     tet.Initialize(undeformed_vertices, elements, 1000, "neohookean", youngs_modulus, poissons_ratio);
//     real la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio));
//     real mu = youngs_modulus / (2 * (1 + poissons_ratio));


//     std::vector<real> state_force_params{0,0,-9.81};
//     tet.AddStateForce("gravity",state_force_params);
//     std::vector<real> pd_corotated_stiffness{2*mu};
//     std::vector<real> pd_volume_stiffness{la};
//     std::vector<int> pd_apply_to_all_elements;
//     pd_apply_to_all_elements.clear();
//     tet.AddPdEnergy("corotated",pd_corotated_stiffness,pd_apply_to_all_elements);
//     tet.AddPdEnergy("volume",pd_volume_stiffness,pd_apply_to_all_elements);

//     std::map<std::string, real> options;
//     options["max_newton_iter"] = 500;
//     options["max_ls_iter"] = 10;
//     options["abs_tol"] = 1e-9;
//     options["rel_tol"] = 1e-4;
//     options["verbose"] = 2;
//     options["thread_ct"] = 4;
//     VectorXr v(12), f_ext(12), q_next(12), v_next(12);
//     Eigen::Map<VectorXr> q(undeformed_vertices.data(), undeformed_vertices.size());
//     v.setZero();
//     f_ext.setZero();
//     q_next.setZero();
//     v_next.setZero();
//     VectorXr a(0);
//     std::vector<int> contact;
//     real dt = 0.005;

//     tet.Forward("sibe",q,v,a,f_ext,dt,options,q_next,v_next,contact);
// }