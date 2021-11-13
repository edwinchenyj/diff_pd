#include <cmath>

#include <fem/tet_deformable.h>
#include <fem/hex_deformable.h>
#include <Eigen/Dense>
#include "common/config.h"
#include "common/common.h"

class TetTest: public TetDeformable {
    public:

    MatrixXr V;
    MatrixXi E;
    TetTest(){
        
    }

    ~TetTest(){
    }

    void LoadMesh(const std::string& filename){
        // V = LoadEigenMatrixXrFromFile(filename + ".node");
        // E = LoadEigenMatrixXiFromFile(filename + ".ele");
        std::ifstream fin(filename + ".node", std::ios::in);
        
        int n_node, v_dim;
        fin >> n_node;
        fin >> v_dim;
        fin.ignore(100, '\n');
        V.resize(v_dim, n_node);
        int line_num = 0;
        for (int i = 0; i < n_node; i++) {
            fin >> line_num;
            for (int j = 0; j < v_dim; j++) {
                fin >> V(j, i);
            }
            fin.ignore(100, '\n');
        }
        fin.close();
        
        fin.open(filename + ".ele", std::ios::in);
        int n_elem, e_dim;
        fin >> n_elem;
        fin >> e_dim;
        fin.ignore(100, '\n');
        line_num = 0;
        E.resize(e_dim, n_elem);
        for (int i = 0; i < n_elem; i++) {
            fin >> line_num;
            for (int j = 0; j < e_dim; j++) {
                fin >> E(j, i);
            }
        }
    }



    VectorXr GetQNextStepForward(const std::string& method, std::vector<int>& active_contact_idx){
        
        std::map<std::string, real> options;
        options["max_newton_iter"] = 500;
        options["max_ls_iter"] = 10;
        options["abs_tol"] = 1e-9;
        options["rel_tol"] = 1e-4;
        options["verbose"] = 2;
        options["thread_ct"] = 12;
        VectorXr v(dofs()), f_ext(dofs()), q_next(dofs()), v_next(dofs());
        MatrixXr undeformed_verts = mesh().vertices();
        Eigen::Map<VectorXr> q(undeformed_verts.data(), dofs());
        v.setZero();
        f_ext.setZero();
        q_next.setZero();
        v_next.setZero();
        VectorXr a(0);
        real dt = 0.005;
        Forward(method,q,v,a,f_ext,dt,options,q_next,v_next,active_contact_idx);

        return q_next;
    }

};


int main(){
    TetTest tet;
    real density = 1000;
    real youngs_modulus = 1e5;
    real poissons_ratio = 0.45;
    tet.LoadMesh("spring_5");
    tet.Initialize(tet.V, tet.E, density, "neohookean", youngs_modulus, poissons_ratio);
    
    std::vector<int> active_contact_idx;
    //  = {716, 873, 982, 1037, 1088, 1133, 1205, 1263, 1275, 1328, 1454, 1489, 1514, 1535, 1565, 1580, 1634, 1676, 1690, 1717, 1731, 1773, 1796, 1861, 1881, 1901, 1913, 1922, 1949, 2014, 2041, 2061, 2078, 2101, 2121, 5390, 5804, 5850, 6085, 6099, 6244, 6275, 6374, 6624, 6965, 7011, 7306, 7310, 7313, 7398, 7572, 7580, 7605, 7623, 7732, 7753, 7884, 7961, 8046, 8119, 8140, 8148, 8171, 8220, 8232, 8298, 8370, 8384, 8459, 8497, 8632, 8730, 8858, 8924, 8934, 8981, 8982, 8998, 9001, 9004, 9067, 9091, 9101, 9261, 9326, 9537, 9816, 9868, 10063, 10196, 10336, 10875, 10961, 11031, 11190, 11226, 11271, 11274, 11672, 11849, 11852, 11884, 11890, 11906, 12158, 12165, 12338, 12638, 12712};
    VectorXr q_next = tet.GetQNextStepForward("siere", active_contact_idx);
    return 0;
}