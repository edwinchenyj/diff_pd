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
    tet.LoadMesh("Beam");
    tet.Initialize(tet.V, tet.E, density, "neohookean", youngs_modulus, poissons_ratio);
    
    std::vector<int> active_contact_idx
     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 15, 16, 17, 18, 19, 20, 21, 23, 26, 28, 29, 30, 31, 32, 33, 34, 36, 39, 41, 42, 43, 44, 45, 51, 54, 56, 57, 58, 59, 60, 61, 62, 64, 67, 69, 70, 71, 72, 73, 86, 87, 94, 97, 98, 99, 100, 101, 122, 123, 124, 125, 126, 127, 188, 194, 206, 222, 228, 234, 245, 246, 247, 248, 261, 270, 297, 300, 313, 315, 316, 317, 346, 347};
    VectorXr q_next = tet.GetQNextStepForward("siere", active_contact_idx);
    return 0;
}