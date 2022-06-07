#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "solver/pardiso_spd_solver.h"
#include "Eigen/SparseCholesky"
#include "Spectra/GenEigsSolver.h"
#include "Spectra/MatOp/SparseGenMatProd.h"
#include "Spectra/GenEigsRealShiftSolver.h"
#include "Spectra/SymEigsShiftSolver.h"
#include "Spectra/MatOp/SparseSymShiftSolve.h"
#include "Spectra/MatOp/SparseGenRealShiftSolve.h"
#include "solver/SparseGenRealShiftSolvePardiso.h"

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardBDFFULL(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        if (verbose_level > 1) std::cout<<"method: "<<method<<std::endl;
        InitializeStepperOptions(options);
        GetG();
        q_next = q; 
        v_next = v;
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            std::map<int, real> augmented_dirichlet = dirichlet_;
            ContactDirichlet(q, active_contact_idx, augmented_dirichlet);
            for (size_t i = 0; i < 20; i++)
            {
                if (verbose_level > 1) std::cout<<"BDF newton iteration "<<i<<"\n";
            
                ApplyDirichlet(augmented_dirichlet, q_next, v_next);

                SetupMatrices(q_next, a, augmented_dirichlet, use_precomputed_data);
                VectorXr force_sol;
                SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol);
                
                SetupJacobian(active_contact_idx);
                A.resize((J12).rows(), (J12).cols());
                
                A.setIdentity();
                A -= dt * (J12 + J21);
                VectorXr rhs;
                rhs.resize(dofs()*2);
                
                rhs.head(dofs()) = (-dt) * v_next;
                rhs.tail(dofs()).noalias() = (-dt) * lumped_mass_inv * force_sol;
                VectorXr diff;
                diff.resize(dofs()*2);
                diff.head(dofs()) = q_next - q;
                diff.tail(dofs()) = v_next - v;
                rhs += diff;
                PardisoSolver solver;
                
                if (verbose_level > 1) Tic();
                solver.Compute(A, options);
                if (verbose_level > 1) Toc("BDF FULL: decomposition");
                VectorXr x0 = solver.Solve(rhs);
                q_next -= x0.head(dofs());
                v_next -= x0.tail(dofs());
                ApplyDirichlet(augmented_dirichlet, q_next, v_next);

            }
            std::cout<<"Elastic Energy: "<<ElasticEnergy(q_next)<<std::endl;
            break; // skip contact for now
        }


    }




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
