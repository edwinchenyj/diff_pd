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
void Deformable<vertex_dim, element_dim>::ForwardSBDF2FULL(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        if (verbose_level > 1) std::cout<<"method: "<<method<<std::endl;
       std::map<std::string, real> opt_si = options;
       opt_si["si_method"] = 1;

        std::cout<<"method: sbdf2\n";
        InitializeStepperOptions(options);
        GetG();

        q_next = q;
        v_next = v;
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            std::map<int, real> augmented_dirichlet;
            ContactDirichlet(q, active_contact_idx, augmented_dirichlet);
            if (q_prev.size() != q.size()) {
                if (verbose_level > 1) std::cout<<"initialize 2 steps\n";
                q_prev.resize(q.size());
                v_prev.resize(v.size());
                q_prev = q;
                v_prev = v;
            }


                ApplyDirichlet(augmented_dirichlet, q_next, v_next);
            
                SetupMatrices(q_next, a, augmented_dirichlet, use_precomputed_data);
                VectorXr force_sol;
                SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol);
                
                SetupJacobian(active_contact_idx);
                VectorXr rhs;
                rhs.resize(dofs()*2);
                
                rhs.head(dofs()) = (2.0/3.0)*(-dt) * v_next;
                rhs.tail(dofs()).noalias() = (2.0/3.0) * (-dt) * lumped_mass_inv * force_sol;
                VectorXr diff_prev;
                diff_prev.resize(dofs()*2);
                diff_prev.setZero();
                diff_prev.head(dofs()) = (q - q_prev);
                diff_prev.tail(dofs()) = (v - v_prev);
                rhs -= (1.0/3.0) * diff_prev;
                VectorXr diff;
                diff.resize(dofs()*2);
                diff.head(dofs()) = q_next - q;
                diff.tail(dofs()) = v_next - v;
                rhs += diff;
                std::cout<<"current residual: "<<(rhs).norm()<<"\n";
                PardisoSolver solver;
                
                A.resize((J12).rows(), (J12).cols());
                
                A.setIdentity();
                A -= (2.0/3.0) * dt * (J12 + J21);

                if (verbose_level > 1) Tic();
                solver.Compute(A, options);
                if (verbose_level > 1) Toc("BDF FULL: decomposition");
                VectorXr x0 = solver.Solve(rhs);
                
                q_next -= x0.head(dofs());
                v_next -= x0.tail(dofs());
                ApplyDirichlet(augmented_dirichlet, q_next, v_next);


                // if (verbose_level > 1) std::cout<<"calculating residual after a newton iteration"<<std::endl; 
                // rhs.head(dofs()) = (2.0/3.0) * (-dt) * v_next;
                // VectorXr force_sol_new;
                // SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol_new);
                // rhs.tail(dofs()).noalias() = (2.0/3.0) * (-dt) * lumped_mass_inv * force_sol_new;
                // rhs -= (1.0/3.0) * diff_prev;
                // diff.head(dofs()) = q_next - q;
                // diff.tail(dofs()) = v_next - v;
                // rhs += diff;
                // double residual = (rhs).norm();
                // std::cout<<"Residual: "<<residual<<std::endl;
            std::cout<<"Elastic Energy: "<<ElasticEnergy(q_next)<<std::endl;
            q_prev = q;
            v_prev = v;
            break; // skip contact for now
        }


       }




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
