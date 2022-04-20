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
void Deformable<vertex_dim, element_dim>::ForwardBDFEREFULL(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        std::cout<<"method: "<<method<<std::endl;
        InitializeStepperOptions(options);
        GetG();
        q_next = q; 
        v_next = v;
        VectorXr reduced_rhs;
        reduced_rhs.resize(dofs()*2);
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            std::map<int, real> augmented_dirichlet = dirichlet_;
            ContactDirichlet(q, active_contact_idx, augmented_dirichlet);
            for (size_t i = 0; i < 10; i++)
            {
                if (verbose_level > 1) std::cout<<"BDF newton iteration "<<i<<"\n";
            
                ApplyDirichlet(augmented_dirichlet, q_next, v_next);

                SetupMatrices(q_next, a, augmented_dirichlet, use_precomputed_data);
                VectorXr force_sol;
                SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol);
                if(i == 0){
                    MassPCA(lumped_mass, MinvK, num_modes, active_contact_idx);
                    ComputePCAProjection(active_contact_idx);
                    SplitVelocityState(v_next); 
                    SplitForceState(force_sol);
                    ComputeReducedRhs(reduced_rhs, v_next, force_sol, dt);
                }
                else{
                    vH = -vG;
                    vH += v_next;
                    fH = force_sol - fG;
                }

                SetupJacobian(active_contact_idx);
                A.resize((J12).rows(), (J12).cols());
                
                A.setIdentity();
                A -= dt * (J12 + J21);

            
                VectorXr rhs1;
                rhs1.resize(dofs()*2);
                VectorXr rhs;
                rhs.resize(dofs()*2);
                
                rhs1.head(dofs()) = (-dt) * vH;
                rhs1.tail(dofs()).noalias() = (-dt) * lumped_mass_inv * fH;
                

                rhs = rhs1 + reduced_rhs;
                VectorXr diff;
                diff.resize(dofs()*2);
                diff.head(dofs()) = q_next - q;
                diff.tail(dofs()) = v_next - v;
                rhs += diff;
                std::cout<<"current residual: "<<rhs.norm()<<std::endl;
                PardisoSolver solver;
                
                if (verbose_level > 1) std::cout<<"Solving for the first sparse J"<<std::endl;

                if (verbose_level > 1) Tic();
                solver.Compute(A, options);
                if (verbose_level > 1) Toc("SIERE: decomposition");
                VectorXr x0 = solver.Solve(rhs);

                SubspaceEREUpdate(x0, solver, dt);
                q_next = q;
                v_next = v;
                q_next -= x0.head(dofs());
                v_next -= x0.tail(dofs());

                ApplyDirichlet(dirichlet_, q_next, v_next);

                if (verbose_level > 1) std::cout<<"calculating residual after a newton iteration"<<std::endl; 
                VectorXr force_sol_new;
                SimpleForce(q_next, a, dirichlet_, use_precomputed_data, g, force_sol_new);

                // can't use split velocity state here because we need the old vG state 
                vH = -vG;
                vH.noalias() += v_next;
                fH = force_sol_new - fG;

                rhs1.head(dofs()) = (-dt) * vH;
                rhs1.tail(dofs()).noalias() = (-dt) * lumped_mass_inv * fH;

                rhs = rhs1 + reduced_rhs;
                diff.head(dofs()) = q_next - q;
                diff.tail(dofs()) = v_next - v;
                rhs += diff;
                double residual = (rhs).norm();
                std::cout<<"Residual: "<<residual<<std::endl;
                if (residual < 1e-6) {
                    break;
                }
            }
            std::cout<<"Elastic Energy: "<<ElasticEnergy(q_next)<<std::endl;
            break; // skip contact for now
        }


    }




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
