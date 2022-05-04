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
void Deformable<vertex_dim, element_dim>::ForwardTREREBDF2EREFULL(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        if (verbose_level > 1) std::cout<<"method: "<<method<<std::endl;
        InitializeStepperOptions(options);
        GetG();
        
        VectorXr q_tr = q;
        VectorXr v_tr = v;
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            std::map<int, real> augmented_dirichlet;
            ContactDirichlet(q, active_contact_idx, augmented_dirichlet);
            if (verbose_level > 1) std::cout<<"semi implicit TR ERE newton iteration \n";
                    
                ApplyDirichlet(augmented_dirichlet, q_tr, v_tr);
                SetupMatrices(q_tr, a, augmented_dirichlet, use_precomputed_data); 

                VectorXr force_tr;
                SimpleForce(q_tr, a, augmented_dirichlet, use_precomputed_data, g, force_tr);

                MassPCA(lumped_mass, MinvK, num_modes, active_contact_idx); 
                ComputePCAProjection(active_contact_idx);
                SplitVelocityState(v_tr);
                SplitForceState(force_tr);
                ApplyDirichlet(augmented_dirichlet, vH);
                ApplyDirichlet(augmented_dirichlet, fH);

                SetupJacobian(active_contact_idx);
                VectorXr rhs;
                rhs.resize(dofs()*2);
                
                // rhs.head(dofs()) = (1.0/2.0)*(-dt) * v_tr;
                // rhs.tail(dofs()).noalias() = (1.0/2.0) * (-dt) * lumped_mass_inv * force_tr;
                rhs.head(dofs()) = (1.0/3.0)*(-dt) * vH;
                rhs.tail(dofs()).noalias() = (1.0/3.0) * (-dt) * lumped_mass_inv * fH;
                VectorXr reduced_rhs;
                reduced_rhs.resize(dofs()*2);
                ComputeReducedRhs(reduced_rhs, v_tr, force_tr, 1.0/2.0 * dt ); // a hack for adjusting the internal step now
                rhs += (2.0/3.0) * reduced_rhs; // using 2/3 is also part of the hack

                std::cout<<"current residual: "<<(rhs).norm()<<"\n";
                PardisoSolver solver;
                
                A.resize((J12).rows(), (J12).cols());
                
                A.setIdentity();
                A -= (1.0/4.0) * dt * (J12 + J21);
                
                
                if (verbose_level > 1) std::cout<<"Solving for the first sparse J for TR"<<std::endl;

                if (verbose_level > 1) Tic();
                solver.Compute(A, options);
                if (verbose_level > 1) Toc("TR: decomposition");
                VectorXr x0 = solver.Solve(rhs);
                

                SubspaceEREUpdate(x0, solver, 1.0/4.0 * dt);

                q_tr -= x0.head(dofs());
                v_tr -= x0.tail(dofs());
                ApplyDirichlet(augmented_dirichlet, q_tr, v_tr);
                q_next = q_tr;
                v_next = v_tr;

                if (verbose_level > 1) std::cout<<"BDF part"<<std::endl;;

                for (size_t i = 0; i < 20; i++)
                {
                    if (verbose_level > 1) std::cout<<"BDF newton iteration "<<i<<"\n";

                    SetupMatrices(q_next, a, augmented_dirichlet, use_precomputed_data); 
                    VectorXr force_next;
                    SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_next);

                    MassPCA(lumped_mass, MinvK, num_modes, active_contact_idx); 
                    ComputePCAProjection(active_contact_idx);
                    SplitVelocityState(v_next);
                    SplitForceState(force_next);
                    ApplyDirichlet(augmented_dirichlet, vH);
                    ApplyDirichlet(augmented_dirichlet, fH);

                    SetupJacobian(active_contact_idx);
                    VectorXr rhs;
                    rhs.resize(dofs()*2);
                    rhs.setZero();
                    
                    rhs.head(dofs()) = (1.0/3.0)*(-dt) * vH;
                    rhs.tail(dofs()).noalias() = (1.0/3.0) * (-dt) * lumped_mass_inv * fH;
                    VectorXr reduced_rhs;
                    reduced_rhs.resize(dofs()*2);
                    ComputeReducedRhs(reduced_rhs, v_next, force_next, 1.0/2.0 * dt ); // a hack for adjusting the internal step now
                    rhs += (2.0/3.0) * reduced_rhs; // using 2/3 is also part of the hack
                    VectorXr diff_tr;
                    diff_tr.resize(dofs()*2);
                    diff_tr.setZero();
                    diff_tr.head(dofs()) = (q_tr - q);
                    diff_tr.tail(dofs()) = (v_tr - v);
                    rhs -= (1.0/3.0) * diff_tr;
                    // VectorXr diff_bdf;
                    // diff_bdf.resize(dofs()*2);
                    // diff_bdf.setZero();
                    // diff_bdf.head(dofs()) = (q_next - q_tr);
                    // diff_bdf.tail(dofs()) = (v_next - v_tr);
                    // rhs += diff_bdf;
                    std::cout<<"current residual: "<<(rhs).norm()<<"\n";
                    PardisoSolver solver;
                    
                    A.resize((J12).rows(), (J12).cols());
                    
                    A.setIdentity();
                    A -= (1.0/3.0) * dt * (J12 + J21);
                    

                    if (verbose_level > 1) std::cout<<"Solving for the first sparse J"<<std::endl;

                    if (verbose_level > 1) Tic();
                    solver.Compute(A, options);
                    if (verbose_level > 1) Toc("BDF FULL: decomposition");
                    VectorXr x0 = solver.Solve(rhs);

                    SubspaceEREUpdate(x0, solver, 1.0/3.0 * dt);

                    q_next -= x0.head(dofs());
                    v_next -= x0.tail(dofs());
                    ApplyDirichlet(augmented_dirichlet, q_next, v_next);

                    // if (verbose_level > 1) std::cout<<"calculating residual after a newton iteration"<<std::endl; 

                    // VectorXr force_sol_new;
                    // SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol_new);
                    // vH = -vG;
                    // vH.noalias() += v_next;
                    // fH = force_sol_new - fG;
                    
                    // rhs.head(dofs()) = (1.0/3.0) * (-dt) * vH;
                    // rhs.tail(dofs()).noalias() = (1.0/3.0) * (-dt) * lumped_mass_inv * fH;
                    // rhs += (2.0/3.0) * reduced_rhs; // using 2/3 is also part of the hack
                    // rhs -= (1.0/3.0) * diff_tr;
                    // diff_bdf.head(dofs()) = (q_next - q_tr);
                    // diff_bdf.tail(dofs()) = (v_next - v_tr);
                    // rhs += diff_bdf;
                    // double residual = (rhs).norm();
                    // std::cout<<"Residual: "<<residual<<std::endl;
                    // if( residual < 1e-6){
                    //     break;
                    // }
                    std::cout<<"Elastic Energy: "<<ElasticEnergy(q_next)<<std::endl;
                    if(si_method){
                        break;
                    }
                }
            
            break; // skip contact for now
                
        }

}




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
