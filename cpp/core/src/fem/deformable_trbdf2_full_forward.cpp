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
void Deformable<vertex_dim, element_dim>::ForwardTRBDF2FULL(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        const int verbose_level = static_cast<int>(options.at("verbose"));
        if (verbose_level > 1) std::cout<<"forward tr bdf 2 full\n";
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        if (verbose_level > 1) std::cout<<"max_ls_iter: "<<options.at("max_ls_iter")<<"\n";
        CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
        if (verbose_level > 1) std::cout<<"abs_tol: "<<options.at("abs_tol")<<"\n";
        CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
        if (verbose_level > 1) std::cout<<"rel_tol: "<<options.at("rel_tol")<<"\n";
        CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
        if (verbose_level > 1) std::cout<<"verbose: "<<options.at("verbose")<<"\n";
        CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
        if (verbose_level > 1) std::cout<<"thread_ct: "<<options.at("thread_ct")<<"\n";
        const int thread_ct = static_cast<int>(options.at("thread_ct"));
        omp_set_num_threads(thread_ct);
        if (verbose_level > 1) std::cout<<"thread_ct: "<<thread_ct<<"\n";
        bool si_method;
        if (options.find("si_method") != options.end()) si_method = true;
        else si_method = false;
        const real h = dt;
        VectorXr g;
        CheckError(state_forces_.size() <= 1, "Only one state force, gravity, is supported for SIERE");
        if(state_forces_.size() == 1) {
            g = state_forces_[0]->parameters().head(vertex_dim);
        } else {
            g = VectorXr::Zero(vertex_dim);
        }
        
        std::vector<real> inv_h2_lumped_mass;
        std::transform(lumped_mass_.begin(),lumped_mass_.end(), std::back_inserter(inv_h2_lumped_mass),[&h](real mass)-> real { return mass/(h * h);});
        const int max_contact_iter = 5;
        const bool use_precomputed_data = !pd_element_energies_.empty();
        VectorXr q_tr = q;
        VectorXr v_tr = v;
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            if (verbose_level > 0) std::cout << "Contact iteration " << contact_iter << std::endl;
            if (verbose_level > 1) std::cout<<"before fix dirichlet\n";
            std::map<int, real> augmented_dirichlet = dirichlet_;

            if (verbose_level > 1) std::cout<<"before active contact idx\n";
            if (verbose_level > 1) std::cout<<active_contact_idx.size()<<"\n";
            for (const int idx : active_contact_idx) {
                for (int i = 0; i < vertex_dim; ++i){
                    augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
                }
            }
            if (verbose_level > 1) std::cout<<"semi implicit TR newton iteration \n";
                    
                ApplyDirichlet(augmented_dirichlet, q_tr, v_tr);
                SetupMatrices(q_tr, a, augmented_dirichlet, use_precomputed_data); 

                VectorXr force_tr;
                SimpleForce(q_tr, a, augmented_dirichlet, use_precomputed_data, g, force_tr);
                SetupJacobian(active_contact_idx);
                VectorXr rhs;
                rhs.resize(dofs()*2);
                
                rhs.head(dofs()) = (1.0/2.0)*(-h) * v_tr;
                rhs.tail(dofs()).noalias() = (1.0/2.0) * (-h) * lumped_mass_inv * force_tr;
                std::cout<<"current residual: "<<(rhs).norm()<<"\n";
                PardisoSolver solver;
                
                A.resize((J12).rows(), (J12).cols());
                
                A.setIdentity();
                A -= (1.0/4.0) * h * (J12 + J21);
                
                
                if (verbose_level > 1) std::cout<<"Solving for the first sparse J for TR"<<std::endl;

                if (verbose_level > 1) Tic();
                solver.Compute(A, options);
                if (verbose_level > 1) Toc("TR: decomposition");
                VectorXr x0 = solver.Solve(rhs);
                

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
    
                    SetupJacobian(active_contact_idx);
                    VectorXr rhs;
                    rhs.resize(dofs()*2);
                    rhs.setZero();
                    
                    rhs.head(dofs()) = (1.0/3.0)*(-h) * v_next;
                    rhs.tail(dofs()).noalias() = (1.0/3.0) * (-h) * lumped_mass_inv * force_next;
                    VectorXr diff_tr;
                    diff_tr.resize(dofs()*2);
                    diff_tr.setZero();
                    diff_tr.head(dofs()) = (q_tr - q);
                    diff_tr.tail(dofs()) = (v_tr - v);
                    rhs -= (1.0/3.0) * diff_tr;
                    VectorXr diff_bdf;
                    diff_bdf.resize(dofs()*2);
                    diff_bdf.setZero();
                    diff_bdf.head(dofs()) = (q_next - q_tr);
                    diff_bdf.tail(dofs()) = (v_next - v_tr);
                    rhs += diff_bdf;
                    std::cout<<"current residual: "<<(rhs).norm()<<"\n";
                    PardisoSolver solver;
                    
                    A.resize((J12).rows(), (J12).cols());
                    
                    A.setIdentity();
                    A -= (1.0/3.0) * h * (J12 + J21);
                    

                    if (verbose_level > 1) std::cout<<"Solving for the first sparse J"<<std::endl;

                    if (verbose_level > 1) Tic();
                    solver.Compute(A, options);
                    if (verbose_level > 1) Toc("BDF FULL: decomposition");
                    VectorXr x0 = solver.Solve(rhs);
                    
                    q_next -= x0.head(dofs());
                    v_next -= x0.tail(dofs());
                    ApplyDirichlet(augmented_dirichlet, q_next, v_next);

                    if (verbose_level > 1) std::cout<<"calculating residual after a newton iteration"<<std::endl; 
                    rhs.head(dofs()) = (1.0/3.0) * (-h) * v_next;

                    VectorXr force_sol_new;
                    SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol_new);

                    rhs.tail(dofs()).noalias() = (1.0/3.0) * (-h) * lumped_mass_inv * force_sol_new;
                    rhs -= (1.0/3.0) * diff_tr;
                    diff_bdf.head(dofs()) = (q_next - q_tr);
                    diff_bdf.tail(dofs()) = (v_next - v_tr);
                    rhs += diff_bdf;
                    double residual = (rhs).norm();
                    std::cout<<"Residual: "<<residual<<std::endl;

                    if(si_method || residual < 1e-6){
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