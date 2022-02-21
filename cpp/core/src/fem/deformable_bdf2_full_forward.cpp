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
void Deformable<vertex_dim, element_dim>::ForwardBDF2FULL(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        const int verbose_level = static_cast<int>(options.at("verbose"));
        if (verbose_level > 1) std::cout<<"forward bdf 2 full\n";
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
        bool si_method = false;
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
        q_next = q;
        v_next = v;
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            if (verbose_level > 0) std::cout << "Contact iteration " << contact_iter << std::endl;
            std::map<int, real> augmented_dirichlet = dirichlet_;
            if (verbose_level > 1) std::cout<<"before active contact idx\n";
            if (verbose_level > 1) std::cout<<active_contact_idx.size()<<"\n";
            for (const int idx : active_contact_idx) {
                for (int i = 0; i < vertex_dim; ++i){
                    augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
                }
            }
            if (q_prev.size() != q.size()) {
                if (verbose_level > 1) std::cout<<"initialize 2 steps\n";
                q_prev.resize(q.size());
                v_prev.resize(v.size());
                q_prev = q;
                v_prev = v;
            }

            for (size_t i = 0; i < 20; i++)
            {
                if (verbose_level > 1) std::cout<<"BDF newton iteration "<<i<<"\n";

                ApplyDirichlet(augmented_dirichlet, q_next, v_next);
            
                SetupMatrices(q_next, a, augmented_dirichlet, use_precomputed_data);
                VectorXr force_sol;
                SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol);
                
                SetupJacobian(active_contact_idx);
                VectorXr rhs;
                rhs.resize(dofs()*2);
                
                rhs.head(dofs()) = (2.0/3.0)*(-h) * v_next;
                rhs.tail(dofs()).noalias() = (2.0/3.0) * (-h) * lumped_mass_inv * force_sol;
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
                A -= (2.0/3.0) * h * (J12 + J21);

                if (verbose_level > 1) Tic();
                solver.Compute(A, options);
                if (verbose_level > 1) Toc("BDF FULL: decomposition");
                VectorXr x0 = solver.Solve(rhs);
                
                q_next -= x0.head(dofs());
                v_next -= x0.tail(dofs());
                ApplyDirichlet(augmented_dirichlet, q_next, v_next);


                if (verbose_level > 1) std::cout<<"calculating residual after a newton iteration"<<std::endl; 
                rhs.head(dofs()) = (2.0/3.0) * (-h) * v_next;
                VectorXr force_sol_new;
                SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol_new);
                rhs.tail(dofs()).noalias() = (2.0/3.0) * (-h) * lumped_mass_inv * force_sol_new;
                rhs -= (1.0/3.0) * diff_prev;
                diff.head(dofs()) = q_next - q;
                diff.tail(dofs()) = v_next - v;
                rhs += diff;
                double residual = (rhs).norm();
                std::cout<<"Residual: "<<residual<<std::endl;
                if (si_method || residual < 1e-6) {
                    break;
                }
            }
            q_prev = q;
            v_prev = v;
            break; // skip contact for now
        }


    }




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
