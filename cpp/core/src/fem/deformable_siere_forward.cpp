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


void phi(MatrixXr &A, MatrixXr &output)
{
    
    
    
    Eigen::EigenSolver<MatrixXr> es(A);
    Eigen::MatrixXcd D;
    D.resize(A.rows(),A.rows());
    D.setZero();
    D = es.eigenvalues().asDiagonal();
    Eigen::MatrixXcd D_new;
    D_new.resize(A.rows(),A.rows());
    D_new.setZero();
    
    for (int j = 0; j < D.rows(); j++) {
        if(norm(D(j,j)) > 1e-8)
        {
            std::complex<double> tempc;
            tempc.real(exp(D(j,j)).real() - 1);
            tempc.imag(exp(D(j,j)).imag());
            tempc = tempc/D(j,j);
            D_new(j,j).real(tempc.real());
            D_new(j,j).imag(tempc.imag());
        }
        else
        {
            D_new(j,j).real(1.0);
            D_new(j,j).imag(0.0);
            
        }
    }
    //
    Eigen::MatrixXcd U;
    U = es.eigenvectors();
    output = ((U) * (D_new) * (U.inverse())).real();
}


template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardSIERE(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        verbose_level = static_cast<int>(options.at("verbose"));
        if (verbose_level > 1) std::cout<<"forward siere\n";
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
        if (verbose_level > 1) std::cout<<"thread_ct: "<<thread_ct<<"\n";
        omp_set_num_threads(thread_ct);
        const real h = dt;
        VectorXr g;
        if (verbose_level > 1) std::cout<<"before g\n";
        CheckError(state_forces_.size() <= 1, "Only one state force, gravity, is supported for SIERE");
        if(state_forces_.size() == 1) {
            g = state_forces_[0]->parameters().head(vertex_dim);
        } else {
            g = VectorXr::Zero(vertex_dim);
        }

        const bool recompute_eigen_decomp_each_step = static_cast<bool>(options.at("recompute_eigen_decomp_each_step"));
        if (verbose_level > 1) std::cout<<"recompute_eigen_decomp_each_step: "<<recompute_eigen_decomp_each_step<<"\n";
        
        const int num_modes = static_cast<int>(options.at("num_modes"));
        if (verbose_level > 1) std::cout<<"num_modes: "<<num_modes<<"\n";

        
        std::vector<real> inv_h2_lumped_mass;
        std::transform(lumped_mass_.begin(),lumped_mass_.end(), std::back_inserter(inv_h2_lumped_mass),[&h](real mass)-> real { return mass/(h * h);});
        if (verbose_level > 1) std::cout<<"lumped_mass_ size:" << lumped_mass_.size()<<"\n";
        const int max_contact_iter = 5;
        const bool use_precomputed_data = !pd_element_energies_.empty();

        if (verbose_level > 1) std::cout<<"main loop\n";
        // Initial guess
        q_next = q;
        v_next = v;
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            if (verbose_level > 0) std::cout << "Contact iteration " << contact_iter << std::endl;
            // Fix dirichlet_ + active_contact_nodes.
            std::map<int, real> augmented_dirichlet = dirichlet_;
            if (verbose_level > 1) std::cout<<"active contact idx\n";
            if (verbose_level > 1) std::cout<<active_contact_idx.size()<<"\n";
            for (const int idx : active_contact_idx) {
                for (int i = 0; i < vertex_dim; ++i){
                    augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
                }
            }
            ApplyDirichlet(augmented_dirichlet, q_next, v_next);

            SetupMatrices(q_next, a, augmented_dirichlet, use_precomputed_data);
            VectorXr force_sol;
            SimpleForce(q_next, a, augmented_dirichlet, use_precomputed_data, g, force_sol);
            ApplyDirichlet(dirichlet_, force_sol);
            
            MassPCA(lumped_mass, MinvK, num_modes, active_contact_idx.size());

            for(auto i: active_contact_idx){
                for(int j = 0; j < vertex_dim; j++){
                    m_Us.first.row(i*vertex_dim+j).setZero();
                }
            }
            ComputeProjection(active_contact_idx); 
            SplitVelocityState(v_next);
            SplitForceState(force_sol);
            
            
            SetupJacobian(active_contact_idx);
            A.resize((J12).rows(), (J12).cols());
            
            A.setIdentity();
            A -= h * (J12+ J21);
            
            VectorXr rhs1;
            rhs1.resize(dofs()*2);
            VectorXr rhs2;
            rhs2.resize(dofs()*2);
            VectorXr rhs;
            rhs.resize(dofs()*2);
            
            rhs1.head(dofs()) = (-h) * vH;
            rhs1.tail(dofs()).noalias() = (-h) * lumped_mass_inv * fH;

            ComputeReducedRhs(rhs2, v_next, force_sol, h);

            rhs = rhs1 + rhs2;
            std::cout<<"current residual: "<<rhs.norm()<<std::endl;
            PardisoSolver solver;
            
            if (verbose_level > 1) std::cout<<"Solving for the first sparse J"<<std::endl;

            if (verbose_level > 1) Tic();
            solver.Compute(A, options);
            if (verbose_level > 1) Toc("SIERE: decomposition");
            VectorXr x0 = solver.Solve(rhs);

            SubspaceEREUpdate(x0, solver, h);
            q_next = q;
            v_next = v;
            q_next -= x0.head(dofs());
            v_next -= x0.tail(dofs());
            
            if (verbose_level > 1) std::cout<<"Calculating residual\n";
            if (verbose_level > 1) Tic();
            for (const auto& pair : augmented_dirichlet) {
                q_next(pair.first) = pair.second;
                v_next(pair.first) = 0;
            }


            if (verbose_level > 1) std::cout<<"Construct force for the next step\n";
            VectorXr force_sol_new; 
            SimpleForce(q_next, a, dirichlet_, use_precomputed_data, g, force_sol_new);
            for (const auto& pair : augmented_dirichlet) {
                force_sol_new(pair.first) = 0;
            }

            vH = -vG;
            vH.noalias() += v_next;
            
            fH = force_sol_new - fG;
            
            rhs1.head(dofs()) = (-h) * vH;
            rhs1.tail(dofs()).noalias() = (-h) * lumped_mass_inv * fH;
             
            if (verbose_level > 1) std::cout<<"Construct rhs for the next step\n"; 
            rhs = rhs1 + rhs2;
            VectorXr diff;
            diff.resize(dofs()*2);
            diff.head(dofs()) = q_next - q;
            diff.tail(dofs()) = v_next - v;
            rhs += diff;
            double residual = (rhs).norm();
            std::cout<<"Residual: "<<residual<<std::endl;
            
            if (verbose_level > 1) Toc("Calculating residual");
            break; // skip contact for now

            
        }


    }




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
