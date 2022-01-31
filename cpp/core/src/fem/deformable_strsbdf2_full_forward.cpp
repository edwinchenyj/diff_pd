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
void Deformable<vertex_dim, element_dim>::ForwardSTRSBDF2FULL(const std::string& method,
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
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            if (verbose_level > 0) std::cout << "Contact iteration " << contact_iter << std::endl;
            // Fix dirichlet_ + active_contact_nodes.
            if (verbose_level > 1) std::cout<<"before fix dirichlet\n";
            std::map<int, real> augmented_dirichlet = dirichlet_;
            // PrintVector(q);

            if (verbose_level > 1) std::cout<<"before active contact idx\n";
            if (verbose_level > 1) std::cout<<active_contact_idx.size()<<"\n";
            for (const int idx : active_contact_idx) {
                for (int i = 0; i < vertex_dim; ++i){
                    augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
                }
            }
            // Initial guess.
            VectorXr q_sol = q;
            VectorXr v_sol = v;
            if (q_prev.size() != q.size()) {
                if (verbose_level > 1) std::cout<<"initialize 2 steps\n";
                q_prev.resize(q.size());
                v_prev.resize(v.size());
                q_prev = q;
                v_prev = v;
            }
            if (verbose_level > 1) std::cout<<"before selected\n";
            VectorXr selected = VectorXr::Ones(dofs_);

            for (size_t i = 0; i < 20; i++)
            {
                if (verbose_level > 1) std::cout<<"BDF newton iteration "<<i<<"\n";
            
                for (const auto& pair : augmented_dirichlet) {
                    q_sol(pair.first) = pair.second;
                    v_sol(pair.first) = 0;
                    selected(pair.first) = 0;
                }

                
                if (verbose_level > 1) std::cout<<"before compute stiffness matrix\n";
                if (verbose_level > 1) Tic();
                SparseMatrix stiffness = -StiffnessMatrix(q_sol, a, augmented_dirichlet, use_precomputed_data);
                if (verbose_level > 1) Toc("Assemble Stiffness Matrix");
                if (verbose_level > 1) Tic();

                if (verbose_level > 1) std::cout<<"before compute mass matrix\n";
                SparseMatrix lumped_mass = LumpedMassMatrix(augmented_dirichlet);
                if (verbose_level > 1) std::cout<<"g:\n";
                if (verbose_level > 1) PrintVector(g);
                const VectorXr gravitational_force = lumped_mass * g.replicate(dofs()/vertex_dim, 1);
                if (verbose_level > 1) Toc("Assemble Mass Matrix");
                if (verbose_level > 1) Tic();
                SparseMatrix lumped_mass_inv = LumpedMassMatrixInverse(augmented_dirichlet);
                if (verbose_level > 1) Toc("Assemble Mass Matrix Inverse");

                VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol, use_precomputed_data) + ActuationForce(q_sol, a) + gravitational_force;
                for (const auto& pair : augmented_dirichlet) {
                    force_sol(pair.first) = 0;
                }
                
                // SparseMatirx MinvK0;
                // MinvK0 = (1)*mass_lumped_inv.asDiagonal()*(K0_map);


                SparseMatrix MinvK = lumped_mass_inv * stiffness;

                SparseMatrix J12;
                SparseMatrix J21;
                SparseMatrix J22;

                J12.resize(dofs()*2,dofs()*2);
                J21.resize(dofs()*2,dofs()*2);
                J22.resize(dofs()*2,dofs()*2);

                
                typedef Eigen::Triplet<double> T;
                std::vector<T> tripletListJ12;
                tripletListJ12.reserve(dofs());
                for(int i = 0; i < dofs(); i++)
                {
                    if(std::find(active_contact_idx.begin(),active_contact_idx.end(),i/vertex_dim) == active_contact_idx.end()){          
                        tripletListJ12.push_back(T(i,i+dofs(),1.0));
                    }
                }
                J12.setFromTriplets(tripletListJ12.begin(),tripletListJ12.end());
                
                
                
                    
                J21_J22_outer_ind_ptr.erase(J21_J22_outer_ind_ptr.begin(),J21_J22_outer_ind_ptr.end());
                J21_outer_ind_ptr.erase(J21_outer_ind_ptr.begin(),J21_outer_ind_ptr.end());
                J22i_outer_ind_ptr.erase(J22i_outer_ind_ptr.begin(),J22i_outer_ind_ptr.end());
                for (int i_row = 0; i_row < MinvK.rows(); i_row++) {
                    J22i_outer_ind_ptr.push_back(0);
                    J22_outer_ind_ptr.push_back(0);
                }
                
                J21_outer_ind_ptr.erase(J21_outer_ind_ptr.begin(),J21_outer_ind_ptr.end());

                J22i_inner_ind.erase(J22i_inner_ind.begin(),J22i_inner_ind.end());
                J22i_identity_val.erase(J22i_identity_val.begin(),J22i_identity_val.end());
                for (int i_row = 0; i_row < MinvK.rows() + 1; i_row++) {
                    J21_J22_outer_ind_ptr.push_back(*(MinvK.outerIndexPtr()+i_row));
                    J21_outer_ind_ptr.push_back(*(MinvK.outerIndexPtr()+i_row));
                    J22i_outer_ind_ptr.push_back(i_row);
                    J22i_inner_ind.push_back(i_row + MinvK.rows());
                    J22i_identity_val.push_back(1.0);
                }

                for (int i_row = 0; i_row < MinvK.rows(); i_row++) {
                    J21_outer_ind_ptr.push_back(0);
                    J22i_outer_ind_ptr.push_back(0);
                    J22_outer_ind_ptr.push_back(0);
                }
                
                J21_inner_ind.erase(J21_inner_ind.begin(),J21_inner_ind.end());
                J22_inner_ind.erase(J22_inner_ind.begin(),J22_inner_ind.end());
                
                for (int i_nnz = 0; i_nnz < MinvK.nonZeros(); i_nnz++)
                {
                    J21_inner_ind.push_back(*(MinvK.innerIndexPtr()+i_nnz) + MinvK.cols());
                    J22_inner_ind.push_back(*(MinvK.innerIndexPtr()+i_nnz) + MinvK.cols());
                }
                
                Eigen::Map<SparseMatrix> J21_map((MinvK.rows())*2, (MinvK.cols())*2, MinvK.nonZeros(), J21_outer_ind_ptr.data(), J21_inner_ind.data(), (MinvK).valuePtr());
                Eigen::Map<SparseMatrix> J22_map((MinvK.rows())*2, (MinvK.cols())*2, MinvK.nonZeros(), J21_J22_outer_ind_ptr.data(), J22_inner_ind.data(), (MinvK).valuePtr());
                Eigen::Map<SparseMatrix> J22i_map((MinvK.rows())*2, (MinvK.cols())*2, MinvK.cols(), J22i_outer_ind_ptr.data(), J22i_inner_ind.data(),J22i_identity_val.data());
                
                VectorXr rhs;
                rhs.resize(dofs()*2);
                
                rhs.head(dofs()) = (2.0/3.0)*(-h) * v_sol;
                rhs.tail(dofs()).noalias() = (2.0/3.0) * (-h) * lumped_mass_inv * force_sol;
                VectorXr diff_prev;
                diff_prev.resize(dofs()*2);
                diff_prev.setZero();
                diff_prev.head(dofs()) = (q - q_prev);
                diff_prev.tail(dofs()) = (v - v_prev);
                rhs -= (1.0/3.0) * diff_prev;
                VectorXr diff;
                diff.resize(dofs()*2);
                diff.head(dofs()) = q_sol - q;
                diff.tail(dofs()) = v_sol - v;
                rhs += diff;
                std::cout<<"current residual: "<<(rhs).norm()<<"\n";
                PardisoSolver solver;
                
                A.resize((J12).rows(), (J12).cols());
                
                A.setIdentity();
                A -= (2.0/3.0) * h * (J12 + J21_map);
                
                
                if (verbose_level > 1) std::cout<<"Solving for the first sparse J"<<std::endl;

                if (verbose_level > 1) Tic();
                solver.Compute(A, options);
                if (verbose_level > 1) Toc("BDF FULL: decomposition");
                VectorXr x0 = solver.Solve(rhs);
                
                for(auto i: active_contact_idx){
                    for(int j = 0; j < vertex_dim; j++){
                        x0.row(i*vertex_dim+j).setZero();
                        x0.row(i*vertex_dim+j+dofs()).setZero();
                    }
                }
                // q_sol = q;
                // v_sol = v;
                q_sol -= x0.head(dofs());
                v_sol -= x0.tail(dofs());
                for (const auto& pair : augmented_dirichlet) {
                    q_sol(pair.first) = pair.second;
                    v_sol(pair.first) = 0;
                    selected(pair.first) = 0;
                }


                if (verbose_level > 1) std::cout<<"calculating residual after a newton iteration"<<std::endl; 
                rhs.head(dofs()) = (2.0/3.0) * (-h) * v_sol;
                VectorXr force_sol_new = ElasticForce(q_sol) + PdEnergyForce(q_sol, use_precomputed_data) + ActuationForce(q_sol, a) + gravitational_force;
                for (const auto& pair : augmented_dirichlet) {
                    force_sol_new(pair.first) = 0;
                }
                rhs.tail(dofs()).noalias() = (2.0/3.0) * (-h) * lumped_mass_inv * force_sol_new;
                rhs -= (1.0/3.0) * diff_prev;
                diff.head(dofs()) = q_sol - q;
                diff.tail(dofs()) = v_sol - v;
                rhs += diff;
                double residual = (rhs).norm();
                std::cout<<"Residual: "<<residual<<std::endl;
                if (residual < 1e-6) {
                    break;
                }
            }
            q_prev = q;
            v_prev = v;
            q_next = q_sol;
            v_next = v_sol; 
            break; // skip contact for now
        }


    }




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
