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
        const int verbose_level = static_cast<int>(options.at("verbose"));
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
        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            if (verbose_level > 0) std::cout << "Contact iteration " << contact_iter << std::endl;
            // Fix dirichlet_ + active_contact_nodes.
            if (verbose_level > 1) std::cout<<"before fix dirichlet\n";
            std::map<int, real> augmented_dirichlet = dirichlet_;
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
            if (verbose_level > 1) std::cout<<"before selected\n";
            VectorXr selected = VectorXr::Ones(dofs_);
            for (const auto& pair : augmented_dirichlet) {
                q_sol(pair.first) = pair.second;
                v_sol(pair.first) = 0;
                selected(pair.first) = 0;
            }

            // std::cout<<"before use_precomputed_data\n";
            // if (use_precomputed_data) ComputeDeformationGradientAuxiliaryDataAndProjection(q_sol);
            
            if (verbose_level > 1) std::cout<<"before compute stiffness matrix\n";
            if (verbose_level > 1) Tic();
            SparseMatrix stiffness = -StiffnessMatrix(q_sol, a, augmented_dirichlet, use_precomputed_data);
            // #ifndef NDEBUG
            // MatrixXr dense_stiffness;
            // dense_stiffness = MatrixXr(stiffness);
            // #endif
            if (verbose_level > 1) Toc("Assemble Stiffness Matrix");
            if (verbose_level > 1) Tic();

            if (verbose_level > 1) std::cout<<"before compute mass matrix\n";
            SparseMatrix lumped_mass = LumpedMassMatrix(augmented_dirichlet);
            // std::cout<<"lumped_mass: "<<lumped_mass.nonZeros()<<"\n";
            if (verbose_level > 1) std::cout<<"g:\n";
            const VectorXr gravitational_force = lumped_mass * g.replicate(dofs()/vertex_dim, 1);
            // std::cout<<"gravitational_force:\n";
            // #endif
            if (verbose_level > 1) Toc("Assemble Mass Matrix");
            if (verbose_level > 1) Tic();
            SparseMatrix lumped_mass_inv = LumpedMassMatrixInverse(augmented_dirichlet);
            // #ifndef NDEBUG
            // MatrixXr dense_lumped_mass_inv;
            // dense_lumped_mass_inv = MatrixXr(lumped_mass_inv);
            // #endif
            if (verbose_level > 1) Toc("Assemble Mass Matrix Inverse");

            if (verbose_level > 1) std::cout<<"force\n";
            VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol, use_precomputed_data) + ActuationForce(q_sol, a) + gravitational_force;
            for (const auto& pair : augmented_dirichlet) {
                force_sol(pair.first) = 0;
            }
            

            SparseMatrix MinvK = lumped_mass_inv * stiffness;
            // #ifndef NDEBUG
            // MatrixXr dense_MinvK;
            // dense_MinvK = MatrixXr(MinvK);
            // #endif
            
            if (verbose_level > 1) std::cout<<"spectra op\n";
            Spectra::SparseGenRealShiftSolvePardiso<real> op(MinvK);
            
            int m_numModes = num_modes;

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
            
            
            if (verbose_level > 1) std::cout<<"eigen solve:\n";
            int DecomposedDim = std::max(m_numModes+2*vertex_dim,m_numModes + vertex_dim * (int)active_contact_idx.size());
            Spectra::GenEigsRealShiftSolver<Spectra::SparseGenRealShiftSolvePardiso<real>> eigs(op, DecomposedDim, std::min(2*(DecomposedDim),dofs()), 0.01);
            
            VectorXr ritz_error = VectorXr::Zero(DecomposedDim);

            
            if(m_Us.second.sum() != 0){
                ritz_error = (MinvK * m_Us.first - m_Us.first * m_Us.second.asDiagonal()).colwise().norm();
                ritz_error_norm = ritz_error.maxCoeff();

            }

            if(true){
            // if(ritz_error_norm > 1){

                
                // Initialize and compute
                eigs.init();
                Tic();
                eigs.compute(Spectra::SortRule::LargestMagn);
                Toc("Eigen Solve");
                Eigen::VectorXd normalizing_const;
                if(eigs.info() == Spectra::CompInfo::Successful)
                {
                    m_Us = std::make_pair(eigs.eigenvectors().real().leftCols(m_numModes), eigs.eigenvalues().real().head(m_numModes));
                    normalizing_const.noalias() = (m_Us.first.transpose() * lumped_mass * m_Us.first).diagonal();
                    normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
                    
                    m_Us.first = m_Us.first * (normalizing_const.asDiagonal());

                }
                else{
                    std::cout<<"eigen solve failed"<<std::endl;
                    exit(1);
                }
            }

            for(auto i: active_contact_idx){
                for(int j = 0; j < vertex_dim; j++){
                    m_Us.first.row(i*vertex_dim+j).setZero();
                }
            }
        
                
            J21_J22_outer_ind_ptr.erase(J21_J22_outer_ind_ptr.begin(),J21_J22_outer_ind_ptr.end());
            J21_outer_ind_ptr.erase(J21_outer_ind_ptr.begin(),J21_outer_ind_ptr.end());
            J22i_outer_ind_ptr.erase(J22i_outer_ind_ptr.begin(),J22i_outer_ind_ptr.end());
            for (int i_row = 0; i_row < MinvK.rows(); i_row++) {
                // J21_J22_outer_ind_ptr.push_back(0);
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
            
            
            U1.resize((MinvK.rows())*2,m_numModes);
            V1.resize((MinvK.rows())*2,m_numModes);
            U2.resize((MinvK.rows())*2,m_numModes);
            V2.resize((MinvK.rows())*2,m_numModes);
            
            U1.setZero();
            V1.setZero();
            U2.setZero();
            V2.setZero();
            
            MatrixXr inertial = lumped_mass * m_Us.first;

            for(auto i: active_contact_idx){
                for(int j = 0; j < vertex_dim; j++){
                    inertial.row(i*vertex_dim+j).setZero();
                }
            }


            U1.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
            V1.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << inertial;
            U2.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first * (m_Us.second.asDiagonal());
            V2.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << inertial;
            
            for(auto i: active_contact_idx){
                for(int j = 0; j < vertex_dim; j++){
                    U1.row(i*vertex_dim+j).setZero();
                    U2.row(i*vertex_dim+j).setZero();
                    V1.row(i*vertex_dim+j).setZero();
                    V2.row(i*vertex_dim+j).setZero();
                    U1.row(i*vertex_dim+j + dofs()).setZero();
                    U2.row(i*vertex_dim+j + dofs()).setZero();
                    V1.row(i*vertex_dim+j + dofs()).setZero();
                    V2.row(i*vertex_dim+j + dofs()).setZero();
                }
            }


            dt_J_G_reduced.resize(m_numModes*2,m_numModes*2);
            dt_J_G_reduced.setZero();
            dt_J_G_reduced.block(0,m_Us.first.cols(),m_Us.first.cols(),m_Us.first.cols()).setIdentity();
            for (int ind = 0; ind < m_Us.first.cols() ; ind++) {
                dt_J_G_reduced(m_Us.first.cols() + ind ,0 + ind ) = m_Us.second(ind);
            }
            dt_J_G_reduced *= h;
            
            vG.noalias() = m_Us.first * (m_Us.first.transpose() * lumped_mass * v_sol);
            
            vH = -vG;
            vH.noalias() += v_sol;
            
            fG.noalias() = (lumped_mass * m_Us.first ) * (m_Us.first.transpose() * force_sol);
            fH = force_sol - fG;
            
            // #ifndef NDEBUG
            // MatrixXr dense_J12, dense_J21;
            // dense_J12 = MatrixXr(J12);
            // dense_J21 = MatrixXr(J21_map);
            // #endif
            
            A.resize((J12).rows(), (J12).cols());
            
            A.setIdentity();
            A -= h * (J12 + J21_map);
            
            //  #ifndef NDEBUG
            // MatrixXr dense_A;
            // dense_A = MatrixXr(A);
            // #endif
            VectorXr rhs1;
            rhs1.resize(dofs()*2);
            VectorXr rhs2;
            rhs2.resize(dofs()*2);
            VectorXr rhs;
            rhs.resize(dofs()*2);
            
            rhs1.head(dofs()) = (-h) * vH;
            rhs1.tail(dofs()).noalias() = (-h) * lumped_mass_inv * fH;
            
            VectorXr reduced_vec;
            reduced_vec.resize(dt_J_G_reduced.cols());
            reduced_vec.head(dt_J_G_reduced.cols()/2).noalias() = m_Us.first.transpose() * (lumped_mass * (v_sol));
            reduced_vec.tail(dt_J_G_reduced.cols()/2).noalias() = (m_Us.first.transpose() * (force_sol));
            
            MatrixXr block_diag_eigv;
            
            block_diag_eigv.resize(m_Us.first.rows()*2,m_Us.first.cols()*2);
            block_diag_eigv.setZero();
            block_diag_eigv.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
            block_diag_eigv.block(m_Us.first.rows(),m_Us.first.cols(),m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
            
            MatrixXr phi_reduced;
            phi_reduced.resize(m_Us.first.cols()*2,m_Us.first.cols()*2);
            phi_reduced.setZero();
            
            phi((dt_J_G_reduced), phi_reduced);
            if (verbose_level > 1) std::cout<<"phi: "<<std::endl;
            
            rhs2.noalias() = (-h) * block_diag_eigv * phi_reduced * reduced_vec;
            
            rhs = rhs1 + rhs2;
            std::cout<<"current residual: "<<rhs.norm()<<std::endl;
            PardisoSolver solver;
            
            if (verbose_level > 1) std::cout<<"Solving for the first sparse J"<<std::endl;

            if (verbose_level > 1) Tic();
            solver.Compute(A, options);
            if (verbose_level > 1) Toc("SIERE: decomposition");
            VectorXr x0 = solver.Solve(rhs);
            
            if (verbose_level > 1) std::cout<<"x0: "<<std::endl;
            for(auto i: active_contact_idx){
                for(int j = 0; j < vertex_dim; j++){
                    x0.row(i*vertex_dim+j).setZero();
                    x0.row(i*vertex_dim+j+dofs()).setZero();
                }
            }

            U1 *= h;
            MatrixXr x1;
            x1 = solver.Solve(U1);

            for(auto i: active_contact_idx){
                for(int j = 0; j < vertex_dim; j++){
                    x1.row(i*vertex_dim+j).setZero();
                    x1.row(i*vertex_dim+j+dofs()).setZero();
                }
            }

            
            U2 *= dt;
            MatrixXr x2;
            x2 = solver.Solve(U2);

            for(auto i: active_contact_idx){
                for(int j = 0; j < vertex_dim; j++){
                    x2.row(i*vertex_dim+j).setZero();
                    x2.row(i*vertex_dim+j+dofs()).setZero();
                }
            }

            
            if (verbose_level > 1) std::cout<<"Solving for the SMW"<<std::endl;
            MatrixXr Is;
            Is.resize(U1.cols(),U1.cols());
            Is.setIdentity();
            
            MatrixXr yLHS = Is + V1.transpose()*x1;
            VectorXr y0;
            y0 = x0;
            y0.noalias() -= x1 * yLHS.ldlt().solve(V1.transpose()*x0);
            VectorXr y1;
            y1.resize(x2.rows());
            MatrixXr yRHS2 = V1.transpose()*x2;
            x2.noalias() -= x1 * (yLHS.ldlt().solve(yRHS2));
            MatrixXr sol2LHS = Is + V2.transpose()*x2;
            VectorXr sol2;
            MatrixXr sol2RHS = V2.transpose()*y0;
            y0.noalias() -= x2 * (sol2LHS).ldlt().solve(sol2RHS);
            
            if (verbose_level > 1) std::cout<<"y0: "<<std::endl;

            q_next = q;
            v_next = v;
            q_next -= y0.head(dofs());
            v_next -= y0.tail(dofs());
            
            if (verbose_level > 1) std::cout<<"Calculating residual\n";
            if (verbose_level > 1) Tic();
            for (const auto& pair : augmented_dirichlet) {
                q_next(pair.first) = pair.second;
                v_next(pair.first) = 0;
            }


            if (verbose_level > 1) std::cout<<"Construct force for the next step\n";
            VectorXr force_sol_new = ElasticForce(q_next) + PdEnergyForce(q_next, use_precomputed_data) + ActuationForce(q_next, a) + gravitational_force;
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
