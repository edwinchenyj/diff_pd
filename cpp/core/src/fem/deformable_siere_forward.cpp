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
        std::cout<<"forward siere\n";
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
        CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
        CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
        CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
        const int max_newton_iter = static_cast<int>(options.at("max_newton_iter"));
        const int max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
        const int verbose_level = static_cast<int>(options.at("verbose"));
        const int thread_ct = static_cast<int>(options.at("thread_ct"));
        CheckError(max_newton_iter > 0, "Invalid max_newton_iter: " + std::to_string(max_newton_iter));
        CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));

        omp_set_num_threads(thread_ct);

        const real h = dt;
        const VectorXr state_force = ForwardStateForce(q, v);
        std::vector<real> inv_h2_lumped_mass;
        std::transform(lumped_mass_.begin(),lumped_mass_.end(), std::back_inserter(inv_h2_lumped_mass),[&h](real mass)-> real { return mass/(h * h);});
        const int max_contact_iter = 5;
        const bool use_precomputed_data = !pd_element_energies_.empty();

        for (int contact_iter = 0; contact_iter < max_contact_iter; ++contact_iter) {
            if (verbose_level > 0) std::cout << "Contact iteration " << contact_iter << std::endl;
            // Fix dirichlet_ + active_contact_nodes.
            std::map<int, real> augmented_dirichlet = dirichlet_;

            for (const int idx : active_contact_idx) {
                for (int i = 0; i < vertex_dim; ++i)
                    augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
            }
            // Initial guess.
            VectorXr q_sol = q;
            VectorXr v_sol = v;
            VectorXr selected = VectorXr::Ones(dofs_);
            for (const auto& pair : augmented_dirichlet) {
                q_sol(pair.first) = pair.second;
                v_sol(pair.first) = 0;
                selected(pair.first) = 0;
            }
            if (use_precomputed_data) ComputeDeformationGradientAuxiliaryDataAndProjection(q_sol);
            VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol, use_precomputed_data) + ActuationForce(q_sol, a);
            for (const auto& pair : augmented_dirichlet) {
                force_sol(pair.first) = 0;
            }
            if (verbose_level > 1) Tic();
            SparseMatrix stiffness = StiffnessMatrix(q_sol, a, augmented_dirichlet, use_precomputed_data);
            #ifndef NDEBUG
            MatrixXr dense_stiffness;
            dense_stiffness = MatrixXr(stiffness);
            #endif
            if (verbose_level > 1) Toc("Assemble Stiffness Matrix");
            if (verbose_level > 1) Tic();
            SparseMatrix lumped_mass = LumpedMassMatrix(augmented_dirichlet);
            #ifndef NDEBUG
            MatrixXr dense_lumped_mass;
            dense_lumped_mass = MatrixXr(lumped_mass);
            #endif
            if (verbose_level > 1) Toc("Assemble Mass Matrix");
            if (verbose_level > 1) Tic();
            SparseMatrix lumped_mass_inv = LumpedMassMatrixInverse(augmented_dirichlet);
            #ifndef NDEBUG
            MatrixXr dense_lumped_mass_inv;
            dense_lumped_mass_inv = MatrixXr(lumped_mass_inv);
            #endif
            if (verbose_level > 1) Toc("Assemble Mass Matrix Inverse");
            // TODO: implement siere

            // SparseMatirx MinvK0;
            // MinvK0 = (1)*mass_lumped_inv.asDiagonal()*(K0_map);


            SparseMatrix MinvK = lumped_mass_inv * stiffness;
            #ifndef NDEBUG
            MatrixXr dense_MinvK;
            dense_MinvK = MatrixXr(MinvK);
            #endif
            

            Spectra::SparseSymShiftSolve<real> op(MinvK);
            
            int m_numModes = 5;

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
                tripletListJ12.push_back(T(i,i+dofs(),1.0));
            }
            J12.setFromTriplets(tripletListJ12.begin(),tripletListJ12.end());
            
                
            MatrixXr U1;
            MatrixXr V1;
            MatrixXr U2;
            MatrixXr V2;
            
            std::pair<MatrixXr, VectorXr > m_Us;
            std::pair<MatrixXr, VectorXr > m_Us2;
            
            MatrixXr dt_J_G_reduced;
            
            VectorXr vG;
            VectorXr vH;
            
            VectorXr fG;
            
            VectorXr fH;
            
            SparseMatrix A;
            
            SparseMatrix Identity;
            
            std::vector<int> J21_J22_outer_ind_ptr;
            std::vector<int> J21_outer_ind_ptr;
            std::vector<int> J22_outer_ind_ptr;
            std::vector<int> J22i_outer_ind_ptr;
            std::vector<int> J21_inner_ind;
            std::vector<int> J22_inner_ind;
            std::vector<int> J22i_inner_ind;
            std::vector<double> J22i_identity_val;
            std::vector<double> stiffness0_val;
            std::vector<int> stiffness0_outer_ind_ptr;
            std::vector<int> stiffness0_inner_ind;

            if(active_contact_idx.size() == 0){

                Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<real>> eigs(op, m_numModes+6, std::min(2*(m_numModes+6),dofs()), 0.01);
                
                // Initialize and compute
                eigs.init();
                eigs.compute(Spectra::SortRule::LargestMagn);
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
            else{
                Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<real>> eigs(op, m_numModes, 2*m_numModes, 0.01);
                
                // Initialize and compute
                eigs.init();
                eigs.compute(Spectra::SortRule::LargestMagn);
                Eigen::VectorXd normalizing_const;
                if(eigs.info() == Spectra::CompInfo::Successful)
                {
                    m_Us = std::make_pair(eigs.eigenvectors().real(), eigs.eigenvalues().real());
                    normalizing_const.noalias() = (m_Us.first.transpose() * lumped_mass * m_Us.first).diagonal();
                    normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
                    
                    m_Us.first = m_Us.first * (normalizing_const.asDiagonal());
                }
                else{
                    std::cout<<"eigen solve failed"<<std::endl;
                    exit(1);
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
            

            U1.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
            V1.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << lumped_mass * m_Us.first;
            U2.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first * (m_Us.second.asDiagonal());
            V2.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << lumped_mass * m_Us.first;
            
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
            
            #ifndef NDEBUG
            MatrixXr dense_J12, dense_J21;
            dense_J12 = MatrixXr(J12);
            dense_J21 = MatrixXr(J21_map);
            #endif
            
            A.resize((J12).rows(), (J12).cols());
            
            A.setIdentity();
            A -= h * (J12 - J21_map);
             #ifndef NDEBUG
            MatrixXr dense_A;
            dense_A = MatrixXr(A);
            #endif
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
            reduced_vec.head(dt_J_G_reduced.cols()/2).noalias() = m_Us.first.transpose() * (lumped_mass * (q_sol));
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
            
            rhs2.noalias() = (-h) * block_diag_eigv * phi_reduced * reduced_vec;
            
            rhs = rhs1 + rhs2;
            
            PardisoSolver solver;
            if (verbose_level > 1) Toc("SIERE: decomposition");
            solver.Compute(A, options);
            if (verbose_level > 1) Tic();
            VectorXr x0 = solver.Solve(rhs);
            
            U1 *= h;
            MatrixXr x1;
            x1 = solver.Solve(U1);
            
            U2 *= dt;
            MatrixXr x2;
            x2 = solver.Solve(U2);
            
            
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
            

            q_next -= y0.head(dofs());
            v_next -= y0.tail(dofs());
            break; // skip contact for now

            
        }


    }




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
