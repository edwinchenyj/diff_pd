#include "fem/deformable.h"
#include "common/common.h"
#include "solver/matrix_op.h"
#include "solver/pardiso_spd_solver.h"
#include "Eigen/SparseCholesky"
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ForwardSIBE(const std::string& method,
    const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
    const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next, std::vector<int>& active_contact_idx) const {
        std::cout<<"forward sibe\n";
        CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
        CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
        CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
        CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
        const int verbose_level = static_cast<int>(options.at("verbose"));
        const int thread_ct = static_cast<int>(options.at("thread_ct"));
        
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
            VectorXr selected = VectorXr::Ones(dofs_);
            for (const auto& pair : augmented_dirichlet) {
                q_sol(pair.first) = pair.second;
                selected(pair.first) = 0;
            }
            if (use_precomputed_data) ComputeDeformationGradientAuxiliaryDataAndProjection(q_sol);
            VectorXr force_sol = ElasticForce(q_sol) + PdEnergyForce(q_sol, use_precomputed_data) + ActuationForce(q_sol, a);
            if (verbose_level > 1) Tic();
            SparseMatrix stiffness = StiffnessMatrix(q_sol, a, augmented_dirichlet, use_precomputed_data);
            if (verbose_level > 1) Toc("Assemble Stiffness Matrix");
            if (verbose_level > 1) Tic();
            SparseMatrix lumped_mass = LumpedMassMatrix(augmented_dirichlet);
            if (verbose_level > 1) Toc("Assemble Mass Matrix");
            SparseMatrix A = lumped_mass + h*h * stiffness;
            VectorXr b = h * ((force_sol + state_force) -h*stiffness*v).array() * selected.array();

            std::cout<<"current residual: "<<( h * (force_sol + state_force)).norm()<<"\n";

            if (verbose_level > 1) Tic();
            PardisoSpdSolver solver;
            solver.Compute(A, options);
            if (verbose_level > 1) Toc("sibe-Pardiso: decomposition");
            if (verbose_level > 1) Tic();
            VectorXr dv = solver.Solve(b);
            if (verbose_level > 1) Toc("sibe-Pardiso: solve");

            v_next = v + dv;
            q_next = q + h * v_next;
            VectorXr force_sol_new = ElasticForce(q_next) + PdEnergyForce(q_next, use_precomputed_data) + ActuationForce(q_next, a);
            double residual = (lumped_mass * dv - h * (force_sol_new + state_force)).norm();
            std::cout<<"Residual: "<<residual<<"\n";
            break; // skip contact for now

            
        }


    }

    

template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::NewtonMatrix(const VectorXr& q_sol, const VectorXr& a,
    const std::vector<real> inv_h2m, const std::map<int, real>& dirichlet_with_friction, const bool use_precomputed_data) const {
    SparseMatrixElements nonzeros = ElasticForceDifferential(q_sol);
    // SparseMatrixElements nonzeros_pd, nonzeros_dummy;
    // PdEnergyForceDifferential(q_sol, true, false, use_precomputed_data, nonzeros_pd, nonzeros_dummy);
    // SparseMatrixElements nonzeros_act_dq, nonzeros_act_da, nonzeros_act_dw;
    // ActuationForceDifferential(q_sol, a, nonzeros_act_dq, nonzeros_act_da, nonzeros_act_dw);
    // nonzeros.insert(nonzeros.end(), nonzeros_pd.begin(), nonzeros_pd.end());
    // nonzeros.insert(nonzeros.end(), nonzeros_act_dq.begin(), nonzeros_act_dq.end());
    SparseMatrixElements nonzeros_new;
    for (const auto& element : nonzeros) {
        const int row = element.row();
        const int col = element.col();
        const real val = element.value();
        if (dirichlet_with_friction.find(row) != dirichlet_with_friction.end()
            || dirichlet_with_friction.find(col) != dirichlet_with_friction.end()) continue;
        nonzeros_new.push_back(Eigen::Triplet<real>(row, col, -val));
    }

    for (int i = 0; i < dofs_; ++i) {
        if (dirichlet_with_friction.find(i) != dirichlet_with_friction.end())
            nonzeros_new.push_back(Eigen::Triplet<real>(i, i, 1));
        else
            nonzeros_new.push_back(Eigen::Triplet<real>(i, i, inv_h2m[i]));
    }
    return ToSparseMatrix(dofs_, dofs_, nonzeros_new);
}




template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
