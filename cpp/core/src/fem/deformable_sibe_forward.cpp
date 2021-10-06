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

        CheckError(options.find("max_newton_iter") != options.end(), "Missing option max_newton_iter.");
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
        CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
        CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
        CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
        const int max_newton_iter = static_cast<int>(options.at("max_newton_iter"));
        const int max_ls_iter = static_cast<int>(options.at("max_ls_iter"));
        const real abs_tol = options.at("abs_tol");
        const real rel_tol = options.at("rel_tol");
        const int verbose_level = static_cast<int>(options.at("verbose"));
        const int thread_ct = static_cast<int>(options.at("thread_ct"));
        CheckError(max_newton_iter > 0, "Invalid max_newton_iter: " + std::to_string(max_newton_iter));
        CheckError(max_ls_iter > 0, "Invalid max_ls_iter: " + std::to_string(max_ls_iter));

        omp_set_num_threads(thread_ct);

        const real h = dt;
        // TODO: this mass is wrong for tri or tet meshes.
        const real mass = element_volume_ * density_;
        const real h2m = dt * dt / mass;
        const VectorXr force = ForwardStateForce(q, v);
        const real inv_h2m = mass / (h * h);
        const int max_contact_iter = 5;
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

        }


    }


template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;
