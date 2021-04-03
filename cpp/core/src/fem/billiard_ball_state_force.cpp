#include "fem/billiard_ball_state_force.h"
#include "common/common.h"
#include "common/geometry.h"
#include "mesh/mesh.h"

template<int vertex_dim>
void BilliardBallStateForce<vertex_dim>::Initialize(const real radius, const int single_ball_vertex_num,
    const real stiffness, const real frictional_coeff) {
    Vector2r parameters(stiffness, frictional_coeff);
    StateForce<vertex_dim>::set_parameters(parameters);
    radius_ = radius;
    single_ball_vertex_num_ = single_ball_vertex_num;
}

template<int vertex_dim>
const VectorXr BilliardBallStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    // Reshape q to n x dim.
    const int vertex_num = static_cast<int>(q.size()) / vertex_dim;
    CheckError(vertex_num * vertex_dim == static_cast<int>(q.size()) && vertex_num % single_ball_vertex_num_ == 0,
        "Incompatible vertex number.");
    const int ball_num = vertex_num / single_ball_vertex_num_;
    std::vector<MatrixXr> vertices(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    std::vector<MatrixXr> vertex_velocities(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    for (int i = 0; i < ball_num; ++i)
        for (int j = 0; j < single_ball_vertex_num_; ++j)
            for (int k = 0; k < vertex_dim; ++k) {
                vertices[i](k, j) = q(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
                vertex_velocities[i](k, j) = v(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
            }

    // Compute the center of each ball.
    MatrixXr centers = MatrixXr::Zero(vertex_dim, ball_num);
    MatrixXr central_velocities = MatrixXr::Zero(vertex_dim, ball_num);
    for (int i = 0; i < ball_num; ++i) {
        centers.col(i) = vertices[i].rowwise().mean();
        central_velocities.col(i) = vertex_velocities[i].rowwise().mean();
    }

    // Compute the distance between centers.
    VectorXr force = VectorXr::Zero(q.size());
    for (int i = 0; i < ball_num; ++i)
        for (int j = i + 1; j < ball_num; ++j) {
            const VectorXr ci = centers.col(i);
            const VectorXr cj = centers.col(j);
            const VectorXr dir_i2j = cj - ci;
            const real cij_dist = dir_i2j.norm();
            // Now compute the spring force.
            const real f_mag = std::max(2 * radius_ - cij_dist, 0.0) * stiffness();
            const VectorXr i2j = dir_i2j / cij_dist;
            const VectorXr fj = i2j * f_mag;
            const VectorXr fi = -fj;
            // Next, compute the friction force.
            const real ff_mag = f_mag * frictional_coeff();
            const VectorXr vi = central_velocities.col(i);
            const VectorXr vj = central_velocities.col(j);
            const VectorXr vi_in_j = vi - vj;
            // Test the friction direction.
            Eigen::Matrix<real, vertex_dim, 1> ffi_dir = Eigen::Matrix<real, vertex_dim, 1>::Zero();
            if (vi_in_j.x() * i2j.y() - vi_in_j.y() * i2j.x() > 0) {
                // Rotate i2j by 90 degrees to obtain ff_i.
                ffi_dir.x() = -i2j.y();
                ffi_dir.y() = i2j.x();
            } else {
                // Rotate i2j by -90 degrees to obtain ff_i.
                ffi_dir.x() = i2j.y();
                ffi_dir.y() = -i2j.x();
            }
            const VectorXr ffi = ffi_dir * ff_mag;
            const VectorXr ffj = -ffi;
            // Now distribute fi and fj to ball i and j, respectively.
            for (int k = 0; k < single_ball_vertex_num_; ++k) {
                force.segment(i * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) +=
                    (fi + ffi) / single_ball_vertex_num_;
                force.segment(j * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) +=
                    (fj + ffj) / single_ball_vertex_num_;
            }
        }
    return force;
}

template<int vertex_dim>
void BilliardBallStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const {
    dl_dp = VectorXr::Zero(2);

    // Reshape q to n x dim.
    const int vertex_num = static_cast<int>(q.size()) / vertex_dim;
    CheckError(vertex_num * vertex_dim == static_cast<int>(q.size()) && vertex_num % single_ball_vertex_num_ == 0,
        "Incompatible vertex number.");
    const int ball_num = vertex_num / single_ball_vertex_num_;
    std::vector<MatrixXr> vertices(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    std::vector<MatrixXr> vertex_velocities(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    for (int i = 0; i < ball_num; ++i)
        for (int j = 0; j < single_ball_vertex_num_; ++j)
            for (int k = 0; k < vertex_dim; ++k) {
                vertices[i](k, j) = q(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
                vertex_velocities[i](k, j) = v(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
            }

    // Compute the center of each ball.
    MatrixXr centers = MatrixXr::Zero(vertex_dim, ball_num);
    MatrixXr central_velocities = MatrixXr::Zero(vertex_dim, ball_num);
    for (int i = 0; i < ball_num; ++i) {
        centers.col(i) = vertices[i].rowwise().mean();
        central_velocities.col(i) = vertex_velocities[i].rowwise().mean();
    }

    // Work on dl_dq.
    dl_dq = VectorXr::Zero(q.size());
    for (int i = 0; i < ball_num; ++i)
        for (int j = i + 1; j < ball_num; ++j) {
            const VectorXr ci = centers.col(i);
            const VectorXr cj = centers.col(j);
            const VectorXr dir_i2j = cj - ci;
            const MatrixXr jac_dir_i2j_ci = -Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity();
            const MatrixXr jac_dir_i2j_cj = Eigen::Matrix<real, vertex_dim, vertex_dim>::Identity();
            const real cij_dist = dir_i2j.norm();
            CheckError(cij_dist > 1.5 * radius_, "Balls are too close to each other");
            const VectorXr d_cij_dist_ci = jac_dir_i2j_ci.transpose() * dir_i2j / cij_dist;
            const VectorXr d_cij_dist_cj = jac_dir_i2j_cj.transpose() * dir_i2j / cij_dist;
            // Now compute the force.
            const real f_mag = std::max(2 * radius_ - cij_dist, 0.0) * stiffness();
            const Vector2r d_f_mag_dp = Vector2r(std::max(2 * radius_ - cij_dist, 0.0), 0);
            VectorXr d_f_mag_ci = VectorXr::Zero(vertex_dim);
            VectorXr d_f_mag_cj = VectorXr::Zero(vertex_dim);
            if (f_mag > 0) {
                d_f_mag_ci = -stiffness() * d_cij_dist_ci;
                d_f_mag_cj = -stiffness() * d_cij_dist_cj;
            }
            const VectorXr i2j = dir_i2j / cij_dist;
            const MatrixXr jac_i2j_ci = jac_dir_i2j_ci / cij_dist - dir_i2j * d_cij_dist_ci.transpose() / (cij_dist * cij_dist);
            const MatrixXr jac_i2j_cj = jac_dir_i2j_cj / cij_dist - dir_i2j * d_cij_dist_cj.transpose() / (cij_dist * cij_dist);
            // const VectorXr fj = i2j * f_mag;
            const MatrixXr jac_fj_p = i2j * d_f_mag_dp.transpose();
            const MatrixXr jac_fj_ci = jac_i2j_ci * f_mag + i2j * d_f_mag_ci.transpose();
            const MatrixXr jac_fj_cj = jac_i2j_cj * f_mag + i2j * d_f_mag_cj.transpose();
            // const VectorXr fi = -fj;
            const MatrixXr jac_fi_p = -jac_fj_p;
            const MatrixXr jac_fi_ci = -jac_fj_ci;
            const MatrixXr jac_fi_cj = -jac_fj_cj;
            // Next, compute the friction force.
            const real ff_mag = f_mag * frictional_coeff();
            const Vector2r d_ff_mag_dp(std::max(2 * radius_ - cij_dist, 0.0) * frictional_coeff(), f_mag);
            const VectorXr d_ff_mag_ci = d_f_mag_ci * frictional_coeff();
            const VectorXr d_ff_mag_cj = d_f_mag_cj * frictional_coeff();
            const VectorXr vi = central_velocities.col(i);
            const VectorXr vj = central_velocities.col(j);
            const VectorXr vi_in_j = vi - vj;
            // Test the friction direction.
            Eigen::Matrix<real, vertex_dim, 1> ffi_dir = Eigen::Matrix<real, vertex_dim, 1>::Zero();
            MatrixXr jac_ffi_dir_ci = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            MatrixXr jac_ffi_dir_cj = Eigen::Matrix<real, vertex_dim, vertex_dim>::Zero();
            if (vi_in_j.x() * i2j.y() - vi_in_j.y() * i2j.x() > 0) {
                // Rotate i2j by 90 degrees to obtain ff_i.
                ffi_dir.x() = -i2j.y();
                ffi_dir.y() = i2j.x();
                jac_ffi_dir_ci.row(0) = -jac_i2j_ci.row(1);
                jac_ffi_dir_ci.row(1) = jac_i2j_ci.row(0);
                jac_ffi_dir_cj.row(0) = -jac_i2j_cj.row(1);
                jac_ffi_dir_cj.row(1) = jac_i2j_cj.row(0);
            } else {
                // Rotate i2j by -90 degrees to obtain ff_i.
                ffi_dir.x() = i2j.y();
                ffi_dir.y() = -i2j.x();
                jac_ffi_dir_ci.row(0) = jac_i2j_ci.row(1);
                jac_ffi_dir_ci.row(1) = -jac_i2j_ci.row(0);
                jac_ffi_dir_cj.row(0) = jac_i2j_cj.row(1);
                jac_ffi_dir_cj.row(1) = -jac_i2j_cj.row(0);
            }
            // const VectorXr ff_i = ff_i_dir * ff_mag;
            // const VectorXr ff_j = -ff_i;
            const MatrixXr jac_ffi_p = ffi_dir * d_ff_mag_dp.transpose();
            const MatrixXr jac_ffj_p = -jac_ffi_p;
            const MatrixXr jac_ffi_ci = jac_ffi_dir_ci * ff_mag + ffi_dir * d_ff_mag_ci.transpose();
            const MatrixXr jac_ffi_cj = jac_ffi_dir_cj * ff_mag + ffi_dir * d_ff_mag_cj.transpose();
            const MatrixXr jac_ffj_ci = -jac_ffi_ci;
            const MatrixXr jac_ffj_cj = -jac_ffi_cj;
            // Now distribute fi and fj to ball i and j, respectively.
            VectorXr dl_ci = VectorXr::Zero(vertex_dim);
            VectorXr dl_cj = VectorXr::Zero(vertex_dim);
            for (int k = 0; k < single_ball_vertex_num_; ++k) {
                // force.segment(i * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) += fi / single_ball_vertex_num_;
                // force.segment(j * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) += fj / single_ball_vertex_num_;
                const VectorXr dl_dfi = dl_df.segment(i * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim)
                    / single_ball_vertex_num_;
                const VectorXr dl_dfj = dl_df.segment(j * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim)
                    / single_ball_vertex_num_;
                dl_ci += (jac_fi_ci + jac_ffi_ci).transpose() * dl_dfi + (jac_fj_ci + jac_ffj_ci).transpose() * dl_dfj;
                dl_cj += (jac_fi_cj + jac_ffi_cj).transpose() * dl_dfi + (jac_fj_cj + jac_ffj_cj).transpose() * dl_dfj;
                dl_dp += (jac_fi_p + jac_ffi_p).transpose() * dl_dfi + (jac_fj_p + jac_ffj_p).transpose() * dl_dfj;
            }
            // Backpropagate from centers[i] and centers[j] to q.
            for (int p = 0; p < single_ball_vertex_num_; ++p)
                for (int k = 0; k < vertex_dim; ++k) {
                    dl_dq(i * single_ball_vertex_num_ * vertex_dim + p * vertex_dim + k) += dl_ci(k) / single_ball_vertex_num_;
                    dl_dq(j * single_ball_vertex_num_ * vertex_dim + p * vertex_dim + k) += dl_cj(k) / single_ball_vertex_num_;
                }
        }

    // Work on dl_dv.
    dl_dv = VectorXr::Zero(v.size());
}

template class BilliardBallStateForce<2>;
template class BilliardBallStateForce<3>;