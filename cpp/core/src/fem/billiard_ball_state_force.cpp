#include "fem/billiard_ball_state_force.h"
#include "common/common.h"
#include "common/geometry.h"
#include "mesh/mesh.h"

template<int vertex_dim>
BilliardBallStateForce<vertex_dim>::BilliardBallStateForce()
    : radius_(0), single_ball_vertex_num_(0), stiffness_(0) {}

template<int vertex_dim>
void BilliardBallStateForce<vertex_dim>::Initialize(const real radius, const int single_ball_vertex_num, const real stiffness) {
    radius_ = radius;
    single_ball_vertex_num_ = single_ball_vertex_num;
    stiffness_ = stiffness;
}

template<int vertex_dim>
const VectorXr BilliardBallStateForce<vertex_dim>::ForwardForce(const VectorXr& q, const VectorXr& v) const {
    // Reshape q to n x dim.
    const int vertex_num = static_cast<int>(q.size()) / vertex_dim;
    CheckError(vertex_num * vertex_dim == static_cast<int>(q.size()) && vertex_num % single_ball_vertex_num_ == 0,
        "Incompatible vertex number.");
    const int ball_num = vertex_num / single_ball_vertex_num_;
    std::vector<MatrixXr> vertices(ball_num, MatrixXr::Zero(vertex_dim, single_ball_vertex_num_));
    for (int i = 0; i < ball_num; ++i)
        for (int j = 0; j < single_ball_vertex_num_; ++j)
            for (int k = 0; k < vertex_dim; ++k) {
                vertices[i](k, j) = q(i * single_ball_vertex_num_ * vertex_dim + j * vertex_dim + k);
            }

    // Compute the center of each ball.
    MatrixXr centers = MatrixXr::Zero(vertex_dim, ball_num);
    for (int i = 0; i < ball_num; ++i) {
        centers.col(i) = vertices[i].rowwise().mean();
    }

    // Compute the distance between centers.
    VectorXr force = VectorXr::Zero(q.size());
    for (int i = 0; i < ball_num; ++i)
        for (int j = i + 1; j < ball_num; ++j) {
            const VectorXr ci = centers.col(i);
            const VectorXr cj = centers.col(j);
            const VectorXr dir_i2j = cj - ci;
            const real cij_dist = dir_i2j.norm();
            // Now compute the force.
            const real f_mag = std::max(2 * radius_ - cij_dist, 0.0) * stiffness_;
            const VectorXr i2j = dir_i2j / cij_dist;
            const VectorXr fj = i2j * f_mag;
            const VectorXr fi = -fj;
            // Now distribute fi and fj to ball i and j, respectively.
            for (int k = 0; k < single_ball_vertex_num_; ++k) {
                force.segment(i * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) += fi / single_ball_vertex_num_;
                force.segment(j * single_ball_vertex_num_ * vertex_dim + k * vertex_dim, vertex_dim) += fj / single_ball_vertex_num_;
            }
        }
    return force;
}

template<int vertex_dim>
void BilliardBallStateForce<vertex_dim>::BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const {
    // TODO.
    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
}

template class BilliardBallStateForce<2>;
template class BilliardBallStateForce<3>;