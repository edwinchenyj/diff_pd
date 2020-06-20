#include "fem/gravitational_state_force.h"

template<int vertex_dim>
GravitationalStateForce<vertex_dim>::GravitationalStateForce()
    : StateForce<vertex_dim>(), mass_(0), g_(Eigen::Matrix<real, vertex_dim, 1>::Zero()) {}

template<int vertex_dim>
void GravitationalStateForce<vertex_dim>::Initialize(const real mass, const Eigen::Matrix<real, vertex_dim, 1>& g) {
    mass_ = mass;
    g_ = g;
}

template<int vertex_dim>
const VectorXr GravitationalStateForce<vertex_dim>::Force(const VectorXr& q, const VectorXr& v) {
    const int dofs = static_cast<int>(q.size());
    CheckError(dofs % vertex_dim == 0, "Incompatible dofs and vertex_dim.");
    const int vertex_num = dofs / vertex_dim;
    VectorXr f = VectorXr::Zero(dofs);
    for (int i = 0; i < vertex_num; ++i)
        f.segment(vertex_dim * i, vertex_dim) = mass_ * g_;
    return f;
}

template<int vertex_dim>
void GravitationalStateForce<vertex_dim>::ForceDifferential(const VectorXr& q, const VectorXr& v, const VectorXr& f,
    const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) {
    dl_dq = VectorXr::Zero(q.size());
    dl_dv = VectorXr::Zero(v.size());
}

template class GravitationalStateForce<2>;
template class GravitationalStateForce<3>;