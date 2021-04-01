#ifndef FEM_BILLIARD_BALL_STATE_FORCE_H
#define FEM_BILLIARD_BALL_STATE_FORCE_H

#include "fem/state_force.h"
#include "common/common.h"

template<int vertex_dim>
class BilliardBallStateForce : public StateForce<vertex_dim> {
public:
    BilliardBallStateForce();

    void Initialize(const real radius, const int single_ball_vertex_num,
        const real stiffness, const real frictional_coeff);

    const real radius() const { return radius_; }
    const int single_ball_vertex_num() const { return single_ball_vertex_num_; }
    const real stiffness() const { return stiffness_; }

    const VectorXr ForwardForce(const VectorXr& q, const VectorXr& v) const override;
    void BackwardForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv) const override;

private:
    real radius_;
    int single_ball_vertex_num_;

    // Parameters for the impulse-based contact model (essentially a spring).
    // Step 1: Compute c.o.m. positions of each billiard ball from q.
    // Step 2: Use the stiffness to compute the spring force.
    // Step 3: Use the frictional_coeff to compute the friction force.
    // Step 4: Distribute the spring and frictional force equally to each vertex in q.
    real stiffness_;
    real frictional_coeff_;
};

#endif