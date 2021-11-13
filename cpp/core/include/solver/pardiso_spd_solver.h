#ifndef SOLVER_PARDISO_SPD_SOLVER_H
#define SOLVER_PARDISO_SPD_SOLVER_H

#include "common/config.h"

class PardisoSolver {
    public:
    PardisoSolver(): ia_(nullptr), ja_(nullptr), a_(nullptr) {}
    ~PardisoSolver();

    virtual void Compute(const SparseMatrix& lhs, const std::map<std::string, real>& options);
    virtual const VectorXr Solve(VectorXr& rhs) const;
    virtual const MatrixXr Solve(MatrixXr& rhs) const;
    virtual const VectorXr Solve(const VectorXr& rhs) const;

protected:
    mutable int n_;
    mutable int* ia_;
    mutable int* ja_;
    mutable double* a_;

    mutable std::vector<int> m_outerArray, m_innerArray;
    mutable std::vector<double> m_a;

    // Solver parameters.
    mutable int mtype_; // Use -2 for real symmetric indefinte matrix, 2 for real SPD, and 1 for structurally symmetric.
    mutable int solver_; // Use 1 for multi-recursive iterative solver.
    mutable int msglvl_; // Output lever. 0 = no output. 1 = print statistical information.
    mutable int maxfct_; // Maximum number of numerical factorizations.
    mutable int mnum_; // Which factorization to use.
    // End of parameters.

    mutable void* pt_[64];
    mutable int iparm_[64];
    mutable double dparm_[64];
};

class PardisoSpdSolver : PardisoSolver {
    public:
    PardisoSpdSolver(): PardisoSolver() {};
    ~PardisoSpdSolver() {};

    void Compute(const SparseMatrix& lhs, const std::map<std::string, real>& options) override;
    const VectorXr Solve(const VectorXr& rhs) const override;
    


};

#endif