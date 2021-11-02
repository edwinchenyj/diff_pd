#ifndef SOLVER_PARDISO_SPD_SOLVER_H
#define SOLVER_PARDISO_SPD_SOLVER_H

#include "common/config.h"

class PardisoSolver {
    public:
    PardisoSolver(): ia_(nullptr), ja_(nullptr), a_(nullptr) {}
    ~PardisoSolver();

    virtual void Compute(const SparseMatrix& lhs, const std::map<std::string, real>& options);
    virtual const VectorXr Solve(VectorXr& rhs);
    virtual const MatrixXr Solve(MatrixXr& rhs);
    virtual const VectorXr Solve(const VectorXr& rhs);
    virtual const MatrixXr Solve(const MatrixXr& rhs);

protected:
    int n_;
    int* ia_;
    int* ja_;
    double* a_;

    std::vector<int> m_outerArray, m_innerArray;
    std::vector<double> m_a;

    // Solver parameters.
    int mtype_; // Use -2 for real symmetric indefinte matrix, 2 for real SPD, and 1 for structurally symmetric.
    int solver_; // Use 1 for multi-recursive iterative solver.
    int msglvl_; // Output lever. 0 = no output. 1 = print statistical information.
    int maxfct_; // Maximum number of numerical factorizations.
    int mnum_; // Which factorization to use.
    // End of parameters.

    void* pt_[64];
    int iparm_[64];
    double dparm_[64];
};

class PardisoSpdSolver : PardisoSolver {
    public:
    PardisoSpdSolver(): PardisoSolver() {};
    ~PardisoSpdSolver() {};

    void Compute(const SparseMatrix& lhs, const std::map<std::string, real>& options) override;
    const VectorXr Solve(const VectorXr& rhs) override;
    const MatrixXr Solve(const MatrixXr& rhs) override;
    


};

#endif