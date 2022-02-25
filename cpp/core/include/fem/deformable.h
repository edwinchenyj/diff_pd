#ifndef FEM_DEFORMABLE_H
#define FEM_DEFORMABLE_H

#include "common/config.h"
#include "mesh/mesh.h"
#include "material/material.h"
#include "state_force/state_force.h"
#include "fem/finite_element_sample.h"
#include "pd_energy/pd_vertex_energy.h"
#include "pd_energy/pd_element_energy.h"
#include "pd_energy/pd_muscle_energy.h"
#include "pd_energy/deformation_gradient_auxiliary_data.h"
#include "friction/frictional_boundary.h"
#include "solver/pardiso_spd_solver.h"
#include "Eigen/SparseCholesky"
#include "Spectra/GenEigsSolver.h"
#include "Spectra/MatOp/SparseGenMatProd.h"
#include "Spectra/GenEigsRealShiftSolver.h"
#include "Spectra/SymEigsShiftSolver.h"
#include "Spectra/MatOp/SparseSymShiftSolve.h"
#include "Spectra/MatOp/SparseGenRealShiftSolve.h"
#include "solver/SparseGenRealShiftSolvePardiso.h"

template<int vertex_dim, int element_dim>
class Deformable {
public:
    Deformable();
    virtual ~Deformable() {}

    // Initialize with the undeformed shape.
    void Initialize(const std::string& binary_file_name, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);
    void Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
        const Eigen::Matrix<int, element_dim, -1>& faces, const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);

    const real density() const { return density_; }
    // TODO: for now we assume all elements have equal volume. This is not true for tet or triangle meshes.
    const real element_volume() const { return element_volume_; }
    const int dofs() const { return dofs_; }
    const int act_dofs() const { return act_dofs_; }
    const Mesh<vertex_dim, element_dim>& mesh() const { return mesh_; }
    const std::map<int, real>& dirichlet() const { return dirichlet_; }
    const std::vector<std::shared_ptr<StateForce<vertex_dim>>>& state_forces() const { return state_forces_; }
    const std::vector<std::pair<std::shared_ptr<PdVertexEnergy<vertex_dim>>, std::set<int>>>& pd_vertex_energies() const {
        return pd_vertex_energies_;
    }
    const std::vector<std::shared_ptr<PdElementEnergy<vertex_dim>>>& pd_element_energies() const {
        return pd_element_energies_;
    }
    const std::vector<std::pair<std::shared_ptr<PdMuscleEnergy<vertex_dim>>, std::vector<int>>>& pd_muscle_energies() const {
        return pd_muscle_energies_;
    }
    const int NumOfPdElementEnergies() const {
        return static_cast<int>(pd_element_energies_.size());
    }
    const int NumOfPdMuscleEnergies() const {
        return static_cast<int>(pd_muscle_energies_.size());
    }
    const std::map<int, int>& frictional_boundary_vertex_indices() const {
        return frictional_boundary_vertex_indices_;
    }

    void SetDirichletBoundaryCondition(const int dof, const real val) {
        if (dirichlet_.find(dof) == dirichlet_.end()) pd_solver_ready_ = false;
        dirichlet_[dof] = val;
    }
    void RemoveDirichletBoundaryCondition(const int dof) {
        if (dirichlet_.find(dof) != dirichlet_.end()) {
            dirichlet_.erase(dof);
            pd_solver_ready_ = false;
        }
    }

    // Elastic force from the material model.
    const real ElasticEnergy(const VectorXr& q) const;
    const VectorXr ElasticForce(const VectorXr& q) const;
    const VectorXr ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const;
    const SparseMatrixElements ElasticForceDifferential(const VectorXr& q) const;

    // Add state-based forces.
    void AddStateForce(const std::string& force_type, const std::vector<real>& params);
    const VectorXr ForwardStateForce(const VectorXr& q, const VectorXr& v) const;
    void BackwardStateForce(const VectorXr& q, const VectorXr& v, const VectorXr& f,
        const VectorXr& dl_df, VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_dp) const;
    const int NumOfStateForceParameters() const;

    // Add PD energies.
    void AddPdEnergy(const std::string& energy_type, const std::vector<real>& params, const std::vector<int>& indices);
    const real ComputePdEnergy(const VectorXr& q, const bool use_precomputed_data) const;
    const VectorXr PdEnergyForce(const VectorXr& q, const bool use_precomputed_data) const;
    const VectorXr PdEnergyForceDifferential(const VectorXr& q, const VectorXr& dq, const VectorXr& dw) const;
    void PdEnergyForceDifferential(const VectorXr& q, const bool require_dq, const bool require_dw,
        const bool use_precomputed_data, SparseMatrixElements& dq, SparseMatrixElements& dw) const;

    // Actuation.
    void AddActuation(const real stiffness, const std::array<real, vertex_dim>& fiber_direction,
        const std::vector<int>& indices);
    const real ActuationEnergy(const VectorXr& q, const VectorXr& a) const;
    const VectorXr ActuationForce(const VectorXr& q, const VectorXr& a) const;
    const VectorXr ActuationForceDifferential(const VectorXr& q, const VectorXr& a,
        const VectorXr& dq, const VectorXr& da, const VectorXr& dw) const;
    void ActuationForceDifferential(const VectorXr& q, const VectorXr& a,
        SparseMatrixElements& dq, SparseMatrixElements& da, SparseMatrixElements& dw) const;

    // Frictional boundary.
    void SetFrictionalBoundary(const std::string& boundary_type, const std::vector<real>& params, const std::vector<int> indices);

    void Forward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
        const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    // mat_w: the stiffness used by PdEnergy.
    // act_w: the stiffness used by ActuationEnergy.
    void Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
        const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
        const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next,
        const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
        VectorXr& dl_dpd_mat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const;
    void GetQuasiStaticState(const std::string& method, const VectorXr& a, const VectorXr& f_ext,
        const std::map<std::string, real>& options, VectorXr& q) const;
    void SaveToMeshFile(const VectorXr& q, const std::string& file_name) const;

    // For Python binding.
    void PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& a,
        const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
        std::vector<real>& q_next, std::vector<real>& v_next, std::vector<int>& active_contact_idx) const;
    void PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& a,
        const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next, const std::vector<real>& v_next,
        const std::vector<int>& active_contact_idx,
        const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
        const std::map<std::string, real>& options,
        std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_da, std::vector<real>& dl_df_ext,
        std::vector<real>& dl_dmat_w, std::vector<real>& dl_dact_w, std::vector<real>& dl_dstate_p) const;
    void PyGetQuasiStaticState(const std::string& method, const std::vector<real>& a, const std::vector<real>& f_ext,
        const std::map<std::string, real>& options, std::vector<real>& q) const;
    void PySaveToMeshFile(const std::vector<real>& q, const std::string& file_name) const;

    const real PyElasticEnergy(const std::vector<real>& q) const;
    const std::vector<real> PyElasticForce(const std::vector<real>& q) const;
    const std::vector<real> PyElasticForceDifferential(const std::vector<real>& q, const std::vector<real>& dq) const;
    const std::vector<std::vector<real>> PyElasticForceDifferential(const std::vector<real>& q) const;

    const std::vector<real> PyForwardStateForce(const std::vector<real>& q, const std::vector<real>& v) const;
    void PyBackwardStateForce(const std::vector<real>& q, const std::vector<real>& v, const std::vector<real>& f,
        const std::vector<real>& dl_df, std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_dp) const;

    const real PyComputePdEnergy(const std::vector<real>& q) const;
    const std::vector<real> PyPdEnergyForce(const std::vector<real>& q) const;
    const std::vector<real> PyPdEnergyForceDifferential(const std::vector<real>& q, const std::vector<real>& dq,
        const std::vector<real>& dw) const;
    void PyPdEnergyForceDifferential(const std::vector<real>& q, const bool require_dq, const bool require_dw,
        std::vector<std::vector<real>>& dq, std::vector<std::vector<real>>& dw) const;

    const real PyActuationEnergy(const std::vector<real>& q, const std::vector<real>& a) const;
    const std::vector<real> PyActuationForce(const std::vector<real>& q, const std::vector<real>& a) const;
    const std::vector<real> PyActuationForceDifferential(const std::vector<real>& q, const std::vector<real>& a,
        const std::vector<real>& dq, const std::vector<real>& da, const std::vector<real>& dw) const;
    void PyActuationForceDifferential(const std::vector<real>& q, const std::vector<real>& a, std::vector<std::vector<real>>& dq,
        std::vector<std::vector<real>>& da, std::vector<std::vector<real>>& dw) const;

    // Unfortunately, the preconditioner would prefer seeing this PdLhsSolve function.
    // If you anticipate the number of contact DoFs to be small, set use_sparse = false. Otherwise, set it to true.
    // By default we set use_sparse = false.
    const VectorXr PdLhsSolve(const std::string& method, const VectorXr& rhs,
        const std::map<int, real>& additional_dirichlet_boundary_condition,
        const bool use_acc, const bool use_sparse) const;
    
    mutable VectorXr gravitational_force;
    mutable VectorXr pd_energy_force;
    mutable VectorXr actuation_force;
    mutable int verbose_level;
    
    
    mutable MatrixXr U1;
    mutable MatrixXr V1;
    mutable MatrixXr U2;
    mutable MatrixXr V2;
    
    mutable std::pair<MatrixXr, VectorXr > m_Us;
    mutable std::pair<MatrixXr, VectorXr > m_Us2;
    
    mutable MatrixXr dt_J_G_reduced;
    
    mutable VectorXr vG;
    mutable VectorXr vH;
    
    mutable VectorXr fG;
    
    mutable VectorXr fH;
    
    mutable SparseMatrix A;
    
    mutable SparseMatrix Identity;
    
    mutable std::vector<int> J21_J22_outer_ind_ptr;
    mutable std::vector<int> J21_outer_ind_ptr;
    mutable std::vector<int> J22_outer_ind_ptr;
    mutable std::vector<int> J22i_outer_ind_ptr;
    mutable std::vector<int> J21_inner_ind;
    mutable std::vector<int> J22_inner_ind;
    mutable std::vector<int> J22i_inner_ind;
    mutable std::vector<double> J22i_identity_val;
    mutable std::vector<double> stiffness0_val;
    mutable std::vector<int> stiffness0_outer_ind_ptr;
    mutable std::vector<int> stiffness0_inner_ind;
    mutable real ritz_error_norm = 2.0;

    mutable VectorXr q_prev;
    mutable VectorXr v_prev;
    mutable VectorXr q_current;
    mutable VectorXr v_current;
    mutable bool two_step_method = false;

    mutable SparseMatrix lumped_mass;
    mutable SparseMatrix lumped_mass_inv;
    mutable SparseMatrix stiffness;
    mutable SparseMatrix MinvK;
    mutable SparseMatrix J12, J21;
    void phi(MatrixXr &A, MatrixXr &output) const;
protected:
    void ForwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardSIBE(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardBDFFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardBDF2FULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardSBDF2FULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardTRBDF2FULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardSTRBDF2FULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardTHETATRBDF2FULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardTHETASTRBDF2FULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;

    void ForwardBDF2EREFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardSBDF2EREFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardTRBDF2EREFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardSTRBDF2EREFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardTHETATRBDF2EREFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardTHETASTRBDF2EREFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;





    void ForwardSIBEFULL(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardSIERE(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;
    void ForwardProjectiveDynamics(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
        const VectorXr& f_ext, const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
        std::vector<int>& active_contact_idx) const;

    void BackwardSemiImplicit(const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext, const real dt,
        const VectorXr& q_next, const VectorXr& v_next, const std::vector<int>& active_contact_idx,
        const VectorXr& dl_dq_next, const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
        VectorXr& dl_dmat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const;
    void BackwardNewton(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a, const VectorXr& f_ext,
        const real dt, const VectorXr& q_next, const VectorXr& v_next, const std::vector<int>& active_contact_idx,
        const VectorXr& dl_dq_next, const VectorXr& dl_dv_next,
        const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
        VectorXr& dl_dmat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const;
    void BackwardProjectiveDynamics(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
        const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next, const std::vector<int>& active_contact_idx,
        const VectorXr& dl_dq_next, const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
        VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
        VectorXr& dl_dmat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const;

    void QuasiStaticStateNewton(const std::string& method, const VectorXr& a, const VectorXr& f_ext,
        const std::map<std::string, real>& options, VectorXr& q) const;
    const VectorXr GetUndeformedShape() const;

    // Check if elements are flipped.
    const bool HasFlippedElement(const VectorXr& q) const;
    const Eigen::Matrix<real, vertex_dim, vertex_dim> DeformationGradient(const int element_idx,
        const Eigen::Matrix<real, vertex_dim, element_dim>& q, const int sample_idx) const;

    // Derived classes should implement the following function and data members.
    virtual void InitializeFiniteElementSamples() {
        CheckError(false, "This function must be implemented in a derived class.");
    }
    virtual const int GetNumOfSamplesInElement() const {    
        CheckError(false, "This function must be implemented in a derived class.");
        return 0;
    }
    // finite_element_samples_[element_idx][sample_idx].
    // See the header file for a detailed explanation.
    std::vector<std::vector<FiniteElementSample<vertex_dim, element_dim>>> finite_element_samples_;
    // End of methods and data members that need implementations from derived classes.
    const SparseMatrix LumpedMassMatrix(const std::map<int, real>& dirichlet_with_friction) const;
    const SparseMatrix LumpedMassMatrixInverse(const std::map<int, real>& dirichlet_with_friction) const;
    std::vector<real> lumped_mass_;



private:
    void AssignToGlobalDeformable() const;
    void ClearGlobalDeformable() const;

    void InitializeAfterMesh(const real density,
        const std::string& material_type, const real youngs_modulus, const real poissons_ratio);
    const std::shared_ptr<Material<vertex_dim>> InitializeMaterial(const std::string& material_type,
        const real youngs_modulus, const real poissons_ratio) const;
    const VectorXr NewtonMatrixOp(const VectorXr& q_sol, const VectorXr& a, const real inv_h2m,
        const std::map<int, real>& dirichlet_with_friction, const VectorXr& dq) const;
    const SparseMatrix NewtonMatrix(const VectorXr& q_sol, const VectorXr& a, const real inv_h2m,
        const std::map<int, real>& dirichlet_with_friction, const bool use_precomputed_data) const;
    const SparseMatrix NewtonMatrix(const VectorXr& q_sol, const VectorXr& a, const std::vector<real> inv_h2m,
        const std::map<int, real>& dirichlet_with_friction, const bool use_precomputed_data) const;
    const SparseMatrix StiffnessMatrix(const VectorXr& q_sol, const VectorXr& a,
        const std::map<int, real>& dirichlet_with_friction, const bool use_precomputed_data) const;
    const VectorXr QuasiStaticMatrixOp(const VectorXr& q, const VectorXr& a, const VectorXr& dq) const;
    const SparseMatrix QuasiStaticMatrix(const VectorXr& q, const VectorXr& a) const;

    void SetupProjectiveDynamicsSolver(const std::string& method, const real dt, const std::map<std::string, real>& options) const;
    const VectorXr ProjectiveDynamicsLocalStep(const VectorXr& q_cur, const VectorXr& a_cur,
        const std::map<int, real>& dirichlet_with_friction) const;

    void SetupProjectiveDynamicsLocalStepDifferential(const VectorXr& q_cur, const VectorXr& a_cur,
        std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
        std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices
    ) const;
    const VectorXr ApplyProjectiveDynamicsLocalStepDifferential(const VectorXr& q_cur, const VectorXr& a_cur,
        const std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>& pd_backward_local_element_matrices,
        const std::vector<std::vector<Eigen::Matrix<real, vertex_dim * element_dim, vertex_dim * element_dim>>>& pd_backward_local_muscle_matrices,
        const VectorXr& dq_cur) const;
    const VectorXr PdLhsMatrixOp(const VectorXr& q, const std::map<int, real>& additional_dirichlet_boundary_condition) const;

    // Compute deformation gradient.
    const Eigen::Matrix<real, vertex_dim, element_dim> ScatterToElement(const VectorXr& q, const int element_idx) const;
    const Eigen::Matrix<real, vertex_dim * element_dim, 1> ScatterToElementFlattened(const VectorXr& q, const int element_idx) const;
    void ComputeDeformationGradientAuxiliaryDataAndProjection(const VectorXr& q) const;

    // A dedicate function of using PD to solve the nonlinear systems of equations.
    // - q_next - h2m * (f_pd(q_next) + f_act(q_next, a)) = rhs.
    // - q_init is the intiial guess.
    // - Return: q_next.
    const VectorXr PdNonlinearSolve(const std::string& method, const VectorXr& q_init, const VectorXr& a,
        const real inv_h2m, const VectorXr& rhs, const std::map<int, real>& additional_dirichlet,
        const std::map<std::string, real>& options) const;

    // Undeformed shape.
    Mesh<vertex_dim, element_dim> mesh_;
    // Element information.
    real density_;
    real element_volume_;
    // Material model that defines the elastic energy --- not used by PD.
    std::shared_ptr<Material<vertex_dim>> material_;
    // Number of variables. Typically vertex_dim * vertex number.
    int dofs_;

    // Boundary conditions.
    std::map<int, real> dirichlet_;

    // Used for stepper
    mutable int thread_ct;
    mutable bool si_method;
    mutable real h;
    mutable bool recompute_eigen_decomp_each_step; 
    mutable int num_modes; 
    mutable VectorXr g;
    mutable bool use_precomputed_data = false;
    mutable int max_contact_iter = 5;
    mutable real theta_parameter;
    void GetG() const;
    const void ContactDirichlet(const VectorXr& q, const std::vector<int>& contact_idx, std::map<int, real>& augmented_dirichlet) const;
    void InitializeStepperOptions(const std::map<std::string, real>& options) const;
    void ApplyDirichlet(const std::map<int, real>& dirichlet_with_friction, VectorXr& q, VectorXr& v) const;
    void ApplyDirichlet(const std::map<int, real>& dirichlet_with_friction, VectorXr& vector) const;
    void SetupMatrices(const VectorXr& q, const VectorXr& a, const std::map<int, real>& dirichlet_with_friction,
        const bool use_precomputed_data) const;
    void SimpleForce(const VectorXr& q, const VectorXr& a, const std::map<int, real>& dirichlet_with_friction,
        const bool use_precomputed_data, const VectorXr& g, VectorXr& force) const;
    void MassPCA(const SparseMatrix lumped_mass, const SparseMatrix MinvK, const int pca_dim, const std::vector<int>& active_contact_idx) const;
    void SetupJacobian(std::vector<int>& active_contact_idx) const;
    void ComputePCAProjection(const std::vector<int>& active_contact_idx) const;
    void SplitVelocityState(const VectorXr& v) const;
    void SplitForceState(const VectorXr& f) const;
    void ComputeReducedRhs(VectorXr& reduced_rhs, const VectorXr& v_sol, const VectorXr& force_sol, const real h) const;
    void SubspaceEREUpdate(VectorXr& state, const PardisoSolver& solver, const real h) const;
    
    // Projective-dynamics-related data members.
    mutable std::array<PardisoSpdSolver, vertex_dim> pd_pardiso_solver_;
    mutable std::array<Eigen::SimplicialLDLT<SparseMatrix>, vertex_dim> pd_eigen_solver_;

    mutable std::array<SparseMatrix, vertex_dim> pd_lhs_;
    mutable bool pd_solver_ready_;

    // State-based forces.
    std::vector<std::shared_ptr<StateForce<vertex_dim>>> state_forces_;

    // Projective-dynamics energies.
    std::vector<std::pair<std::shared_ptr<PdVertexEnergy<vertex_dim>>, std::set<int>>> pd_vertex_energies_;
    std::vector<std::shared_ptr<PdElementEnergy<vertex_dim>>> pd_element_energies_;

    // Actuation forces.
    std::vector<std::pair<std::shared_ptr<PdMuscleEnergy<vertex_dim>>, std::vector<int>>> pd_muscle_energies_;
    int act_dofs_;

    // Friction.
    std::shared_ptr<FrictionalBoundary<vertex_dim>> frictional_boundary_;
    std::map<int, int> frictional_boundary_vertex_indices_;
    // See the paper for the definition of the data members below.
    mutable std::array<MatrixXr, vertex_dim> Acc_;
    mutable std::array<MatrixXr, vertex_dim> AinvIc_;

    // Cache auxiliary data.
    // F_projections_[element_idx][sample_idx].
    mutable std::vector<std::vector<DeformationGradientAuxiliaryData<vertex_dim>>> F_auxiliary_;
    // projections_[energy_cnt][element_idx][sample_idx].
    mutable std::vector<std::vector<std::vector<Eigen::Matrix<real, vertex_dim, vertex_dim>>>> projections_;
};

extern const void* global_deformable;
extern int global_vertex_dim;
extern int global_element_dim;
extern std::map<int, real> global_additional_dirichlet_boundary;
extern std::string global_pd_backward_method;

#endif