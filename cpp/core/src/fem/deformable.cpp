#include "fem/deformable.h"
#include "common/common.h"
#include "common/geometry.h"
#include "solver/matrix_op.h"
#include "material/linear.h"
#include "material/corotated.h"
#include "material/neohookean.h"
#include "Eigen/SparseCholesky"

template<int vertex_dim, int element_dim>
Deformable<vertex_dim, element_dim>::Deformable()
    : mesh_(), density_(0), element_volume_(0), material_(nullptr), dofs_(0), pd_solver_ready_(false), act_dofs_(0),
    frictional_boundary_(nullptr) {}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const std::string& binary_file_name, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(binary_file_name);
    InitializeAfterMesh(density, material_type, youngs_modulus, poissons_ratio);
    lumped_mass_.resize(mesh_.vertices().rows()*mesh_.vertices().cols());
    const int element_num = static_cast<int>(mesh_.elements().cols());
    for (int e = 0; e < element_num; ++e) {
        for(int k = 0; k < element_dim; ++k){
            int vertex_index = mesh_.elements()(k,e);
            for(int vi = 0; vi < vertex_dim; ++vi){
                lumped_mass_[vertex_dim*vertex_index +vi] += density*mesh_.element_volume(e)/element_dim;
            }
        }
    }

    m_Us.second.setZero();
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Initialize(const Eigen::Matrix<real, vertex_dim, -1>& vertices,
    const Eigen::Matrix<int, element_dim, -1>& elements, const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    mesh_.Initialize(vertices, elements);
    InitializeAfterMesh(density, material_type, youngs_modulus, poissons_ratio);
    lumped_mass_.resize(vertices.rows()*vertices.cols());
    const int element_num = static_cast<int>(elements.cols());
    for (int e = 0; e < element_num; ++e) {
        for(int k = 0; k < element_dim; ++k){
            int vertex_index = elements(k,e);
            for(int vi = 0; vi < vertex_dim; ++vi){
                lumped_mass_[vertex_dim*vertex_index +vi] += density*mesh_.element_volume(e)/element_dim;
            }
        }
    }

    m_Us.second.setZero();

}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::InitializeAfterMesh(const real density,
    const std::string& material_type, const real youngs_modulus, const real poissons_ratio) {
    density_ = density;
    element_volume_ = mesh_.average_element_volume();
    
    material_ = InitializeMaterial(material_type, youngs_modulus, poissons_ratio);
    dofs_ = vertex_dim * mesh_.NumOfVertices();
    InitializeFiniteElementSamples();
    pd_solver_ready_ = false;
}

template<int vertex_dim, int element_dim>
const std::shared_ptr<Material<vertex_dim>> Deformable<vertex_dim, element_dim>::InitializeMaterial(const std::string& material_type,
    const real youngs_modulus, const real poissons_ratio) const {
    std::shared_ptr<Material<vertex_dim>> material(nullptr);
    if (material_type == "linear") {
        material = std::make_shared<LinearMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "corotated") {
        material = std::make_shared<CorotatedMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "neohookean") {
        material = std::make_shared<NeohookeanMaterial<vertex_dim>>();
        material->Initialize(youngs_modulus, poissons_ratio);
    } else if (material_type == "none") {
        material = nullptr;
    } else {
        PrintError("Unidentified material: " + material_type);
    }
    return material;
}

template<int vertex_dim,  int element_dim>
void Deformable<vertex_dim, element_dim>::InitializeStepperOptions(const std::map<std::string, real>& options) const {
        verbose_level = static_cast<int>(options.at("verbose"));
        CheckError(options.find("max_ls_iter") != options.end(), "Missing option max_ls_iter.");
        if (verbose_level > 1) std::cout<<"max_ls_iter: "<<options.at("max_ls_iter")<<"\n";
        CheckError(options.find("abs_tol") != options.end(), "Missing option abs_tol.");
        if (verbose_level > 1) std::cout<<"abs_tol: "<<options.at("abs_tol")<<"\n";
        CheckError(options.find("rel_tol") != options.end(), "Missing option rel_tol.");
        if (verbose_level > 1) std::cout<<"rel_tol: "<<options.at("rel_tol")<<"\n";
        CheckError(options.find("verbose") != options.end(), "Missing option verbose.");
        if (verbose_level > 1) std::cout<<"verbose: "<<options.at("verbose")<<"\n";
        CheckError(options.find("thread_ct") != options.end(), "Missing option thread_ct.");
        if (verbose_level > 1) std::cout<<"thread_ct: "<<options.at("thread_ct")<<"\n";
        thread_ct = static_cast<int>(options.at("thread_ct"));
        omp_set_num_threads(thread_ct);
        if (verbose_level > 1) std::cout<<"thread_ct: "<<thread_ct<<"\n";
        if (options.find("si_method") != options.end()) si_method = true;
        else si_method = false;
        std::cout<<"si_method: "<<si_method<<"\n";
        if (options.find("recompute_eigen_decomp_each_step") !=options.end()){
            recompute_eigen_decomp_each_step = static_cast<bool>(options.at("recompute_eigen_decomp_each_step"));
            if (verbose_level > 1) std::cout<<"recompute_eigen_decomp_each_step: "<<recompute_eigen_decomp_each_step<<"\n";
        }
        if (options.find("num_modes") != options.end()){
            num_modes = static_cast<int>(options.at("num_modes"));
            if (verbose_level > 1) std::cout<<"num_modes: "<<num_modes<<"\n";
        }
        if (options.find("theta_parameter") != options.end()){
            theta_parameter = static_cast<real>(options.at("theta_parameter"));
            if (verbose_level > 1) std::cout<<"theta_parameter: "<<theta_parameter<<"\n";
        }
}
template<int vertex_dim, int element_dim>
const void Deformable<vertex_dim, element_dim>::ContactDirichlet(const VectorXr& q, const std::vector<int>& contact_idx, std::map<int, real>& augmented_dirichlet) const{

    std::cout<<"Fix dirichlet_ + active_contact_nodes.\n";
    augmented_dirichlet = dirichlet_;
    for (const int idx : contact_idx) {
        for (int i = 0; i < vertex_dim; ++i){
            augmented_dirichlet[idx * vertex_dim + i] = q(idx * vertex_dim + i);
        }
    }
}
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::GetG() const{
    if (verbose_level > 1) std::cout<<"before g\n";
    CheckError(state_forces_.size() <= 1, "Only one state force, gravity, is supported for SIERE");
    if(state_forces_.size() == 1) {
        g = state_forces_[0]->parameters().head(vertex_dim);
    } else {
        g = VectorXr::Zero(vertex_dim);
    }
}
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ApplyDirichlet(const std::map<int, real>& dirichlet_with_friction, VectorXr& q, VectorXr& v) const{
    std::cout<<"ApplyDirichlet"<<std::endl;
    for (auto& it : dirichlet_with_friction) {
        const int index = it.first;
        const real value = it.second;
        q(index) = value;
        v(index) = 0;
    }
}
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ApplyDirichlet(const std::map<int, real>& dirichlet_with_friction, VectorXr& vector) const{
    std::cout<<"ApplyDirichlet to vector"<<std::endl;
    for (auto& it : dirichlet_with_friction) {
        const int index = it.first;
        vector(index) = 0;
    }
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupMatrices(const VectorXr& q, const VectorXr& a, const std::map<int, real>& augmented_dirichlet,
        const bool use_precomputed_data) const {

            if (verbose_level > 1) Tic();
            Deformable<vertex_dim,element_dim>::stiffness = -StiffnessMatrix(q, a, augmented_dirichlet, use_precomputed_data);
            if (verbose_level > 1) Toc("Assemble Stiffness Matrix");

            if (verbose_level > 1) Tic();
            Deformable<vertex_dim,element_dim>::lumped_mass = LumpedMassMatrix(augmented_dirichlet);
            if (verbose_level > 1) Toc("Assemble Mass Matrix");

            if (verbose_level > 1) Tic();
            Deformable<vertex_dim,element_dim>::lumped_mass_inv = LumpedMassMatrixInverse(augmented_dirichlet);
            if (verbose_level > 1) Toc("Assemble Mass Matrix Inverse");

            Deformable<vertex_dim,element_dim>::MinvK = Deformable<vertex_dim,element_dim>::lumped_mass_inv * stiffness;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SimpleForce(const VectorXr& q, const VectorXr& a, const std::map<int, real>& augmented_dirichlet,
        const bool use_precomputed_data, const VectorXr& g, VectorXr& force) const {

            std::cout<<"force\n";
            gravitational_force = lumped_mass * g.replicate(dofs()/vertex_dim, 1);
            force= ElasticForce(q) + PdEnergyForce(q, use_precomputed_data) + ActuationForce(q, a) + gravitational_force;
            ApplyDirichlet(augmented_dirichlet, force);
        }
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::MassPCA(const SparseMatrix lumped_mass, const SparseMatrix MinvK, const int pca_dim, const std::vector<int>& active_contact_idx) const{
    std::cout<<"MassPCA"<<std::endl;
    int constraint_dim = static_cast<int>(active_contact_idx.size());
    Spectra::SparseGenRealShiftSolvePardiso<real> op(MinvK);
    int DecomposedDim = std::max(pca_dim+2*vertex_dim,pca_dim+ vertex_dim * (int)constraint_dim);
    Spectra::GenEigsRealShiftSolver<Spectra::SparseGenRealShiftSolvePardiso<real>> eigs(op, DecomposedDim, std::min(2*(DecomposedDim),dofs()), 0.01);
    
    VectorXr ritz_error = VectorXr::Zero(DecomposedDim);
    
    if(m_Us.second.sum() != 0){
        ritz_error = (MinvK * m_Us.first - m_Us.first * m_Us.second.asDiagonal()).colwise().norm();
        ritz_error_norm = ritz_error.maxCoeff();
    }

    if(true){
    // if(ritz_error_norm > 1){
        // Initialize and compute
        eigs.init();
        Tic();
        eigs.compute(Spectra::SortRule::LargestMagn);
        Toc("Eigen Solve");
        Eigen::VectorXd normalizing_const;
        if(eigs.info() == Spectra::CompInfo::Successful)
        {
            m_Us = std::make_pair(eigs.eigenvectors().real().leftCols(pca_dim), eigs.eigenvalues().real().head(pca_dim));
            normalizing_const.noalias() = (m_Us.first.transpose() * lumped_mass * m_Us.first).diagonal();
            normalizing_const = normalizing_const.cwiseSqrt().cwiseInverse();
            
            m_Us.first = m_Us.first * (normalizing_const.asDiagonal());

        }
        else{
            std::cout<<"eigen solve failed"<<std::endl;
            exit(1);
        }
    }

    for(auto i: active_contact_idx){
        for(int j = 0; j < vertex_dim; j++){
            m_Us.first.row(i*vertex_dim+j).setZero();
        }
    }
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SetupJacobian(std::vector<int>& active_contact_idx) const{
    std::cout<<"SetupJacobian"<<std::endl;

    J12.resize(dofs()*2,dofs()*2);
    J21.resize(dofs()*2,dofs()*2);

    J12.setZero();
    J21.setZero();

    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletListJ12;
    tripletListJ12.reserve(dofs());
    for(int i = 0; i < dofs(); i++)
    {
        if(std::find(active_contact_idx.begin(),active_contact_idx.end(),i/vertex_dim) == active_contact_idx.end()){          
            tripletListJ12.push_back(T(i,i+dofs(),1.0));
        }
    }
    J12.setFromTriplets(tripletListJ12.begin(),tripletListJ12.end());
        
    J21_J22_outer_ind_ptr.erase(J21_J22_outer_ind_ptr.begin(),J21_J22_outer_ind_ptr.end());
    J21_outer_ind_ptr.erase(J21_outer_ind_ptr.begin(),J21_outer_ind_ptr.end());
    
    J21_outer_ind_ptr.erase(J21_outer_ind_ptr.begin(),J21_outer_ind_ptr.end());

    for (int i_row = 0; i_row < MinvK.rows() + 1; i_row++) {
        J21_J22_outer_ind_ptr.push_back(*(MinvK.outerIndexPtr()+i_row));
        J21_outer_ind_ptr.push_back(*(MinvK.outerIndexPtr()+i_row));
    }

    for (int i_row = 0; i_row < MinvK.rows(); i_row++) {
        J21_outer_ind_ptr.push_back(0);
    }
    
    J21_inner_ind.erase(J21_inner_ind.begin(),J21_inner_ind.end());
    
    for (int i_nnz = 0; i_nnz < MinvK.nonZeros(); i_nnz++)
    {
        J21_inner_ind.push_back(*(MinvK.innerIndexPtr()+i_nnz) + MinvK.cols());
    }
    
    Eigen::Map<SparseMatrix> J21_map((MinvK.rows())*2, (MinvK.cols())*2, MinvK.nonZeros(), J21_outer_ind_ptr.data(), J21_inner_ind.data(), (MinvK).valuePtr());
    J21 = J21_map;
}


template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ComputePCAProjection(const std::vector<int>& active_contact_idx) const{
    std::cout<<"ComputeProjection"<<std::endl;

    U1.resize((MinvK.rows())*2,m_Us.first.cols());
    V1.resize((MinvK.rows())*2,m_Us.first.cols());
    U2.resize((MinvK.rows())*2,m_Us.first.cols());
    V2.resize((MinvK.rows())*2,m_Us.first.cols());
    
    U1.setZero();
    V1.setZero();
    U2.setZero();
    V2.setZero();
    
    MatrixXr inertial = lumped_mass * m_Us.first;

    for(auto i: active_contact_idx){
        for(int j = 0; j < vertex_dim; j++){
            inertial.row(i*vertex_dim+j).setZero();
        }
    }


    U1.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first;
    V1.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << inertial;
    U2.block(m_Us.first.rows(),0,m_Us.first.rows(),m_Us.first.cols()) << m_Us.first * (m_Us.second.asDiagonal());
    V2.block(0,0,m_Us.first.rows(),m_Us.first.cols()) << inertial;
    
    for(auto i: active_contact_idx){
        for(int j = 0; j < vertex_dim; j++){
            U1.row(i*vertex_dim+j).setZero();
            U2.row(i*vertex_dim+j).setZero();
            V1.row(i*vertex_dim+j).setZero();
            V2.row(i*vertex_dim+j).setZero();
            U1.row(i*vertex_dim+j + dofs()).setZero();
            U2.row(i*vertex_dim+j + dofs()).setZero();
            V1.row(i*vertex_dim+j + dofs()).setZero();
            V2.row(i*vertex_dim+j + dofs()).setZero();
        }
    }

}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SplitVelocityState(const VectorXr& v) const{
    std::cout<<"SplitStates"<<std::endl;

    vG.noalias() = m_Us.first * (m_Us.first.transpose() * lumped_mass * v);
    
    vH = -vG;
    vH.noalias() += v;

}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SplitForceState(const VectorXr& f) const{
    std::cout<<"SplitForceStates"<<std::endl;

    fG.noalias() = (lumped_mass * m_Us.first ) * (m_Us.first.transpose() * f);
    fH = f - fG;

}
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ComputeReducedRhs(VectorXr& reduced_rhs, const VectorXr& v_sol, const VectorXr& force_sol, const real h) const{
    std::cout<<"ComputeReducedRhs"<<std::endl;

    dt_J_G_reduced.resize(m_Us.first.cols()*2,m_Us.first.cols()*2);
    dt_J_G_reduced.setZero();
    dt_J_G_reduced.block(0,m_Us.first.cols(),m_Us.first.cols(),m_Us.first.cols()).setIdentity();
    for (int ind = 0; ind < m_Us.first.cols() ; ind++) {
        dt_J_G_reduced(m_Us.first.cols() + ind ,0 + ind ) = m_Us.second(ind);
    }
    dt_J_G_reduced *= h;
    VectorXr reduced_vec;
    reduced_vec.resize(dt_J_G_reduced.cols());
    reduced_vec.head(dt_J_G_reduced.cols()/2).noalias() = m_Us.first.transpose() * (lumped_mass * (v_sol));
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
    if (verbose_level > 1) std::cout<<"phi: "<<std::endl;
    
    reduced_rhs = (-h) * block_diag_eigv * phi_reduced * reduced_vec;
    
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SubspaceEREUpdate(VectorXr& x0, const PardisoSolver& solver, const real h) const{
    std::cout<<"SubspaceEREUpdate"<<std::endl;
    U1 *= h;
    MatrixXr x1;
    x1 = solver.Solve(U1);

    U2 *= h;
    MatrixXr x2;
    x2 = solver.Solve(U2);

    if (verbose_level > 1) std::cout<<"Solving for the SMW"<<std::endl;
    MatrixXr Is;
    Is.resize(U1.cols(),U1.cols());
    Is.setIdentity();
    
    MatrixXr sol1LHS = Is + V1.transpose()*x1;
    MatrixXr sol1RHS = V1.transpose()*x2;
    x0.noalias() -= x1 * sol1LHS.ldlt().solve(V1.transpose()*x0);
    x2.noalias() -= x1 * sol1LHS.ldlt().solve(sol1RHS);
    MatrixXr sol2LHS = Is + V2.transpose()*x2;
    MatrixXr sol2RHS = V2.transpose()*x0;
    x0.noalias() -= x2 * (sol2LHS).ldlt().solve(sol2RHS);
}
template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::StiffnessMatrix(const VectorXr& q_sol, const VectorXr& a, const std::map<int, real>& dirichlet_with_friction, const bool use_precomputed_data) const {
    std::cout<<"StiffnessMatrix"<<std::endl;
    SparseMatrixElements nonzeros = ElasticForceDifferential(q_sol);
    SparseMatrixElements nonzeros_pd, nonzeros_dummy;
    SparseMatrixElements nonzeros_new;
    for (const auto& element : nonzeros) {
        const int row = element.row();
        const int col = element.col();
        const real val = element.value();
        if (dirichlet_with_friction.find(row) != dirichlet_with_friction.end()
            || dirichlet_with_friction.find(col) != dirichlet_with_friction.end()) continue;
        nonzeros_new.push_back(Eigen::Triplet<real>(row, col, -val));
    }
    return ToSparseMatrix(dofs_, dofs_, nonzeros_new);
}
    

template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::LumpedMassMatrix(const std::map<int, real>& dirichlet_with_friction) const {
    std::cout<<"LumpedMassMatrix"<<std::endl;
    SparseMatrixElements nonzeros_new;
        for (int i = 0; i < dofs_; ++i) {
        if (dirichlet_with_friction.find(i) != dirichlet_with_friction.end())
            nonzeros_new.push_back(Eigen::Triplet<real>(i, i, 1));
        else
            nonzeros_new.push_back(Eigen::Triplet<real>(i, i, lumped_mass_[i]));
    }

    return ToSparseMatrix(dofs_, dofs_, nonzeros_new);
}
template<int vertex_dim, int element_dim>
const SparseMatrix Deformable<vertex_dim, element_dim>::LumpedMassMatrixInverse(const std::map<int, real>& dirichlet_with_friction) const {
    std::cout<<"LumpedMassMatrixInverse"<<std::endl;
    SparseMatrixElements nonzeros_new;
        for (int i = 0; i < dofs_; ++i) {
        if (dirichlet_with_friction.find(i) != dirichlet_with_friction.end())
            nonzeros_new.push_back(Eigen::Triplet<real>(i, i, 1));
        else
            nonzeros_new.push_back(Eigen::Triplet<real>(i, i, 1/lumped_mass_[i]));
    }

    return ToSparseMatrix(dofs_, dofs_, nonzeros_new);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::phi(MatrixXr &A, MatrixXr &output) const
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
void Deformable<vertex_dim, element_dim>::Forward(const std::string& method, const VectorXr& q, const VectorXr& v,
    const VectorXr& a, const VectorXr& f_ext,
    const real dt, const std::map<std::string, real>& options, VectorXr& q_next, VectorXr& v_next,
    std::vector<int>& active_contact_idx) const {
    if (method == "semi_implicit") ForwardSemiImplicit(q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "pd")) ForwardProjectiveDynamics(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "newton")) ForwardNewton(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "sibe")) ForwardSIBE(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "bdf")) ForwardBDFFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "bdfere")) ForwardBDFEREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "bdf2")) ForwardBDF2FULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "bdf2ere")) ForwardBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "sbdf2")) ForwardSBDF2FULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "sbdf2ere")) ForwardSBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "trbdf2")) ForwardTRBDF2FULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "trbdf2ere")) ForwardTRBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "trerebdf2ere")) ForwardTREREBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "strbdf2")) ForwardSTRBDF2FULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "strbdf2ere")) ForwardSTRBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "strerebdf2ere")) ForwardSTREREBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "seretrbdf2")) ForwardSERETRBDF2FULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "seretrbdf2fixedj")) ForwardSERETRBDF2FIXEDJFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "theta") && EndsWith(method,"strbdf2")) ForwardTHETASTRBDF2FULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "theta") && EndsWith(method,"strbdf2ere")) ForwardTHETASTRBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "theta") && EndsWith(method,"strerebdf2ere")) ForwardTHETASTREREBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "theta") && EndsWith(method,"trbdf2")) ForwardTHETATRBDF2FULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "theta") && EndsWith(method,"trbdf2ere")) ForwardTHETATRBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (BeginsWith(method, "theta") && EndsWith(method,"trerebdf2ere")) ForwardTHETATREREBDF2EREFULL(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else if (StringsEqual(method, "siere")) ForwardSIERE(method, q, v, a, f_ext, dt, options, q_next, v_next, active_contact_idx);
    else PrintError("Unsupported forward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::Backward(const std::string& method, const VectorXr& q, const VectorXr& v, const VectorXr& a,
    const VectorXr& f_ext, const real dt, const VectorXr& q_next, const VectorXr& v_next,
    const std::vector<int>& active_contact_idx, const VectorXr& dl_dq_next,
    const VectorXr& dl_dv_next, const std::map<std::string, real>& options,
    VectorXr& dl_dq, VectorXr& dl_dv, VectorXr& dl_da, VectorXr& dl_df_ext,
    VectorXr& dl_dmat_w, VectorXr& dl_dact_w, VectorXr& dl_dstate_p) const {
    if (method == "semi_implicit")
        BackwardSemiImplicit(q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dmat_w, dl_dact_w, dl_dstate_p);
    else if (BeginsWith(method, "pd"))
        BackwardProjectiveDynamics(method, q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dmat_w, dl_dact_w, dl_dstate_p);
    else if (BeginsWith(method, "newton"))
        BackwardNewton(method, q, v, a, f_ext, dt, q_next, v_next, active_contact_idx, dl_dq_next, dl_dv_next, options,
            dl_dq, dl_dv, dl_da, dl_df_ext, dl_dmat_w, dl_dact_w, dl_dstate_p);
    else
        PrintError("Unsupported backward method: " + method);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::SaveToMeshFile(const VectorXr& q, const std::string& obj_file_name) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent q size. " + std::to_string(q.size())
        + " != " + std::to_string(dofs_));
    Mesh<vertex_dim, element_dim> mesh;
    mesh.Initialize(Eigen::Map<const MatrixXr>(q.data(), vertex_dim, dofs_ / vertex_dim), mesh_.elements());
    mesh.SaveToFile(obj_file_name);
}

// For Python binding.
template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyForward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& a, const std::vector<real>& f_ext, const real dt, const std::map<std::string, real>& options,
    std::vector<real>& q_next, std::vector<real>& v_next, std::vector<int>& active_contact_idx) const {
    VectorXr q_next_eig, v_next_eig;
    Forward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(a), ToEigenVector(f_ext), dt, options, q_next_eig, v_next_eig,
        active_contact_idx);
    q_next = ToStdVector(q_next_eig);
    v_next = ToStdVector(v_next_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PyBackward(const std::string& method, const std::vector<real>& q, const std::vector<real>& v,
    const std::vector<real>& a, const std::vector<real>& f_ext, const real dt, const std::vector<real>& q_next,
    const std::vector<real>& v_next, const std::vector<int>& active_contact_idx,
    const std::vector<real>& dl_dq_next, const std::vector<real>& dl_dv_next,
    const std::map<std::string, real>& options,
    std::vector<real>& dl_dq, std::vector<real>& dl_dv, std::vector<real>& dl_da, std::vector<real>& dl_df_ext,
    std::vector<real>& dl_dmat_w, std::vector<real>& dl_dact_w, std::vector<real>& dl_dstate_p) const {
    VectorXr dl_dq_eig, dl_dv_eig, dl_da_eig, dl_df_ext_eig, dl_dmat_w_eig, dl_dact_w_eig, dl_dstate_p_eig;
    Backward(method, ToEigenVector(q), ToEigenVector(v), ToEigenVector(a), ToEigenVector(f_ext), dt, ToEigenVector(q_next),
        ToEigenVector(v_next), active_contact_idx, ToEigenVector(dl_dq_next), ToEigenVector(dl_dv_next), options,
        dl_dq_eig, dl_dv_eig, dl_da_eig, dl_df_ext_eig, dl_dmat_w_eig, dl_dact_w_eig, dl_dstate_p_eig);
    dl_dq = ToStdVector(dl_dq_eig);
    dl_dv = ToStdVector(dl_dv_eig);
    dl_da = ToStdVector(dl_da_eig);
    dl_df_ext = ToStdVector(dl_df_ext_eig);
    dl_dmat_w = ToStdVector(dl_dmat_w_eig);
    dl_dact_w = ToStdVector(dl_dact_w_eig);
    dl_dstate_p = ToStdVector(dl_dstate_p_eig);
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::PySaveToMeshFile(const std::vector<real>& q, const std::string& obj_file_name) const {
    SaveToMeshFile(ToEigenVector(q), obj_file_name);
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::PyElasticEnergy(const std::vector<real>& q) const {
    return ElasticEnergy(ToEigenVector(q));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyElasticForce(const std::vector<real>& q) const {
    return ToStdVector(ElasticForce(ToEigenVector(q)));
}

template<int vertex_dim, int element_dim>
const std::vector<real> Deformable<vertex_dim, element_dim>::PyElasticForceDifferential(
    const std::vector<real>& q, const std::vector<real>& dq) const {
    return ToStdVector(ElasticForceDifferential(ToEigenVector(q), ToEigenVector(dq)));
}

template<int vertex_dim, int element_dim>
const std::vector<std::vector<real>> Deformable<vertex_dim, element_dim>::PyElasticForceDifferential(
    const std::vector<real>& q) const {
    PrintWarning("PyElasticForceDifferential should only be used for small-scale problems and for testing purposes.");
    const SparseMatrixElements nonzeros = ElasticForceDifferential(ToEigenVector(q));
    std::vector<std::vector<real>> K(dofs_, std::vector<real>(dofs_, 0));
    for (const auto& triplet : nonzeros) {
        K[triplet.row()][triplet.col()] += triplet.value();
    }
    return K;
}

template<int vertex_dim, int element_dim>
const real Deformable<vertex_dim, element_dim>::ElasticEnergy(const VectorXr& q) const {
    if (!material_) return 0;

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();

    std::vector<real> element_energy(element_num, 0);
    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            element_energy[i] += material_->EnergyDensity(F) * element_volume_ / sample_num;
        }
    }
    real energy = 0;
    for (const real e : element_energy) energy += e;
    return energy;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForce(const VectorXr& q) const {
    if (!material_) return VectorXr::Zero(dofs_);

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();

    std::vector<Eigen::Matrix<real, vertex_dim, 1>> f_ints(element_num * element_dim,
        Eigen::Matrix<real, vertex_dim, 1>::Zero());

    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            const auto P = material_->StressTensor(F);
            const Eigen::Matrix<real, 1, vertex_dim * element_dim> f_kd =
                -Flatten(P).transpose() * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
            for (int k = 0; k < element_dim; ++k) {
                f_ints[i * element_dim + k] += Eigen::Matrix<real, vertex_dim, 1>(f_kd.segment(k * vertex_dim, vertex_dim));
            }
        }
    }

    VectorXr f_int = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_num; ++i) {
        const auto vi = mesh_.element(i);
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                f_int(vi(j) * vertex_dim + k) += f_ints[i * element_dim + j](k);
            }
        }
    }

    return f_int;
}

template<int vertex_dim, int element_dim>
const SparseMatrixElements Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q) const {
    if (!material_) return SparseMatrixElements();

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();
    // The sequential version:
    // SparseMatrixElements nonzeros;
    SparseMatrixElements nonzeros(element_num * sample_num * element_dim * vertex_dim * element_dim * vertex_dim);

    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(i);
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            MatrixXr dF(vertex_dim * vertex_dim, element_dim * vertex_dim); dF.setZero();
            for (int s = 0; s < element_dim; ++s)
                for (int t = 0; t < vertex_dim; ++t) {
                    const Eigen::Matrix<real, vertex_dim, vertex_dim> dF_single =
                        Eigen::Matrix<real, vertex_dim, 1>::Unit(t) * finite_element_samples_[i][j].grad_undeformed_sample_weight().row(s);
                    dF.col(s * vertex_dim + t) += Flatten(dF_single);
                }
            const auto dP = material_->StressTensorDifferential(F) * dF;
            const Eigen::Matrix<real, element_dim * vertex_dim, element_dim * vertex_dim> df_kd =
                -dP.transpose() * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
            for (int k = 0; k < element_dim; ++k)
                for (int d = 0; d < vertex_dim; ++d)
                    for (int s = 0; s < element_dim; ++s)
                        for (int t = 0; t < vertex_dim; ++t)
                            nonzeros[i * sample_num * element_dim * vertex_dim * element_dim * vertex_dim
                                + j * element_dim * vertex_dim * element_dim * vertex_dim
                                + k * vertex_dim * element_dim * vertex_dim
                                + d * element_dim * vertex_dim
                                + s * vertex_dim
                                + t] = Eigen::Triplet<real>(vertex_dim * vi(k) + d, vertex_dim * vi(s) + t,
                                    df_kd(s * vertex_dim + t, k * vertex_dim + d));
                            // Below is the sequential version:
                            // nonzeros.push_back(Eigen::Triplet<real>(vertex_dim * vi(k) + d,
                            //     vertex_dim * vi(s) + t, df_kd(s * vertex_dim + t, k * vertex_dim + d)));
        }
    }
    return nonzeros;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::ElasticForceDifferential(const VectorXr& q, const VectorXr& dq) const {
    if (!material_) return VectorXr::Zero(dofs_);

    const int element_num = mesh_.NumOfElements();
    const int sample_num = GetNumOfSamplesInElement();

    std::vector<Eigen::Matrix<real, vertex_dim, 1>> df_ints(element_num * element_dim,
        Eigen::Matrix<real, vertex_dim, 1>::Zero());

    #pragma omp parallel for
    for (int i = 0; i < element_num; ++i) {
        const auto deformed = ScatterToElement(q, i);
        const auto ddeformed = ScatterToElement(dq, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            const auto dF = DeformationGradient(i, ddeformed, j);
            const auto dP = material_->StressTensorDifferential(F, dF);
            const Eigen::Matrix<real, 1, vertex_dim * element_dim> df_kd =
                -Flatten(dP).transpose() * finite_element_samples_[i][j].dF_dxkd_flattened() * element_volume_ / sample_num;
            for (int k = 0; k < element_dim; ++k) {
                df_ints[i * element_dim + k] += Eigen::Matrix<real, vertex_dim, 1>(df_kd.segment(k * vertex_dim, vertex_dim));
            }
        }
    }

    VectorXr df_int = VectorXr::Zero(dofs_);
    for (int i = 0; i < element_num; ++i) {
        const auto vi = mesh_.element(i);
        for (int j = 0; j < element_dim; ++j) {
            for (int k = 0; k < vertex_dim; ++k) {
                df_int(vi(j) * vertex_dim + k) += df_ints[i * element_dim + j](k);
            }
        }
    }

    return df_int;
}

template<int vertex_dim, int element_dim>
const VectorXr Deformable<vertex_dim, element_dim>::GetUndeformedShape() const {
    VectorXr q = VectorXr::Zero(dofs_);
    const int vertex_num = mesh_.NumOfVertices();
    for (int i = 0; i < vertex_num; ++i) q.segment(vertex_dim * i, vertex_dim) = mesh_.vertex(i);
    for (const auto& pair : dirichlet_) {
        q(pair.first) = pair.second;
    }
    return q;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim, element_dim> Deformable<vertex_dim, element_dim>::ScatterToElement(
    const VectorXr& q, const int element_idx) const {
    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_idx);
    Eigen::Matrix<real, vertex_dim, element_dim> deformed;
    for (int j = 0; j < element_dim; ++j) {
        deformed.col(j) = q.segment(vertex_dim * vi(j), vertex_dim);
    }
    return deformed;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim * element_dim, 1> Deformable<vertex_dim, element_dim>::ScatterToElementFlattened(
    const VectorXr& q, const int element_idx) const {
    const Eigen::Matrix<int, element_dim, 1> vi = mesh_.element(element_idx);
    Eigen::Matrix<real, vertex_dim * element_dim, 1> deformed;
    for (int j = 0; j < element_dim; ++j) {
        deformed.segment(j * vertex_dim, vertex_dim) = q.segment(vertex_dim * vi(j), vertex_dim);
    }
    return deformed;
}

template<int vertex_dim, int element_dim>
const Eigen::Matrix<real, vertex_dim, vertex_dim> Deformable<vertex_dim, element_dim>::DeformationGradient(
    const int element_idx, const Eigen::Matrix<real, vertex_dim, element_dim>& q, const int sample_idx) const {
    return q * finite_element_samples_[element_idx][sample_idx].grad_undeformed_sample_weight();
}

template<int vertex_dim, int element_dim>
const bool Deformable<vertex_dim, element_dim>::HasFlippedElement(const VectorXr& q) const {
    CheckError(static_cast<int>(q.size()) == dofs_, "Inconsistent number of elements.");
    const int sample_num = element_dim;
    for (int i = 0; i < mesh_.NumOfElements(); ++i) {
        const auto deformed = ScatterToElement(q, i);
        for (int j = 0; j < sample_num; ++j) {
            const auto F = DeformationGradient(i, deformed, j);
            if (F.determinant() < std::numeric_limits<real>::epsilon()) return true;
        }
    }
    return false;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::AssignToGlobalDeformable() const {
    global_deformable = this;
    global_vertex_dim = vertex_dim;
    global_element_dim = element_dim;
}

template<int vertex_dim, int element_dim>
void Deformable<vertex_dim, element_dim>::ClearGlobalDeformable() const {
    global_deformable = nullptr;
}

template class Deformable<2, 3>;
template class Deformable<2, 4>;
template class Deformable<3, 4>;
template class Deformable<3, 8>;

// Initialize the global variable used for the preconditioner.
const void* global_deformable = nullptr;
int global_vertex_dim = 0;
int global_element_dim = 0;
std::map<int, real> global_additional_dirichlet_boundary = std::map<int, real>();
std::string global_pd_backward_method = "";