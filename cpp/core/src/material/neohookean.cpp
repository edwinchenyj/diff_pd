#include "material/neohookean.h"
#include "common/geometry.h"

template<int dim>
const real NeohookeanMaterial<dim>::EnergyDensity(const Eigen::Matrix<real, dim, dim>& F) const {
    const real I1 = (F.transpose() * F).trace();
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    return mu / 2 * (I1 - dim) - mu * log_j + la / 2 * log_j * log_j;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeohookeanMaterial<dim>::StressTensor(const Eigen::Matrix<real, dim, dim>& F) const {
    // Useful derivatives:
    // grad J = grad |F| = |F| * F^-T
    // grad log(J) = F^-T
    // grad mu / 2 * (I1 - dim) = mu / 2 * (F : F - dim) = mu * F
    // grad mu * log_J = mu * F^-T
    // grad la / 2 * log_j^2 = la * log_j * F^-T.
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    const Eigen::Matrix<real, dim, dim> F_inv_T = F.inverse().transpose();
    return mu * (F - F_inv_T) + la * log_j * F_inv_T;
}

template<int dim>
const Eigen::Matrix<real, dim, dim> NeohookeanMaterial<dim>::StressTensorDifferential(const Eigen::Matrix<real, dim, dim>& F,
    const Eigen::Matrix<real, dim, dim>& dF) const {
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    const Eigen::Matrix<real, dim, dim> F_inv = F.inverse();
    const Eigen::Matrix<real, dim, dim> F_inv_T = F_inv.transpose();
    // F * F_inv = I.
    // dF * F_inv + F * dF_inv = 0.
    // dF_inv = -F_inv * dF * F_inv.
    const Eigen::Matrix<real, dim, dim> dF_inv = -F_inv * dF * F_inv;
    const Eigen::Matrix<real, dim, dim> dF_inv_T = dF_inv.transpose();
    const real dlog_j = (F_inv_T.array() * dF.array()).sum();
    return mu * (dF - dF_inv_T) + la * (log_j * dF_inv_T + dlog_j * F_inv_T);
}

template<int dim>
const Eigen::Matrix<real, dim * dim, dim * dim> NeohookeanMaterial<dim>::StressTensorDifferential(
    const Eigen::Matrix<real, dim, dim>& F) const {
    Eigen::Matrix<real, dim * dim, dim * dim> ret; ret.setZero();
    const real J = F.determinant();
    const real mu = Material<dim>::mu();
    const real la = Material<dim>::lambda();
    const real log_j = std::log(J);
    const Eigen::Matrix<real, dim, dim> F_inv = F.inverse();
    const Eigen::Matrix<real, dim, dim> F_inv_T = F_inv.transpose();
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            Eigen::Matrix<real, dim, dim> dF; dF.setZero(); dF(i, j) = 1;
            const Eigen::Matrix<real, dim, dim> dF_inv = -F_inv.col(i) * F_inv.row(j);
            const Eigen::Matrix<real, dim, dim> dF_inv_T = dF_inv.transpose();
            const real dlog_j = F_inv_T(i, j);
            const Eigen::Matrix<real, dim, dim> dP = mu * (dF - dF_inv_T) + la * (log_j * dF_inv_T + dlog_j * F_inv_T);
            const int idx = i + j * dim;
            ret.col(idx) = Eigen::Map<const Eigen::Matrix<real, dim * dim, 1>>(dP.data(), dP.size());
        }
    return ret;
}

template class NeohookeanMaterial<2>;
template class NeohookeanMaterial<3>;