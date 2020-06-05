#ifndef COMMON_QUAD_MESH_H
#define COMMON_QUAD_MESH_H

#include "common/config.h"
#include "common/common.h"

class QuadMesh {
public:
    QuadMesh();

    void Initialize(const std::string& obj_file_name);
    void Initialize(const Matrix2Xr& vertices, const Matrix4Xi& faces);
    void SaveToFile(const std::string& obj_file_name) const;

    const Matrix2Xr& vertices() const { return vertices_; }
    const Matrix4Xi& faces() const { return faces_; }

    const int NumOfVertices() const {
        return static_cast<int>(vertices_.cols());
    }
    const int NumOfFaces() const {
        return static_cast<int>(faces_.cols());
    }
    const Vector2r vertex(const int i) const {
        return vertices_.col(i);
    }
    const std::array<real, 2> py_vertex(const int i) const {
        return std::array<real, 2>{
            vertices_(0, i),
            vertices_(1, i)
        };
    }
    const std::vector<real> py_vertices() const {
        const VectorXr q(Eigen::Map<const VectorXr>(vertices_.data(), vertices_.size()));
        return ToStdVector(q);
    }
    const Vector4i face(const int i) const {
        return faces_.col(i);
    }
    const std::array<int, 4> py_face(const int i) const {
        return std::array<int, 4>{
            faces_(0, i),
            faces_(1, i),
            faces_(2, i),
            faces_(3, i)
        };
    }

private:
    Matrix2Xr vertices_;
    Matrix4Xi faces_;
};

#endif