//! Frustum

use vectormath::float::Float;
use ::{Pos3, Vec3, Vec4, Mat3, Mat4};
use plane::Plane;
use camera::ProjectionMatrix;

// Let `v` = (x, y, z, w=1)^T be a vertex and `M` = (m[i,j]) be a 4x4 projection matrix.
// Transforming `v` with `M` results in the transformed vertex `v'` as below:
//
//   Mv = v' = (x',y',z',w')^T
//      = (dot(v, row[0]), dot(v, row[1]), dot(v, row[2]), dot(v, row[3]))^T
//
//   where row[i] is the i-th row of matrix `M`
//
// After the transformation, the vertex `v'` is in homogeneous clipping space.
// In the space, the viewing frustum actually is an axis-aligned box.
// Hence, the vertex `v'` is inside in the box if the following inequalities are all true for the components of `v'`:
//
//       -1 < (x' / w') < 1
//       -1 < (y' / w') < 1
//       near_depth < (z' / w') < far_depth
//
//       where near_depth and far_depth are API-specific z clipping range.
//       (near_depth, far_depth) = (-1, 1) in OpenGL, while
//       (near_depth, far_depth) = (0, 1) in Direct3D
//
// In conclusion, we can define the six clipping planes from these inequalities.
//
//    -w' < x'   : x' is in the inside-halfspace of the left clipping plane
//    x' < w'    : x' is in the inside-halfspace of the right clipping plane
//    -w' < y'   : y' is in the inside-halfspace of the bottom clipping plane
//    y' < w'    : y' is in the inside-halfspace of the top clipping plane
//    near_depth * w' < z' : z' is in the inside-halfspace of the near clipping plane
//    z' < far_depth * w' : z' is in the inside-halfspace of the far clipping plane
//
// Now suppose that we wanted to test the case, -w' < x'.
// Using the information from the beginning, the inequality can be rewritten as:
//
//       -(dot(v, row[3]) < (dot(v, row[0]))
//    => 0 < (dot(v, row[0])) + (dot(v, row[3]))
//    => 0 < dot(v, (row[0] + row[3]))
//
// Finally, the inequality becomes the inside-halfspace of a plane.
// And the plane equation is out:
//
//       a * x + b * y + c * z + d = 0
//
//       where a = (row[0] + row[3]).x
//             b = (row[0] + row[3]).y
//             c = (row[0] + row[3]).z
//             d = (row[0] + row[3]).w
//

// Supporting a Non-Identity Model-View Matrix
//
// 1. If the matrix `M` is a projection matrix P (i.e. M = P),
//    then the algorithm gives the clipping planes in view space.
//
// 2. If the matrix `M` is a combined World-View-Projection matrix
//    (i.e. M = P * V * W),
//    then the algorithm gives the clipping planes in world space.

/// Extract left clipping plane from Proj/MVP matrix
pub fn extract_left<T: Float>(m: ProjectionMatrix<T>) -> Plane<T> {
    let v = row3(&m.matrix) + row0(&m.matrix);
    Plane(v.x, v.y, v.z, v.w)
}

/// Extract right clipping plane from Proj/MVP matrix
pub fn extract_right<T: Float>(m: ProjectionMatrix<T>) -> Plane<T> {
    let v = row3(&m.matrix) - row0(&m.matrix);
    Plane(v.x, v.y, v.z, v.w)
}

/// Extract bottom clipping plane from Proj/MVP matrix
pub fn extract_bottom<T: Float>(m: ProjectionMatrix<T>) -> Plane<T> {
    let v = row3(&m.matrix) + row1(&m.matrix);
    Plane(v.x, v.y, v.z, v.w)
}

/// Extract top clipping plane from Proj/MVP matrix
pub fn extract_top<T: Float>(m: ProjectionMatrix<T>) -> Plane<T> {
    let v = row3(&m.matrix) - row1(&m.matrix);
    Plane(v.x, v.y, v.z, v.w)
}

/// Extract near clipping plane from Proj/MVP matrix
pub fn extract_near<T: Float>(m: ProjectionMatrix<T>) -> Plane<T> {
    let v = row3(&m.matrix) * (-m.depth_range.0) + row2(&m.matrix);
    Plane(v.x, v.y, v.z, v.w)
}

/// Extract far clipping plane from Proj/MVP matrix
pub fn extract_far<T: Float>(m: ProjectionMatrix<T>) -> Plane<T> {
    let v = row3(&m.matrix) * (m.depth_range.1) - row2(&m.matrix);
    Plane(v.x, v.y, v.z, v.w)
}

/// Extract near-left-bottom corner point
pub fn corner_near_left_bottom<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: -T::one(), y: -T::one(), z: m.depth_range.0  };
    find_corner(v, &m.matrix)
}

/// Extract near-left-top corner point
pub fn corner_near_left_top<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: -T::one(), y: T::one(), z: m.depth_range.0  };
    find_corner(v, &m.matrix)
}

/// Extract near-right-bottom corner point
pub fn corner_near_right_bottom<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: T::one(), y: -T::one(), z: m.depth_range.0  };
    find_corner(v, &m.matrix)
}

/// Extract near-right-top corner point
pub fn corner_near_right_top<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: T::one(), y: T::one(), z: m.depth_range.0  };
    find_corner(v, &m.matrix)
}

/// Extract far-left-bottom corner point
pub fn corner_far_left_bottom<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: -T::one(), y: -T::one(), z: m.depth_range.1  };
    find_corner(v, &m.matrix)
}

/// Extract far-left-top corner point
pub fn corner_far_left_top<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: -T::one(), y: T::one(), z: m.depth_range.1  };
    find_corner(v, &m.matrix)
}

/// Extract far-right-bottom corner point
pub fn corner_far_right_bottom<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: T::one(), y: -T::one(), z: m.depth_range.1  };
    find_corner(v, &m.matrix)
}

/// Extract far-right-top corner point
pub fn corner_far_right_top<T: Float>(m: ProjectionMatrix<T>) -> Pos3<T> {
    let v = Pos3 { x: T::one(), y: T::one(), z: m.depth_range.1  };
    find_corner(v, &m.matrix)
}

#[inline(always)]
fn row0<T: Copy>(m: &Mat4<T>) -> Vec4<T> {
    Vec4 { x: m.0.x, y: m.1.x, z: m.2.x, w: m.3.x }
}

#[inline(always)]
fn row1<T: Copy>(m: &Mat4<T>) -> Vec4<T> {
    Vec4 { x: m.0.y, y: m.1.y, z: m.2.y, w: m.3.y }
}

#[inline(always)]
fn row2<T: Copy>(m: &Mat4<T>) -> Vec4<T> {
    Vec4 { x: m.0.z, y: m.1.z, z: m.2.z, w: m.3.z }
}

#[inline(always)]
fn row3<T: Copy>(m: &Mat4<T>) -> Vec4<T> {
    Vec4 { x: m.0.w, y: m.1.w, z: m.2.w, w: m.3.w }
}

#[inline(always)]
fn find_corner<T: Float>(cp: Pos3<T>, a: &Mat4<T>) -> Pos3<T> {
    let (r0, r1, r2, r3) = (row0(&a), row1(&a), row2(&a), row3(&a));
    let m = Mat3(
        Vec3::from(r0 - r3 * cp.x),
        Vec3::from(r1 - r3 * cp.y),
        Vec3::from(r2 - r3 * cp.z),
    ).transpose();
    let b = Vec3 {
        x: cp.x * r3.w - r0.w,
        y: cp.y * r3.w - r1.w,
        z: cp.z * r3.w - r2.w,
    };
    Pos3::from(m.inverse() * b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::Pos3;
    use camera::ProjectionMatrix;

    fn proj_mat() -> ProjectionMatrix<f64> {
        use camera::CameraOrthographicProjection;
        let proj = CameraOrthographicProjection {
            left: -1.0,
            right: 2.0,
            bottom: -3.0,
            top: 4.0,
            near: 5.0,
            far: 6.0,
            near_depth: -7.0,
            far_depth: 8.0,
        };
        ProjectionMatrix::from(proj)
    }

    #[test]
    fn test_frustum_left() {
        let pl = extract_left(proj_mat());
        assert!(pl.distance_to(Pos3 { x: -0.5, y: 0.0, z: 5.5 }) > 0.0);
        assert_approx_eq!(pl.distance_to(Pos3 { x: -1.0, y: 0.0, z: 5.5 }), 0.0);
        assert!(pl.distance_to(Pos3 { x: -1.5, y: 0.0, z: 5.5 }) < 0.0);
    }

    #[test]
    fn test_frustum_right() {
        let pl = extract_right(proj_mat());
        assert!(pl.distance_to(Pos3 { x: 1.5, y: 0.0, z: 5.5 }) > 0.0);
        assert_approx_eq!(pl.distance_to(Pos3 { x: 2.0, y: 0.0, z: 5.5 }), 0.0);
        assert!(pl.distance_to(Pos3 { x: 2.5, y: 0.0, z: 5.5 }) < 0.0);
    }

    #[test]
    fn test_frustum_bottom() {
        let pl = extract_bottom(proj_mat());
        assert!(pl.distance_to(Pos3 { x: 0.0, y: -2.5, z: -5.5 }) > 0.0);
        assert_approx_eq!(pl.distance_to(Pos3 { x: 0.0, y: -3.0, z: -5.5 }), 0.0);
        assert!(pl.distance_to(Pos3 { x: 0.0, y: -3.5, z: -5.5 }) < 0.0);
    }

    #[test]
    fn test_frustum_top() {
        let pl = extract_top(proj_mat());
        assert!(pl.distance_to(Pos3 { x: 0.0, y: 3.5, z: -5.5 }) > 0.0);
        assert_approx_eq!(pl.distance_to(Pos3 { x: 0.0, y: 4.0, z: -5.5 }), 0.0);
        assert!(pl.distance_to(Pos3 { x: 0.0, y: 4.5, z: -5.5 }) < 0.0);
    }

    #[test]
    fn test_frustum_near() {
        let pl = extract_near(proj_mat());
        assert!(pl.distance_to(Pos3 { x: 0.0, y: 0.0, z: -5.5 }) > 0.0);
        assert_approx_eq!(pl.distance_to(Pos3 { x: 0.0, y: 0.0, z: -5.0 }), 0.0);
        assert!(pl.distance_to(Pos3 { x: 0.0, y: 0.0, z: -4.5 }) < 0.0);
    }

    #[test]
    fn test_frustum_far() {
        let pl = extract_far(proj_mat());
        assert!(pl.distance_to(Pos3 { x: 0.0, y: 0.0, z: -5.5 }) > 0.0);
        assert_approx_eq!(pl.distance_to(Pos3 { x: 0.0, y: 0.0, z: -6.0 }), 0.0);
        assert!(pl.distance_to(Pos3 { x: 0.0, y: 0.0, z: -6.5 }) < 0.0);
    }

    #[test]
    fn test_corner_near_left_bottom() {
        let p = corner_near_left_bottom(proj_mat());
        assert_approx_eq!(p.x, -1.0);
        assert_approx_eq!(p.y, -3.0);
        assert_approx_eq!(p.z, -5.0);
    }

    #[test]
    fn test_corner_near_left_top() {
        let p = corner_near_left_top(proj_mat());
        assert_approx_eq!(p.x, -1.0);
        assert_approx_eq!(p.y, 4.0);
        assert_approx_eq!(p.z, -5.0);
    }

    #[test]
    fn test_corner_near_right_bottom() {
        let p = corner_near_right_bottom(proj_mat());
        assert_approx_eq!(p.x, 2.0);
        assert_approx_eq!(p.y, -3.0);
        assert_approx_eq!(p.z, -5.0);
    }

    #[test]
    fn test_corner_near_right_top() {
        let p = corner_near_right_top(proj_mat());
        assert_approx_eq!(p.x, 2.0);
        assert_approx_eq!(p.y, 4.0);
        assert_approx_eq!(p.z, -5.0);
    }

    #[test]
    fn test_corner_far_left_bottom() {
        let p = corner_far_left_bottom(proj_mat());
        assert_approx_eq!(p.x, -1.0);
        assert_approx_eq!(p.y, -3.0);
        assert_approx_eq!(p.z, -6.0);
    }

    #[test]
    fn test_corner_far_left_top() {
        let p = corner_far_left_top(proj_mat());
        assert_approx_eq!(p.x, -1.0);
        assert_approx_eq!(p.y, 4.0);
        assert_approx_eq!(p.z, -6.0);
    }

    #[test]
    fn test_corner_far_right_bottom() {
        let p = corner_far_right_bottom(proj_mat());
        assert_approx_eq!(p.x, 2.0);
        assert_approx_eq!(p.y, -3.0);
        assert_approx_eq!(p.z, -6.0);
    }

    #[test]
    fn test_corner_far_right_top() {
        let p = corner_far_right_top(proj_mat());
        assert_approx_eq!(p.x, 2.0);
        assert_approx_eq!(p.y, 4.0);
        assert_approx_eq!(p.z, -6.0);
    }
}
