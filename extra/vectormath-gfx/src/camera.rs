//! Camera

use ::Float;
use vectormath::{Rad, Pos3, Vec3, Vec4, Mat3, Mat4, Tfm3, Quat};

#[derive(Debug, Copy, Clone)]
pub struct CameraTransform<T> {
    pub position: Pos3<T>,
    pub orientation: Quat<T>,
}

/// Camera Transform Matrix (view to world)
#[derive(Debug, Copy, Clone)]
pub struct CameraTransformMatrix<T>(Tfm3<T>);

#[derive(Debug, Copy, Clone)]
pub struct CameraPerspectiveProjection<T> {
    pub left: T,
    pub right: T,
    pub bottom: T,
    pub top: T,
    /// Near Distance
    pub near: T,
    /// Far Distance
    pub far: T,
    /// -1 in OpenGL, 0 in Direct3D
    pub near_depth: T,
    /// 1 in both OpenGL and Direct3D
    pub far_depth: T,
}

#[derive(Debug, Copy, Clone)]
pub struct CameraPerspectiveFovProjection<T> {
    /// Field of View
    pub fovy: Rad<T>,
    /// Aspect Ratio = Width / Height
    pub aspect: T,
    /// Near Distance
    pub near: T,
    /// Far Distance
    pub far: T,
    /// -1 in OpenGL, 0 in Direct3D
    pub near_depth: T,
    /// 1 in both OpenGL and Direct3D
    pub far_depth: T,
}

#[derive(Debug, Copy, Clone)]
pub struct CameraOrthographicProjection<T> {
    pub left: T,
    pub right: T,
    pub bottom: T,
    pub top: T,
    /// Near Distance
    pub near: T,
    /// Far Distance
    pub far: T,
    /// -1 in OpenGL, 0 in Direct3D
    pub near_depth: T,
    /// 1 in both OpenGL and Direct3D
    pub far_depth: T,
}

/// Projection Matrix
#[derive(Debug, Copy, Clone)]
pub struct ProjectionMatrix<T> {
    pub matrix: Mat4<T>,
    pub depth_range: (T, T),
}

impl<T: Float> CameraTransform<T> {
    // Constructor
    //pub fn new(eye: Pos3<T>, at: Pos3<T>, roll: Rad<T>) -> CameraTransform<T> {
    //    let forward0 = Vec3 { x: T::zero(), y: T::zero(), z: -T::one() };
    //    let forward1 = normalize(at - eye);
    //    let rot = Quat::from_vectors(forward0, forward1) * Quat::from_angle_axis(roll, forward0);
    //    CameraTransform{ position: eye, orientation: rot }
    //}
}

impl<T: Float> From<CameraTransformMatrix<T>> for CameraTransform<T> {
    fn from(mat: CameraTransformMatrix<T>) -> CameraTransform<T> {
        let q = Quat::from(Mat3::from(mat.0));
        let p = Pos3::from((mat.0).3);
        CameraTransform { position: p, orientation: q }
    }
}

impl<T: Float> From<CameraTransform<T>> for CameraTransformMatrix<T> {
    fn from(cam: CameraTransform<T>) -> CameraTransformMatrix<T> {
        let m = Mat3::from(cam.orientation);
        let t = Vec3::from(cam.position);
        CameraTransformMatrix(Tfm3(m.0, m.1, m.2, t))
    }
}

impl<T: Float> CameraTransformMatrix<T> {
    /// LookAt constructor
    pub fn look_at(eye: Pos3<T>, at: Pos3<T>, up: Vec3<T>) -> Self {
        let forward: Vec3<T> = (at - eye).normalize();
        let up = up.normalize();
        let right = forward.cross(up).normalize();
        let up = right.cross(forward);
        CameraTransformMatrix(Tfm3(right, up, -forward, Vec3::from(eye)))
    }
    /// World to View Matrix
    pub fn as_view_matrix(&self) -> Tfm3<T> {
        self.0.inverse()
    }
    /// World to View Normal Matrix
    pub fn as_normal_matrix(&self) -> Mat3<T> {
        // normal matrix = transpose( inverse ( M[0:3;0:3] ) ), where M is a view matrix
        Mat3::from(self.0).transpose()
    }
}

impl<T: Float> From<CameraPerspectiveProjection<T>> for ProjectionMatrix<T> {
    fn from(proj: CameraPerspectiveProjection<T>) -> ProjectionMatrix<T> {
        assert!(proj.left   < proj.right, "`left` should be less than `right`, found: left: {:?} right: {:?}", proj.left, proj.right);
        assert!(proj.bottom < proj.top,   "`bottom` should be less than `top`, found: bottom: {:?} top: {:?}", proj.bottom, proj.top);
        assert!(proj.near   < proj.far,   "`near` should be less than `far`, found: near: {:?} far: {:?}", proj.near, proj.far);
        assert!(proj.near   > T::zero(),  "`near` should be a positive number, found: near: {:?}", proj.near);
        assert!(proj.near_depth < proj.far_depth, "`near_depth` should be less than `far_depth`, found: near_depth: {:?}, far_depth: {:?}", proj.near_depth, proj.far_depth);

        let width = proj.right - proj.left;
        let height = proj.top - proj.bottom;
        let depth = proj.near - proj.far;
        let zero = T::zero();
        let zx = (proj.right + proj.left) / width;
        let zy = (proj.top + proj.bottom) / height;
        let zz = (proj.far_depth * proj.far - proj.near_depth * proj.near) / depth;
        let wz = (proj.far_depth - proj.near_depth) * proj.far * proj.near / depth;
        let m = Mat4(Vec4{ x: T::two() * proj.near / width, y: zero, z: zero, w: zero },
                     Vec4{ x: zero, y: T::two() * proj.near / height, z: zero, w: zero },
                     Vec4{ x: zx, y: zy, z: zz, w: -T::one() },
                     Vec4{ x: zero, y: zero, z: wz, w: zero });
        ProjectionMatrix{ matrix: m, depth_range: (proj.near_depth, proj.far_depth) }
    }
}

impl<T: Float> From<CameraPerspectiveFovProjection<T>> for ProjectionMatrix<T> {
    fn from(proj: CameraPerspectiveFovProjection<T>) -> ProjectionMatrix<T> {
        ProjectionMatrix::from(CameraPerspectiveProjection::from(proj))
    }
}

impl<T: Float> From<CameraOrthographicProjection<T>> for ProjectionMatrix<T> {
    fn from(proj: CameraOrthographicProjection<T>) -> ProjectionMatrix<T> {
        assert!(proj.left   < proj.right, "`left` should be less than `right`, found: left: {:?} right: {:?}", proj.left, proj.right);
        assert!(proj.bottom < proj.top,   "`bottom` should be less than `top`, found: bottom: {:?} top: {:?}", proj.bottom, proj.top);
        assert!(proj.near   < proj.far,   "`near` should be less than `far`, found: near: {:?} far: {:?}", proj.near, proj.far);
        assert!(proj.near   > T::zero(),  "`near` should be a positive number, found: near: {:?}", proj.near);
        assert!(proj.near_depth < proj.far_depth, "`near_depth` should be less than `far_depth`, found: near_depth: {:?}, far_depth: {:?}", proj.near_depth, proj.far_depth);

        let width = proj.right - proj.left;
        let height = proj.top - proj.bottom;
        let depth = proj.near - proj.far;
        let zero = T::zero();
        let tx = -(proj.right + proj.left) / width;
        let ty = -(proj.top + proj.bottom) / height;
        let tz = (proj.near * proj.far_depth - proj.far * proj.near_depth) / depth;
        let zfactor = proj.far_depth - proj.near_depth;
        let m = Mat4(Vec4{ x: T::two() / width, y: zero, z: zero, w: zero },
                     Vec4{ x: zero, y: T::two() / height, z: zero, w: zero },
                     Vec4{ x: zero, y: zero, z: zfactor / depth, w: zero },
                     Vec4{ x: tx, y: ty, z: tz, w: T::one() });
        ProjectionMatrix{ matrix: m, depth_range: (proj.near_depth, proj.far_depth) }
    }
}

impl<T: Float> From<CameraPerspectiveFovProjection<T>> for CameraPerspectiveProjection<T> {
    #[inline]
    fn from(p: CameraPerspectiveFovProjection<T>) -> CameraPerspectiveProjection<T> {
        assert!(p.fovy.0 > T::zero(), "The vertical field of view cannot be below zero, found: {:?}", p.fovy);
        //assert!(p.fovy   < TODO, "The vertical field of view cannot be greater than a half turn, found: {:?}", p.fovy);
        assert!(p.aspect > T::zero(), "The aspect ratio cannot be below zero, found: {:?}", p.aspect);

        let f = (p.fovy.0 * T::onehalf()).tan();
        let t = p.near * f;
        let r = t * p.aspect;
        CameraPerspectiveProjection {
            left: -r, right: r, bottom: -t, top: t, near: p.near, far: p.far,
            near_depth: p.near_depth, far_depth: p.far_depth,
        }
    }
}

//
// Reference:
//
// - http://www.songho.ca/opengl/gl_projectionmatrix.html
// - http://www.songho.ca/opengl/gl_normaltransform.html
// - https://solarianprogrammer.com/2013/05/22/opengl-101-matrices-projection-view-model/
//

#[cfg(test)]
mod tests {
    use super::*;
    use vectormath::{Rad, Deg, Vec3, Pos3, Quat, Tfm3};

    #[test]
    fn test_camera_transform() {
        let cam = CameraTransform {
                      position: Pos3 { x: -1.0, y: 2.0, z: 3.0 },
                      orientation: Quat::from_angle_axis(Rad::from(Deg(90.0)), Vec3::x_axis()),
                  };
        assert_eq!(cam.position.x, -1.0);
        assert_eq!(cam.position.y, 2.0);
        assert_eq!(cam.position.z, 3.0);
        let forward = cam.orientation.rotate(-Vec3::z_axis());
        assert_approx_eq!(forward.x, 0.0);
        assert_approx_eq!(forward.y, 1.0);
        assert_approx_eq!(forward.z, 0.0);
        let right = cam.orientation.rotate(Vec3::x_axis());
        assert_approx_eq!(right.x, 1.0);
        assert_approx_eq!(right.y, 0.0);
        assert_approx_eq!(right.z, 0.0);
        let up = cam.orientation.rotate(Vec3::y_axis());
        assert_approx_eq!(up.x, 0.0);
        assert_approx_eq!(up.y, 0.0);
        assert_approx_eq!(up.z, 1.0);
    }

    #[test]
    fn test_camera_transform_2_matrix() {
        let cam = CameraTransform {
                      position: Pos3 { x: -1.0, y: 2.0, z: 3.0 },
                      orientation: Quat::from_angle_axis(Rad::from(Deg(90.0)), Vec3::x_axis()),
                  };
        let mat = CameraTransformMatrix::from(cam).0;
        assert_approx_eq!(mat.0.x, 1.0);
        assert_approx_eq!(mat.0.y, 0.0);
        assert_approx_eq!(mat.0.z, 0.0);
        assert_approx_eq!(mat.1.x, 0.0);
        assert_approx_eq!(mat.1.y, 0.0);
        assert_approx_eq!(mat.1.z, 1.0);
        assert_approx_eq!(mat.2.x, 0.0);
        assert_approx_eq!(mat.2.y, -1.0);
        assert_approx_eq!(mat.2.z, 0.0);
        assert_eq!(mat.3.x, -1.0);
        assert_eq!(mat.3.y, 2.0);
        assert_eq!(mat.3.z, 3.0);
    }

    #[test]
    fn test_camera_matrix_2_transform() {
        let mat = CameraTransformMatrix(Tfm3(
            Vec3 { x: 1.0, y: 0.0, z: 0.0 },
            Vec3 { x: 0.0, y: 0.0, z: 1.0 },
            Vec3 { x: 0.0, y: -1.0, z: 0.0 },
            Vec3 { x: -1.0, y: 2.0, z: 3.0 },
        ));
        let cam = CameraTransform::from(mat);
        assert_approx_eq!(cam.orientation.w, 0.70710678118);
        assert_approx_eq!(cam.orientation.x, 0.70710678118);
        assert_approx_eq!(cam.orientation.y, 0.0);
        assert_approx_eq!(cam.orientation.z, 0.0);
        assert_approx_eq!(cam.position.x, -1.0);
        assert_approx_eq!(cam.position.y, 2.0);
        assert_approx_eq!(cam.position.z, 3.0);
    }

    #[test]
    fn test_camera_matrix_2_view_matrix() {
        let mat = CameraTransformMatrix(Tfm3(
            Vec3 { x: 1.0, y: 0.0, z: 0.0 },
            Vec3 { x: 0.0, y: 0.0, z: 1.0 },
            Vec3 { x: 0.0, y: -1.0, z: 0.0 },
            Vec3 { x: -1.0, y: 2.0, z: 3.0 },
        ));
        let m: Tfm3<_> = mat.as_view_matrix();
        let v1 = Pos3 { x: 10.0, y: 2.0, z: -3.0 };
        let v2 = m * v1;
        assert_approx_eq!(v2.x, 11.0);
        assert_approx_eq!(v2.y, -6.0);
        assert_approx_eq!(v2.z, 0.0);
    }

    #[test]
    fn test_camera_matrix_lookat() {
        let eye = Pos3 { x: -1.0, y: 3.0, z: 10.0 };
        let at = Pos3 { x: -1.0, y: 3.0, z: -1.0 };
        let up = Vec3 { x: 0.0, y: 1.0, z: 1.0 };
        let mat = CameraTransformMatrix::look_at(eye, at, up);
        assert_approx_eq!((mat.0).0.x, 1.0);
        assert_approx_eq!((mat.0).0.y, 0.0);
        assert_approx_eq!((mat.0).0.z, 0.0);
        assert_approx_eq!((mat.0).1.x, 0.0);
        assert_approx_eq!((mat.0).1.y, 1.0);
        assert_approx_eq!((mat.0).1.z, 0.0);
        assert_approx_eq!((mat.0).2.x, 0.0);
        assert_approx_eq!((mat.0).2.y, 0.0);
        assert_approx_eq!((mat.0).2.z, 1.0);
        assert_approx_eq!((mat.0).3.x, -1.0);
        assert_approx_eq!((mat.0).3.y, 3.0);
        assert_approx_eq!((mat.0).3.z, 10.0);
    }

    #[test]
    fn test_camera_matrix_2_normal_matrix() {
        let mat = CameraTransformMatrix(Tfm3(
            Vec3 { x: 0.0, y: 1.0, z: 0.0 },
            Vec3 { x: -2.0, y: 0.0, z: 0.0 },
            Vec3 { x: 0.0, y: 0.0, z: 3.0 },
            Vec3 { x: -10.0, y: 20.0, z: 30.0 },
        ));
        let mt = mat.as_view_matrix();
        let mn = mat.as_normal_matrix();
        let p0 = Pos3 { x: 0.0, y: 0.0, z: 0.0 };
        let p1 = Pos3 { x: -1.0, y: 2.0, z: 1.0 };
        let v0 = Vec3 { x: 1.0, y: 1.0, z: -1.0 };
        let v1 = (mn * v0).normalize();
        assert_approx_eq!((p1 - p0).dot(v0), 0.0);
        assert_approx_eq!((mt * p1 - mt * p0).dot(v1), 0.0);
    }

    #[test]
    fn test_perspective_projection() {
        let proj = CameraPerspectiveProjection {
            left: -1.5 * 5.0,
            right: 1.5 * 5.0,
            bottom: -1.0 * 5.0,
            top: 1.0 * 5.0,
            near: 5.0,
            far: 25.0,
            near_depth: -10.0,
            far_depth: 20.0,
        };
        let m = ProjectionMatrix::from(proj);
        assert_eq!(m.depth_range.0, -10.0);
        assert_eq!(m.depth_range.1, 20.0);
        let v = Pos3::from(m.matrix * Pos3 { x: -7.5, y: 0.0, z: -5.0 });
        assert_approx_eq!(v.x, -1.0);
        assert_approx_eq!(v.y, 0.0);
        assert_approx_eq!(v.z, -10.0);
        let v = Pos3::from(m.matrix * Pos3 { x: 0.0, y: -12.5, z: -25.0 });
        assert_approx_eq!(v.x, 0.0);
        assert_approx_eq!(v.y, -0.5);
        assert_approx_eq!(v.z, 20.0);
    }

    #[test]
    fn test_perspective_from_fov() {
        let proj = CameraPerspectiveFovProjection {
            fovy: Rad::from(Deg(45.0)),
            aspect: 3.0 / 2.0,
            near: 5.0,
            far: 25.0,
            near_depth: -10.0,
            far_depth: 20.0,
        };
        let proj = CameraPerspectiveProjection::from(proj);
        let tan_225 = 0.41421356237;
        assert_approx_eq!(proj.left, -7.5 * tan_225);
        assert_approx_eq!(proj.right, 7.5 * tan_225);
        assert_approx_eq!(proj.bottom, -5.0 * tan_225);
        assert_approx_eq!(proj.top, 5.0 * tan_225);
        assert_eq!(proj.near, 5.0);
        assert_eq!(proj.far, 25.0);
        assert_eq!(proj.near_depth, -10.0);
        assert_eq!(proj.far_depth, 20.0);
    }

    #[test]
    fn test_perspective_fov_projection() {
        let proj = CameraPerspectiveFovProjection {
            fovy: Rad::from(Deg(90.0)),
            aspect: 3.0 / 2.0,
            near: 5.0,
            far: 25.0,
            near_depth: -10.0,
            far_depth: 20.0,
        };
        let mat0 = ProjectionMatrix::from(proj);
        let proj = CameraPerspectiveProjection {
            left: -1.5 * 5.0,
            right: 1.5 * 5.0,
            bottom: -1.0 * 5.0,
            top: 1.0 * 5.0,
            near: 5.0,
            far: 25.0,
            near_depth: -10.0,
            far_depth: 20.0,
        };
        let matr = ProjectionMatrix::from(proj);
        assert_approx_eq!(mat0.depth_range.0, matr.depth_range.0);
        assert_approx_eq!(mat0.depth_range.1, matr.depth_range.1);
        assert_approx_eq!(mat0.matrix.0.x, matr.matrix.0.x);
        assert_approx_eq!(mat0.matrix.0.y, matr.matrix.0.y);
        assert_approx_eq!(mat0.matrix.0.z, matr.matrix.0.z);
        assert_approx_eq!(mat0.matrix.0.w, matr.matrix.0.w);
        assert_approx_eq!(mat0.matrix.1.x, matr.matrix.1.x);
        assert_approx_eq!(mat0.matrix.1.y, matr.matrix.1.y);
        assert_approx_eq!(mat0.matrix.1.z, matr.matrix.1.z);
        assert_approx_eq!(mat0.matrix.1.w, matr.matrix.1.w);
        assert_approx_eq!(mat0.matrix.2.x, matr.matrix.2.x);
        assert_approx_eq!(mat0.matrix.2.y, matr.matrix.2.y);
        assert_approx_eq!(mat0.matrix.2.z, matr.matrix.2.z);
        assert_approx_eq!(mat0.matrix.2.w, matr.matrix.2.w);
        assert_approx_eq!(mat0.matrix.3.x, matr.matrix.3.x);
        assert_approx_eq!(mat0.matrix.3.y, matr.matrix.3.y);
        assert_approx_eq!(mat0.matrix.3.z, matr.matrix.3.z);
        assert_approx_eq!(mat0.matrix.3.w, matr.matrix.3.w);
    }

    #[test]
    fn test_orthographic_projection() {
        let proj = CameraOrthographicProjection {
            left: -0.5,
            right: 1.5,
            bottom: 1.0,
            top: 2.0,
            near: 5.0,
            far: 25.0,
            near_depth: 10.0,
            far_depth: 100.0,
        };
        let mat = ProjectionMatrix::from(proj);
        assert_eq!(mat.depth_range.0, 10.0);
        assert_eq!(mat.depth_range.1, 100.0);
        let v = Pos3::from(mat.matrix * Pos3 { x: -0.5, y: 2.0, z: -5.0 });
        assert_approx_eq!(v.x, -1.0);
        assert_approx_eq!(v.y, 1.0);
        assert_approx_eq!(v.z, 10.0);
        let v = Pos3::from(mat.matrix * Pos3 { x: 0.0, y: 1.5, z: -15.0 });
        assert_approx_eq!(v.x, -0.5);
        assert_approx_eq!(v.y, 0.0);
        assert_approx_eq!(v.z, 55.0);
        let v = Pos3::from(mat.matrix * Pos3 { x: 1.0, y: 2.0, z: -25.0 });
        assert_approx_eq!(v.x, 0.5);
        assert_approx_eq!(v.y, 1.0);
        assert_approx_eq!(v.z, 100.0);
    }
}
