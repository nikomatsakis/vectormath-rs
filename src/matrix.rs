#![deny(missing_docs)]
//! Matrix

use float::Float;
use angle::Rad;
use vector::{Vec2, Vec3, Vec4, Pos2, Pos3};
use std::ops::{Add, Sub, Mul, Neg};
use std::convert::From;

/// 2x2 Column-major Matrix
///
/// # Definition
///
/// <div>$$
///   \mathbf{M} = \begin{bmatrix}
///                    m_{0,x} & m_{1,x} \\
///                    m_{0,y} & m_{1,y}
///                \end{bmatrix}
///     = \left[ \begin{array}{c|c} \mathbf{m_0} & \mathbf{m_1} \end{array} \right]
///     \in \mathbb{R}^{2 \times 2} \\
///   \textrm{where } \mathbf{m_0}, \mathbf{m_1} \textrm{ are column vectors } \in \mathbb{R}^2
/// $$</div>
#[derive(Debug, Copy, Clone)]
pub struct Mat2<T>(pub Vec2<T>, pub Vec2<T>);

/// 3x3 Column-major Matrix
///
/// # Definition
///
/// <div>$$
///   \mathbf{M} = \begin{bmatrix}
///                    m_{0,x} & m_{1,x} & m_{2,x} \\
///                    m_{0,y} & m_{1,y} & m_{2,y} \\
///                    m_{0,z} & m_{1,z} & m_{2,z}
///                \end{bmatrix}
///     = \left[ \begin{array}{c|c|c} \mathbf{m_0} & \mathbf{m_1} & \mathbf{m_2} \end{array} \right]
///     \in \mathbb{R}^{3 \times 3} \\
///   \textrm{where } \mathbf{m_0}, \mathbf{m_1}, \mathbf{m_2} \textrm{ are column vectors } \in \mathbb{R}^3
/// $$</div>
#[derive(Debug, Copy, Clone)]
pub struct Mat3<T>(pub Vec3<T>, pub Vec3<T>, pub Vec3<T>);

/// 4x4 Column-major Matrix
///
/// # Definition
///
/// <div>$$
///   \mathbf{M} = \begin{bmatrix}
///                    m_{0,x} & m_{1,x} & m_{2,x} & m_{3,x} \\
///                    m_{0,y} & m_{1,y} & m_{2,y} & m_{3,y} \\
///                    m_{0,z} & m_{1,z} & m_{2,z} & m_{3,z} \\
///                    m_{0,w} & m_{1,w} & m_{2,w} & m_{3,w}
///                \end{bmatrix}
///     = \left[ \begin{array}{c|c|c|c} \mathbf{m_0} & \mathbf{m_1} & \mathbf{m_2} & \mathbf{m_3} \end{array} \right]
///     \in \mathbb{R}^{4 \times 4} \\
///   \textrm{where } \mathbf{m_0}, \mathbf{m_1}, \mathbf{m_2}, \mathbf{m_3} \textrm{ are column vectors } \in \mathbb{R}^4
/// $$</div>
#[derive(Debug, Copy, Clone)]
pub struct Mat4<T>(pub Vec4<T>, pub Vec4<T>, pub Vec4<T>, pub Vec4<T>);

/// 3-D Affine Transform Matrix
///
/// # Definition
///
/// <div>$$
///   \mathbf{M} = \begin{bmatrix}
///                    m_{0,x} & m_{1,x} & m_{2,x} & m_{3,x} \\
///                    m_{0,y} & m_{1,y} & m_{2,y} & m_{3,y} \\
///                    m_{0,z} & m_{1,z} & m_{2,z} & m_{3,z}
///                \end{bmatrix}
///     = \left[ \begin{array}{c|c} \mathbf{R} & \mathbf{t} \end{array} \right]
///     \in \mathbb{R}^{3 \times 4} \\
///   \textrm{where } \mathbf{R} \textrm{ is a 3x3 linear transformation matrix } \in \mathbb{R}^{3 \times 3}
///   \textrm{, and } \mathbf{t} \textrm{ is a 3-D translation vector } \in \mathbb{R}^3
/// $$</div>
#[derive(Debug, Copy, Clone)]
pub struct Tfm3<T>(pub Vec3<T>, pub Vec3<T>, pub Vec3<T>, pub Vec3<T>);

/// Addition
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} + \mathbf{B} =
///       \left[ \begin{array}{c|c}
///           \mathbf{a_0} + \mathbf{b_0} &
///           \mathbf{a_1} + \mathbf{b_1}
///       \end{array} \right]
/// $$</div>
impl<T> Add for Mat2<T> where T: Float {
    type Output = Mat2<T>;
    fn add(self, _rhs: Mat2<T>) -> Mat2<T> {
        Mat2((self.0 + _rhs.0),
             (self.1 + _rhs.1))
    }
}

/// Addition
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} + \mathbf{B} =
///       \left[ \begin{array}{c|c|c}
///           \mathbf{a_0} + \mathbf{b_0} &
///           \mathbf{a_1} + \mathbf{b_1} &
///           \mathbf{a_2} + \mathbf{b_2}
///       \end{array} \right]
/// $$</div>
impl<T> Add for Mat3<T> where T: Float {
    type Output = Mat3<T>;
    fn add(self, _rhs: Mat3<T>) -> Mat3<T> {
        Mat3((self.0 + _rhs.0),
             (self.1 + _rhs.1),
             (self.2 + _rhs.2))
    }
}

/// Addition
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} + \mathbf{B} =
///       \left[ \begin{array}{c|c|c|c}
///           \mathbf{a_0} + \mathbf{b_0} &
///           \mathbf{a_1} + \mathbf{b_1} &
///           \mathbf{a_2} + \mathbf{b_2} &
///           \mathbf{a_3} + \mathbf{b_3}
///       \end{array} \right]
/// $$</div>
impl<T> Add for Mat4<T> where T: Float {
    type Output = Mat4<T>;
    fn add(self, _rhs: Mat4<T>) -> Mat4<T> {
        Mat4((self.0 + _rhs.0),
             (self.1 + _rhs.1),
             (self.2 + _rhs.2),
             (self.3 + _rhs.3))
    }
}

/// Addition
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} + \mathbf{B} =
///       \left[ \begin{array}{c|c|c|c}
///           \mathbf{a_0} + \mathbf{b_0} &
///           \mathbf{a_1} + \mathbf{b_1} &
///           \mathbf{a_2} + \mathbf{b_2} &
///           \mathbf{a_3} + \mathbf{b_3}
///       \end{array} \right]
/// $$</div>
impl<T> Add for Tfm3<T> where T: Float {
    type Output = Tfm3<T>;
    fn add(self, _rhs: Tfm3<T>) -> Tfm3<T> {
        Tfm3((self.0 + _rhs.0),
             (self.1 + _rhs.1),
             (self.2 + _rhs.2),
             (self.3 + _rhs.3))
    }
}

/// Subtraction
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} - \mathbf{B} =
///       \left[ \begin{array}{c|c}
///           \mathbf{a_0} - \mathbf{b_0} &
///           \mathbf{a_1} - \mathbf{b_1}
///       \end{array} \right]
/// $$</div>
impl<T> Sub for Mat2<T> where T: Float {
    type Output = Mat2<T>;
    fn sub(self, _rhs: Mat2<T>) -> Mat2<T> {
        Mat2((self.0 - _rhs.0),
             (self.1 - _rhs.1))
    }
}

/// Subtraction
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} - \mathbf{B} =
///       \left[ \begin{array}{c|c|c}
///           \mathbf{a_0} - \mathbf{b_0} &
///           \mathbf{a_1} - \mathbf{b_1} &
///           \mathbf{a_2} - \mathbf{b_2}
///       \end{array} \right]
/// $$</div>
impl<T> Sub for Mat3<T> where T: Float {
    type Output = Mat3<T>;
    fn sub(self, _rhs: Mat3<T>) -> Mat3<T> {
        Mat3((self.0 - _rhs.0),
             (self.1 - _rhs.1),
             (self.2 - _rhs.2))
    }
}

/// Subtraction
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} - \mathbf{B} =
///       \left[ \begin{array}{c|c|c|c}
///           \mathbf{a_0} - \mathbf{b_0} &
///           \mathbf{a_1} - \mathbf{b_1} &
///           \mathbf{a_2} - \mathbf{b_2} &
///           \mathbf{a_3} - \mathbf{b_3}
///       \end{array} \right]
/// $$</div>
impl<T> Sub for Mat4<T> where T: Float {
    type Output = Mat4<T>;
    fn sub(self, _rhs: Mat4<T>) -> Mat4<T> {
        Mat4((self.0 - _rhs.0),
             (self.1 - _rhs.1),
             (self.2 - _rhs.2),
             (self.3 - _rhs.3))
    }
}

/// Subtraction
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} - \mathbf{B} =
///       \left[ \begin{array}{c|c|c|c}
///           \mathbf{a_0} - \mathbf{b_0} &
///           \mathbf{a_1} - \mathbf{b_1} &
///           \mathbf{a_2} - \mathbf{b_2} &
///           \mathbf{a_3} - \mathbf{b_3}
///       \end{array} \right]
/// $$</div>
impl<T> Sub for Tfm3<T> where T: Float {
    type Output = Tfm3<T>;
    fn sub(self, _rhs: Tfm3<T>) -> Tfm3<T> {
        Tfm3((self.0 - _rhs.0),
             (self.1 - _rhs.1),
             (self.2 - _rhs.2),
             (self.3 - _rhs.3))
    }
}

/// Multiplication by scalar
///
/// # Definition
///
/// <div>$$
///   s \mathbf{A} =
///       \left[ \begin{array}{c|c}
///           s \mathbf{a_0} &
///           s \mathbf{a_1}
///       \end{array} \right], \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Mul<T> for Mat2<T> where T: Float {
    type Output = Mat2<T>;
    fn mul(self, s: T) -> Mat2<T> {
        Mat2((self.0 * s),
             (self.1 * s))
    }
}

/// Multiplication by vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{v}
///       = \left[ \begin{array}{c|c}
///             \mathbf{a_0} & \mathbf{a_1}
///         \end{array} \right]
///         \begin{bmatrix} v_x \\ v_y  \end{bmatrix}
///       = v_x \mathbf{a_0} + v_y \mathbf{a_1}
/// $$</div>
impl<T> Mul<Vec2<T>> for Mat2<T> where T: Float {
    type Output = Vec2<T>;
    fn mul(self, v: Vec2<T>) -> Vec2<T> {
        let v0 = self.0 * v.x;
        let v1 = self.1 * v.y;
        (v0 + v1)
    }
}

/// Multiplication by position
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{p}
///       = \left[ \begin{array}{c|c}
///             \mathbf{a_0} & \mathbf{a_1}
///         \end{array} \right]
///         \begin{bmatrix} p_x \\ p_y \end{bmatrix}
///       = p_x \mathbf{a_0} + p_y \mathbf{a_1}
/// $$</div>
impl<T: Float> Mul<Pos2<T>> for Mat2<T> {
    type Output = Pos2<T>;
    fn mul(self, p: Pos2<T>) -> Pos2<T> {
        let v: Vec2<T> = From::from(p);
        From::from(self * v)
    }
}

/// Multiplication by matrix
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{B}
///       = \left[ \begin{array}{c|c}
///             \mathbf{a_0} & \mathbf{a_1}
///         \end{array} \right]
///         \left[ \begin{array}{c|c}
///             \mathbf{b_0} & \mathbf{b_1}
///         \end{array} \right]
///       = \left[ \begin{array}{c|c}
///             \mathbf{A} \mathbf{b_0} &
///             \mathbf{A} \mathbf{b_1}
///         \end{array} \right]
/// $$</div>
impl<T> Mul for Mat2<T> where T: Float {
    type Output = Mat2<T>;
    fn mul(self, other: Mat2<T>) -> Mat2<T> {
        let v0 = self * other.0;
        let v1 = self * other.1;
        Mat2(v0, v1)
    }
}

/// Multiplication by scalar
///
/// # Definition
///
/// <div>$$
///   s \mathbf{A} =
///       \left[ \begin{array}{c|c|c}
///           s \mathbf{a_0} &
///           s \mathbf{a_1} &
///           s \mathbf{a_2}
///       \end{array} \right], \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Mul<T> for Mat3<T> where T: Float {
    type Output = Mat3<T>;
    fn mul(self, s: T) -> Mat3<T> {
        Mat3((self.0 * s),
             (self.1 * s),
             (self.2 * s))
    }
}

/// Multiplication by vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{v}
///       = \left[ \begin{array}{c|c|c}
///             \mathbf{a_0} &
///             \mathbf{a_1} &
///             \mathbf{a_2}
///         \end{array} \right]
///         \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix}
///       = v_x \mathbf{a_0} + v_y \mathbf{a_1} + v_z \mathbf{a_2}
/// $$</div>
impl<T> Mul<Vec3<T>> for Mat3<T> where T: Float {
    type Output = Vec3<T>;
    fn mul(self, v: Vec3<T>) -> Vec3<T> {
        let v0 = self.0 * v.x;
        let v1 = self.1 * v.y;
        let v2 = self.2 * v.z;
        (v0 + v1 + v2)
    }
}

/// Multiplication by position
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{p}
///       = \left[ \begin{array}{c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2}
///         \end{array} \right]
///         \begin{bmatrix} p_x \\ p_y \\ p_z \end{bmatrix}
///       = p_x \mathbf{a_0} + p_y \mathbf{a_1} + p_z \mathbf{a_2}
/// $$</div>
impl<T: Float> Mul<Pos3<T>> for Mat3<T> {
    type Output = Pos3<T>;
    fn mul(self, p: Pos3<T>) -> Pos3<T> {
        let v: Vec3<T> = From::from(p);
        From::from(self * v)
    }
}

/// Multiplication by matrix
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{B}
///       = \left[ \begin{array}{c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2}
///         \end{array} \right]
///         \left[ \begin{array}{c|c|c}
///             \mathbf{b_0} & \mathbf{b_1} & \mathbf{b_2}
///         \end{array} \right]
///       = \left[ \begin{array}{c|c|c}
///             \mathbf{A} \mathbf{b_0} &
///             \mathbf{A} \mathbf{b_1} &
///             \mathbf{A} \mathbf{b_2}
///         \end{array} \right]
/// $$</div>
impl<T> Mul for Mat3<T> where T: Float {
    type Output = Mat3<T>;
    fn mul(self, other: Mat3<T>) -> Mat3<T> {
        let v0 = self * other.0;
        let v1 = self * other.1;
        let v2 = self * other.2;
        Mat3(v0, v1, v2)
    }
}

/// Multiplication by scalar
///
/// # Definition
///
/// <div>$$
///   s \mathbf{A} =
///       \left[ \begin{array}{c|c|c|c}
///           s \mathbf{a_0} &
///           s \mathbf{a_1} &
///           s \mathbf{a_2} &
///           s \mathbf{a_3}
///       \end{array} \right], \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Mul<T> for Mat4<T> where T: Float {
    type Output = Mat4<T>;
    fn mul(self, s: T) -> Mat4<T> {
        Mat4((self.0 * s),
             (self.1 * s),
             (self.2 * s),
             (self.3 * s))
    }
}

/// Multiplication by vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{v}
///       = \left[ \begin{array}{c|c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2} & \mathbf{a_3}
///         \end{array} \right]
///         \begin{bmatrix} v_x \\ v_y \\ v_z \\ v_w \end{bmatrix}
///       = v_x \mathbf{a_0} + v_y \mathbf{a_1} + v_z \mathbf{a_2} + v_w \mathbf{a_3}
/// $$</div>
impl<T> Mul<Vec4<T>> for Mat4<T> where T: Float {
    type Output = Vec4<T>;
    fn mul(self, v: Vec4<T>) -> Vec4<T> {
        let v0 = self.0 * v.x;
        let v1 = self.1 * v.y;
        let v2 = self.2 * v.z;
        let v3 = self.3 * v.w;
        (v0 + v1 + v2 + v3)
    }
}

/// Multiplication by position
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{p}
///       = \left[ \begin{array}{c|c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2} & \mathbf{a_3}
///         \end{array} \right]
///         \begin{bmatrix} p_x \\ p_y \\ p_z \\ 1 \end{bmatrix}
///       = p_x \mathbf{a_0} + p_y \mathbf{a_1} + p_z \mathbf{a_2} + \mathbf{a_3}
/// $$</div>
impl<T: Float> Mul<Pos3<T>> for Mat4<T> {
    type Output = Vec4<T>;
    fn mul(self, p: Pos3<T>) -> Vec4<T> {
        let v: Vec4<T> = From::from(p);
        (self * v)
    }
}

/// Multiplication by matrix
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{B}
///       = \left[ \begin{array}{c|c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2} & \mathbf{a_3}
///         \end{array} \right]
///         \left[ \begin{array}{c|c|c|c}
///             \mathbf{b_0} & \mathbf{b_1} & \mathbf{b_2} & \mathbf{b_3}
///         \end{array} \right]
///       = \left[ \begin{array}{c|c|c|c}
///             \mathbf{A} \mathbf{b_0} &
///             \mathbf{A} \mathbf{b_1} &
///             \mathbf{A} \mathbf{b_2} &
///             \mathbf{A} \mathbf{b_3}
///         \end{array} \right]
/// $$</div>
impl<T> Mul for Mat4<T> where T: Float {
    type Output = Mat4<T>;
    fn mul(self, other: Mat4<T>) -> Mat4<T> {
        let v0 = self * other.0;
        let v1 = self * other.1;
        let v2 = self * other.2;
        let v3 = self * other.3;
        Mat4(v0, v1, v2, v3)
    }
}

/// Multiplication by scalar
///
/// # Definition
///
/// <div>$$
///   s \mathbf{A} =
///       \left[ \begin{array}{c|c|c|c}
///           s \mathbf{a_0} &
///           s \mathbf{a_1} &
///           s \mathbf{a_2} &
///           s \mathbf{a_3}
///       \end{array} \right], \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Mul<T> for Tfm3<T> where T: Float {
    type Output = Tfm3<T>;
    fn mul(self, s: T) -> Tfm3<T> {
        Tfm3((self.0 * s),
             (self.1 * s),
             (self.2 * s),
             (self.3 * s))
    }
}

/// Multiplication by 3-D vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{v}
///       = \left[ \begin{array}{c|c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2} & \mathbf{a_3}
///         \end{array} \right]
///         \begin{bmatrix} v_x \\ v_y \\ v_z \\ 0 \end{bmatrix}
///       = v_x \mathbf{a_0} + v_y \mathbf{a_1} + v_z \mathbf{a_2}
/// $$</div>
impl<T> Mul<Vec3<T>> for Tfm3<T> where T: Float {
    type Output = Vec3<T>;
    fn mul(self, v: Vec3<T>) -> Vec3<T> {
        let v0 = self.0 * v.x;
        let v1 = self.1 * v.y;
        let v2 = self.2 * v.z;
        (v0 + v1 + v2)
    }
}

/// Multiplication by 4-D vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{v}
///       = \left[ \begin{array}{c|c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2} & \mathbf{a_3}
///         \end{array} \right]
///         \begin{bmatrix} v_x \\ v_y \\ v_z \\ v_w \end{bmatrix}
///       = v_x \mathbf{a_0} + v_y \mathbf{a_1} + v_z \mathbf{a_2} + v_w \mathbf{a_3}
/// $$</div>
impl<T> Mul<Vec4<T>> for Tfm3<T> where T: Float {
    type Output = Vec3<T>;
    fn mul(self, v: Vec4<T>) -> Vec3<T> {
        let v0 = self.0 * v.x;
        let v1 = self.1 * v.y;
        let v2 = self.2 * v.z;
        let v3 = self.3 * v.w;
        (v0 + v1 + v2 + v3)
    }
}

/// Multiplication by 3-D position
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{p}
///       = \left[ \begin{array}{c|c|c|c}
///             \mathbf{a_0} & \mathbf{a_1} & \mathbf{a_2} & \mathbf{a_3}
///         \end{array} \right]
///         \begin{bmatrix} p_x \\ p_y \\ p_z \\ 1 \end{bmatrix}
///       = p_x \mathbf{a_0} + p_y \mathbf{a_1} + p_z \mathbf{a_2} + \mathbf{a_3}
/// $$</div>
impl<T> Mul<Pos3<T>> for Tfm3<T> where T: Float {
    type Output = Pos3<T>;
    fn mul(self, p: Pos3<T>) -> Pos3<T> {
        let v: Vec4<T> = From::from(p);
        Pos3::from(self * v)
    }
}

/// Composition
///
/// # Definition
///
/// <div>$$
///   \mathbf{A} \mathbf{B}
///       = \left[ \begin{array}{c|c}
///             \mathbf{R_A} & \mathbf{t_A}
///         \end{array} \right]
///         \left[ \begin{array}{c|c}
///             \mathbf{R_B} & \mathbf{t_B}
///         \end{array} \right]
///       = \left[ \begin{array}{c|c}
///             \mathbf{R_A} \mathbf{R_B} &
///             \mathbf{t_A} + \mathbf{R_A} \mathbf{t_B}
///         \end{array} \right]
/// $$</div>
impl<T> Mul for Tfm3<T> where T: Float {
    type Output = Tfm3<T>;
    fn mul(self, other: Tfm3<T>) -> Tfm3<T> {
        let v0 = self * other.0;
        let v1 = self * other.1;
        let v2 = self * other.2;
        let v3 = Vec3::from(self * Pos3::from(other.3));
        Tfm3(v0, v1, v2, v3)
    }
}

/// Negation
///
/// # Definition
///
/// <div>$$
///   -\mathbf{A} =
///       \left[ \begin{array}{c|c}
///           - \mathbf{a_0} &
///           - \mathbf{a_1}
///       \end{array} \right]
/// $$</div>
impl<T> Neg for Mat2<T> where T: Float {
    type Output = Mat2<T>;
    fn neg(self) -> Mat2<T> {
        Mat2(-self.0, -self.1)
    }
}

/// Negation
///
/// # Definition
///
/// <div>$$
///   -\mathbf{A} =
///       \left[ \begin{array}{c|c|c}
///           - \mathbf{a_0} &
///           - \mathbf{a_1} &
///           - \mathbf{a_2}
///       \end{array} \right]
/// $$</div>
impl<T> Neg for Mat3<T> where T: Float {
    type Output = Mat3<T>;
    fn neg(self) -> Mat3<T> {
        Mat3(-self.0, -self.1, -self.2)
    }
}

/// Negation
///
/// # Definition
///
/// <div>$$
///   -\mathbf{A} =
///       \left[ \begin{array}{c|c|c|c}
///           - \mathbf{a_0} &
///           - \mathbf{a_1} &
///           - \mathbf{a_2} &
///           - \mathbf{a_3}
///       \end{array} \right]
/// $$</div>
impl<T> Neg for Mat4<T> where T: Float {
    type Output = Mat4<T>;
    fn neg(self) -> Mat4<T> {
        Mat4(-self.0, -self.1, -self.2, -self.3)
    }
}

/// Negation
///
/// # Definition
///
/// <div>$$
///   -\mathbf{A} =
///       \left[ \begin{array}{c|c|c|c}
///           - \mathbf{a_0} &
///           - \mathbf{a_1} &
///           - \mathbf{a_2} &
///           - \mathbf{a_3}
///       \end{array} \right]
/// $$</div>
impl<T> Neg for Tfm3<T> where T: Float {
    type Output = Tfm3<T>;
    fn neg(self) -> Tfm3<T> {
        Tfm3(-self.0, -self.1, -self.2, -self.3)
    }
}

impl<T> Mat2<T> where T: Float {
    /// Transposition
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{M}^T
    ///     = \begin{bmatrix}
    ///           a_{0,x} & a_{1,x} \\
    ///           a_{0,y} & a_{1,y}
    ///       \end{bmatrix}^T
    ///     = \begin{bmatrix}
    ///           a_{0,x} & a_{0,y} \\
    ///           a_{1,x} & a_{1,y}
    ///       \end{bmatrix}
    /// $$</div>
    pub fn transpose(self) -> Mat2<T> {
        let v0 = Vec2 { x: self.0.x, y: self.1.x };
        let v1 = Vec2 { x: self.0.y, y: self.1.y };
        Mat2(v0, v1)
    }
}

impl<T> Mat3<T> where T: Float {
    /// Transposition
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{M}^T
    ///     = \begin{bmatrix}
    ///           a_{0,x} & a_{1,x} & a_{2,x} \\
    ///           a_{0,y} & a_{1,y} & a_{2,y} \\
    ///           a_{0,z} & a_{1,z} & a_{2,z}
    ///       \end{bmatrix}^T
    ///     = \begin{bmatrix}
    ///           a_{0,x} & a_{0,y} & a_{0,z} \\
    ///           a_{1,x} & a_{1,y} & a_{1,z} \\
    ///           a_{2,x} & a_{2,y} & a_{2,z}
    ///       \end{bmatrix}
    /// $$</div>
    pub fn transpose(self) -> Mat3<T> {
        let v0 = Vec3 { x: self.0.x, y: self.1.x, z: self.2.x };
        let v1 = Vec3 { x: self.0.y, y: self.1.y, z: self.2.y };
        let v2 = Vec3 { x: self.0.z, y: self.1.z, z: self.2.z };
        Mat3(v0, v1, v2)
    }
}

impl<T> Mat4<T> where T: Float {
    /// Transposition
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{M}^T
    ///     = \begin{bmatrix}
    ///           a_{0,x} & a_{1,x} & a_{2,x} & a_{3,x} \\
    ///           a_{0,y} & a_{1,y} & a_{2,y} & a_{3,y} \\
    ///           a_{0,z} & a_{1,z} & a_{2,z} & a_{3,z} \\
    ///           a_{0,w} & a_{1,w} & a_{2,w} & a_{3,w}
    ///       \end{bmatrix}^T
    ///     = \begin{bmatrix}
    ///           a_{0,x} & a_{0,y} & a_{0,z} & a_{0,w} \\
    ///           a_{1,x} & a_{1,y} & a_{1,z} & a_{1,w} \\
    ///           a_{2,x} & a_{2,y} & a_{2,z} & a_{2,w} \\
    ///           a_{3,x} & a_{3,y} & a_{3,z} & a_{3,w}
    ///       \end{bmatrix}
    /// $$</div>
    pub fn transpose(self) -> Mat4<T> {
        let v0 = Vec4 { x: self.0.x, y: self.1.x, z: self.2.x, w: self.3.x };
        let v1 = Vec4 { x: self.0.y, y: self.1.y, z: self.2.y, w: self.3.y };
        let v2 = Vec4 { x: self.0.z, y: self.1.z, z: self.2.z, w: self.3.z };
        let v3 = Vec4 { x: self.0.w, y: self.1.w, z: self.2.w, w: self.3.w };
        Mat4(v0, v1, v2, v3)
    }
}

//
// Inversion
//

impl<T> Mat2<T> where T: Float {
    /// Inversion
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{M}^{-1}
    ///       = \begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1}
    ///       = \frac{1}{a d - b c} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
    /// $$</div>
    pub fn inverse(self) -> Mat2<T> {
        let c0 = Vec2 { x: self.1.y, y: -self.0.y };
        let c1 = Vec2 { x: -self.1.x, y: self.0.x };
        let invd = (self.0.x * self.1.y - self.0.y * self.1.x).recip();
        Mat2(c0 * invd, c1 * invd)
    }
}

impl<T> Mat3<T> where T: Float {
    /// Inverse of the matrix
    pub fn inverse(self) -> Mat3<T> {
        let tmp0 = self.1.cross(self.2);
        let tmp1 = self.2.cross(self.0);
        let tmp2 = self.0.cross(self.1);
        let detinv = self.2.dot(tmp2).recip();
        let v0 = Vec3 { x: tmp0.x, y: tmp1.x, z: tmp2.x };
        let v1 = Vec3 { x: tmp0.y, y: tmp1.y, z: tmp2.y };
        let v2 = Vec3 { x: tmp0.z, y: tmp1.z, z: tmp2.z };
        Mat3(v0 * detinv, v1 * detinv, v2 * detinv)
    }
}

impl<T> Mat4<T> where T: Float {
    /// Inverse of the matrix
    pub fn inverse(self) -> Mat4<T> {
        let m = &self;
        let tmp0 = m.2.z * m.0.w - m.0.z * m.2.w;
        let tmp1 = m.3.z * m.1.w - m.1.z * m.3.w;
        let tmp2 = m.0.y * m.2.z - m.2.y * m.0.z;
        let tmp3 = m.1.y * m.3.z - m.3.y * m.1.z;
        let tmp4 = m.2.y * m.0.w - m.0.y * m.2.w;
        let tmp5 = m.3.y * m.1.w - m.1.y * m.3.w;
        let r0 = Vec4 { x: m.2.y * tmp1 - m.2.w * tmp3 - m.2.z * tmp5,
                        y: m.3.y * tmp0 - m.3.w * tmp2 - m.3.z * tmp4,
                        z: m.0.w * tmp3 + m.0.z * tmp5 - m.0.y * tmp1,
                        w: m.1.w * tmp2 + m.1.z * tmp4 - m.1.y * tmp0 };
        let r1 = Vec4 { x: m.2.x * tmp1, y: m.3.x * tmp0, z: m.0.x * tmp1, w: m.1.x * tmp0 };
        let r3 = Vec4 { x: m.2.x * tmp3, y: m.3.x * tmp2, z: m.0.x * tmp3, w: m.1.x * tmp2 };
        let r2 = Vec4 { x: m.2.x * tmp5, y: m.3.x * tmp4, z: m.0.x * tmp5, w: m.1.x * tmp4 };
        let detinv = r0.dot(Vec4 { x: m.0.x, y: m.1.x, z: m.2.x, w: m.3.x }).recip();
        let tmp0 = m.2.x * m.0.y - m.0.x * m.2.y;
        let tmp1 = m.3.x * m.1.y - m.1.x * m.3.y;
        let tmp2 = m.2.x * m.0.w - m.0.x * m.2.w;
        let tmp3 = m.3.x * m.1.w - m.1.x * m.3.w;
        let tmp4 = m.2.x * m.0.z - m.0.x * m.2.z;
        let tmp5 = m.3.x * m.1.z - m.1.x * m.3.z;
        let r2 = Vec4 { x: m.2.w * tmp1 - m.2.y * tmp3 + r2.x,
                        y: m.3.w * tmp0 - m.3.y * tmp2 + r2.y,
                        z: m.0.y * tmp3 - m.0.w * tmp1 - r2.z,
                        w: m.1.y * tmp2 - m.1.w * tmp0 - r2.w };
        let r3 = Vec4 { x: m.2.y * tmp5 - m.2.z * tmp1 + r3.x,
                        y: m.3.y * tmp4 - m.3.z * tmp0 + r3.y,
                        z: m.0.z * tmp1 - m.0.y * tmp5 - r3.z,
                        w: m.1.z * tmp0 - m.1.y * tmp4 - r3.w };
        let r1 = Vec4 { x: m.2.z * tmp3 - m.2.w * tmp5 - r1.x,
                        y: m.3.z * tmp2 - m.3.w * tmp4 - r1.y,
                        z: m.0.w * tmp5 - m.0.z * tmp3 + r1.z,
                        w: m.1.w * tmp4 - m.1.z * tmp2 + r1.w };
        Mat4(r0 * detinv, r1 * detinv, r2 * detinv, r3 * detinv)
    }
}

impl<T> Tfm3<T> where T: Float {
    /// Inversion
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{T}^{-1}
    ///       = \left[ \begin{array}{c|c} \mathbf{R} & \mathbf{t} \end{array}^{-1} \right]
    ///       = \left[ \begin{array}{c|c} \mathbf{R}^{-1} & - \mathbf{R}^{-1} \mathbf{t} \end{array} \right]
    /// $$</div>
    pub fn inverse(self) -> Tfm3<T> {
        let m1 = Mat3(self.0, self.1, self.2);
        let m2 = m1.inverse();
        let t2 = -(m2 * self.3);
        Tfm3(m2.0, m2.1, m2.2, t2)
    }
}

//
// Conversion
//

impl<T> From<Mat4<T>> for Mat3<T> where T: Float {
    fn from(m: Mat4<T>) -> Mat3<T> {
        Mat3(<Vec3<T>>::from(m.0), <Vec3<T>>::from(m.1), <Vec3<T>>::from(m.2))
    }
}

impl<T> From<Tfm3<T>> for Mat3<T> where T: Float {
    fn from(m: Tfm3<T>) -> Mat3<T> {
        Mat3(m.0, m.1, m.2)
    }
}

impl<T> From<Mat3<T>> for Mat4<T> where T: Float {
    fn from(m: Mat3<T>) -> Mat4<T> {
        Mat4(<Vec4<T>>::from(m.0),
             <Vec4<T>>::from(m.1),
             <Vec4<T>>::from(m.2),
             Vec4 { x: T::zero(), y: T::zero(), z: T::zero(), w: T::one() })
    }
}

impl<T> From<Tfm3<T>> for Mat4<T> where T: Float {
    fn from(m: Tfm3<T>) -> Mat4<T> {
        Mat4(<Vec4<T>>::from(m.0),
             <Vec4<T>>::from(m.1),
             <Vec4<T>>::from(m.2),
             <Vec4<T>>::from(<Pos3<T>>::from(m.3)))
    }
}

impl<T> From<Mat3<T>> for Tfm3<T> where T: Float {
    fn from(m: Mat3<T>) -> Tfm3<T> {
        Tfm3(m.0, m.1, m.2, Vec3{ x: T::zero(), y: T::zero(), z: T::zero() })
    }
}

/// Constructors

impl<T: Float> Mat2<T> {
    /// Construct an identity matrix
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{M} = \begin{bmatrix}
    ///                    1 & 0 \\
    ///                    0 & 1
    ///                \end{bmatrix}
    /// $$</div>
    pub fn identity() -> Mat2<T> {
        Mat2(Vec2 { x: T::one(), y: T::zero() },
             Vec2 { x: T::zero(), y: T::one() })
    }
}

impl<T: Float> Mat3<T> {
    /// Construct an identity matrix
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{M} = \begin{bmatrix}
    ///                    1 & 0 & 0 \\
    ///                    0 & 1 & 0 \\
    ///                    0 & 0 & 1
    ///                \end{bmatrix}
    /// $$</div>
    pub fn identity() -> Mat3<T> {
        Mat3(Vec3 { x: T::one(), y: T::zero(), z: T::zero() },
             Vec3 { x: T::zero(), y: T::one(), z: T::zero() },
             Vec3 { x: T::zero(), y: T::zero(), z: T::one() })
    }

    /// Construct a rotation matrix
    pub fn rotation_angle_axis(angle: Rad<T>, axis: Vec3<T>) -> Mat3<T> {
        let (x, y, z) = (axis.x, axis.y, axis.z);
        let (s, c) = angle.0.sincos();
        let t = T::one() - c;
        Mat3(Vec3 { x: t * x * x + c, y: t * x * y + z * s, z: t * x * z - y * s },
             Vec3 { x: t * x * y - z * s, y: t * y * y + c, z: t * y * z + x * s },
             Vec3 { x: t * x * z + y * s, y: t * y * z - x * s, z: t * z * z + c })
    }
}

impl<T: Float> Mat4<T> {
    /// Construct an identity matrix
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{M} = \begin{bmatrix}
    ///                    1 & 0 & 0 & 0 \\
    ///                    0 & 1 & 0 & 0 \\
    ///                    0 & 0 & 1 & 0 \\
    ///                    0 & 0 & 0 & 1
    ///                \end{bmatrix}
    /// $$</div>
    pub fn identity() -> Mat4<T> {
        Mat4(Vec4 { x: T::one(), y: T::zero(), z: T::zero(), w: T::zero() },
             Vec4 { x: T::zero(), y: T::one(), z: T::zero(), w: T::zero() },
             Vec4 { x: T::zero(), y: T::zero(), z: T::one(), w: T::zero() },
             Vec4 { x: T::zero(), y: T::zero(), z: T::zero(), w: T::one() })
    }
}

impl<T: Float> Tfm3<T> {
    /// Construct an identity matrix
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{T} = \begin{bmatrix}
    ///                    1 & 0 & 0 & 0 \\
    ///                    0 & 1 & 0 & 0 \\
    ///                    0 & 0 & 1 & 0
    ///                \end{bmatrix}
    /// $$</div>
    pub fn identity() -> Tfm3<T> {
        Tfm3(Vec3 { x: T::one(), y: T::zero(), z: T::zero() },
             Vec3 { x: T::zero(), y: T::one(), z: T::zero() },
             Vec3 { x: T::zero(), y: T::zero(), z: T::one() },
             Vec3 { x: T::zero(), y: T::zero(), z: T::zero() })
    }
}

#[cfg(test)]
mod tests_mat2 {
    use super::Mat2;
    use vector::Vec2;

    #[test]
    fn test_add() {
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let m2 = Mat2(Vec2 { x: 9.0, y: 8.1 },
                      Vec2 { x: 7.2, y: 6.3 });
        let m3 = m1 + m2;
        assert_approx_eq!(m3.0.x, 10.0);
        assert_approx_eq!(m3.0.y, 10.1);
        assert_approx_eq!(m3.1.x, 10.2);
        assert_approx_eq!(m3.1.y, 10.3);
    }

    #[test]
    fn test_sub(){
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let m2 = Mat2(Vec2 { x: 9.0, y: 8.1 },
                      Vec2 { x: 7.2, y: 6.3 });
        let m3 = m1 - m2;
        assert_approx_eq!(m3.0.x, -8.0);
        assert_approx_eq!(m3.0.y, -6.1);
        assert_approx_eq!(m3.1.x, -4.2);
        assert_approx_eq!(m3.1.y, -2.3);
    }

    #[test]
    fn test_neg() {
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let m2 = -m1;
        assert_approx_eq!(m2.0.x, -1.0);
        assert_approx_eq!(m2.0.y, -2.0);
        assert_approx_eq!(m2.1.x, -3.0);
        assert_approx_eq!(m2.1.y, -4.0);
    }

    #[test]
    fn test_mul_scalar() {
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let m2 = m1 * 0.5;
        assert_eq!(m2.0.x, 0.5);
        assert_eq!(m2.0.y, 1.0);
        assert_eq!(m2.1.x, 1.5);
        assert_eq!(m2.1.y, 2.0);
    }

    #[test]
    fn test_mul_vec2() {
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let v1 = Vec2 { x: 1.0, y: 2.0 };
        let v2: Vec2<_> = m1 * v1;
        assert_eq!(v2.x, 7.0);
        assert_eq!(v2.y, 10.0);
    }

    #[test]
    fn test_mul_pos2() {
        use vector::Pos2;
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let v1 = Pos2 { x: 1.0, y: 2.0 };
        let v2: Pos2<_> = m1 * v1;
        assert_eq!(v2.x, 7.0);
        assert_eq!(v2.y, 10.0);
    }

    #[test]
    fn test_mul_mat() {
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let m2 = Mat2(Vec2 { x: -2.0, y: 1.0 },
                      Vec2 { x: 0.5, y: 1.25 });
        let m3 = m1 * m2;
        assert_eq!(m3.0.x, 1.0);
        assert_eq!(m3.0.y, 0.0);
        assert_eq!(m3.1.x, 4.25);
        assert_eq!(m3.1.y, 6.0);
    }

    #[test]
    fn test_transpose() {
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let m2 = m1.transpose();
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 3.0);
        assert_eq!(m2.1.x, 2.0);
        assert_eq!(m2.1.y, 4.0);
    }

    #[test]
    fn test_inverse() {
        let m1 = Mat2(Vec2 { x: 1.0, y: 2.0 },
                      Vec2 { x: 3.0, y: 4.0 });
        let m2 = m1.inverse();
        assert_eq!(m2.0.x, -2.0);
        assert_eq!(m2.0.y, 1.0);
        assert_eq!(m2.1.x, 1.5);
        assert_eq!(m2.1.y, -0.5);
        let m3 = m1 * m2;
        assert_eq!(m3.0.x, 1.0);
        assert_eq!(m3.0.y, 0.0);
        assert_eq!(m3.1.x, 0.0);
        assert_eq!(m3.1.y, 1.0);
    }

    #[test]
    fn test_identity() {
        let m1: Mat2<f64> = Mat2::identity();
        assert_eq!(m1.0.x, 1.0);
        assert_eq!(m1.0.y, 0.0);
        assert_eq!(m1.1.x, 0.0);
        assert_eq!(m1.1.y, 1.0);
    }
}

#[cfg(test)]
mod tests_mat3 {
    use super::Mat3;
    use vector::Vec3;

    #[test]
    fn test_add() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = Mat3(Vec3 { x: 9.0, y: 8.1, z: 7.2 },
                      Vec3 { x: 6.3, y: 5.4, z: 4.5 },
                      Vec3 { x: 3.6, y: 2.7, z: 1.8 });
        let m3 = m1 + m2;
        assert_approx_eq!(m3.0.x, 10.0);
        assert_approx_eq!(m3.0.y, 10.1);
        assert_approx_eq!(m3.0.z, 10.2);
        assert_approx_eq!(m3.1.x, 10.3);
        assert_approx_eq!(m3.1.y, 10.4);
        assert_approx_eq!(m3.1.z, 10.5);
        assert_approx_eq!(m3.2.x, 10.6);
        assert_approx_eq!(m3.2.y, 10.7);
        assert_approx_eq!(m3.2.z, 10.8);
    }

    #[test]
    fn test_sub() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = Mat3(Vec3 { x: 9.0, y: 8.1, z: 7.2 },
                      Vec3 { x: 6.3, y: 5.4, z: 4.5 },
                      Vec3 { x: 3.6, y: 2.7, z: 1.8 });
        let m3 = m1 - m2;
        assert_approx_eq!(m3.0.x, -8.0);
        assert_approx_eq!(m3.0.y, -6.1);
        assert_approx_eq!(m3.0.z, -4.2);
        assert_approx_eq!(m3.1.x, -2.3);
        assert_approx_eq!(m3.1.y, -0.4);
        assert_approx_eq!(m3.1.z, 1.5);
        assert_approx_eq!(m3.2.x, 3.4);
        assert_approx_eq!(m3.2.y, 5.3);
        assert_approx_eq!(m3.2.z, 7.2);
    }

    #[test]
    fn test_mul_scalar() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = m1 * 0.25;
        assert_eq!(m2.0.x, 0.25);
        assert_eq!(m2.0.y, 0.5);
        assert_eq!(m2.0.z, 0.75);
        assert_eq!(m2.1.x, 1.0);
        assert_eq!(m2.1.y, 1.25);
        assert_eq!(m2.1.z, 1.5);
        assert_eq!(m2.2.x, 1.75);
        assert_eq!(m2.2.y, 2.0);
        assert_eq!(m2.2.z, 2.25);
    }

    #[test]
    fn test_mul_vec3() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2: Vec3<_> = m1 * v1;
        assert_eq!(v2.x, 30.0);
        assert_eq!(v2.y, 36.0);
        assert_eq!(v2.z, 42.0);
    }

    #[test]
    fn test_mul_pos3() {
        use vector::Pos3;
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let v1 = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2: Pos3<_> = m1 * v1;
        assert_eq!(v2.x, 30.0);
        assert_eq!(v2.y, 36.0);
        assert_eq!(v2.z, 42.0);
    }

    #[test]
    fn test_mul_mat() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = Mat3(Vec3 { x: 9.0, y: 8.0, z: 7.0 },
                      Vec3 { x: 6.0, y: 5.0, z: 4.0 },
                      Vec3 { x: 3.0, y: 2.0, z: 1.0 });
        let m3 = m1 * m2;
        assert_eq!(m3.0.x, 90.0);
        assert_eq!(m3.0.y, 114.0);
        assert_eq!(m3.0.z, 138.0);
        assert_eq!(m3.1.x, 54.0);
        assert_eq!(m3.1.y, 69.0);
        assert_eq!(m3.1.z, 84.0);
        assert_eq!(m3.2.x, 18.0);
        assert_eq!(m3.2.y, 24.0);
        assert_eq!(m3.2.z, 30.0);
    }

    #[test]
    fn test_neg() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = -m1;
        assert_eq!(m2.0.x, -1.0);
        assert_eq!(m2.0.y, -2.0);
        assert_eq!(m2.0.z, -3.0);
        assert_eq!(m2.1.x, -4.0);
        assert_eq!(m2.1.y, -5.0);
        assert_eq!(m2.1.z, -6.0);
        assert_eq!(m2.2.x, -7.0);
        assert_eq!(m2.2.y, -8.0);
        assert_eq!(m2.2.z, -9.0);
    }

    #[test]
    fn test_transpose() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = m1.transpose();
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 4.0);
        assert_eq!(m2.0.z, 7.0);
        assert_eq!(m2.1.x, 2.0);
        assert_eq!(m2.1.y, 5.0);
        assert_eq!(m2.1.z, 8.0);
        assert_eq!(m2.2.x, 3.0);
        assert_eq!(m2.2.y, 6.0);
        assert_eq!(m2.2.z, 9.0);
    }

    #[test]
    fn test_inverse() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 0.0, z: 1.0 },
                      Vec3 { x: 2.0, y: 4.0, z: 0.0 },
                      Vec3 { x: 3.0, y: 5.0, z: 6.0 });
        let m2 = m1.inverse();
        assert_approx_eq!(m2.0.x, 12.0 / 11.0);
        assert_approx_eq!(m2.0.y, 5.0 / 22.0);
        assert_approx_eq!(m2.0.z, -2.0 / 11.0);
        assert_approx_eq!(m2.1.x, -6.0 / 11.0);
        assert_approx_eq!(m2.1.y, 3.0 / 22.0);
        assert_approx_eq!(m2.1.z, 1.0 / 11.0);
        assert_approx_eq!(m2.2.x, -1.0 / 11.0);
        assert_approx_eq!(m2.2.y, -5.0 / 22.0);
        assert_approx_eq!(m2.2.z, 2.0 / 11.0);
    }

    #[test]
    fn test_identity() {
        let m: Mat3<f64> = Mat3::identity();
        assert_eq!(m.0.x, 1.0);
        assert_eq!(m.0.y, 0.0);
        assert_eq!(m.0.z, 0.0);
        assert_eq!(m.1.x, 0.0);
        assert_eq!(m.1.y, 1.0);
        assert_eq!(m.1.z, 0.0);
        assert_eq!(m.2.x, 0.0);
        assert_eq!(m.2.y, 0.0);
        assert_eq!(m.2.z, 1.0);
    }

    #[test]
    fn test_rotation_angle_axis() {
        use angle::{Rad, Deg};
        let m = Mat3::rotation_angle_axis(Rad::from(Deg(30.0)), Vec3::x_axis());
        let p = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v = m * p;
        assert_approx_eq!(v.x, 1.0);
        assert_approx_eq!(v.y, 0.232050808);
        assert_approx_eq!(v.z, 3.598076211);
        let m = Mat3::rotation_angle_axis(Rad::from(Deg(30.0)), Vec3::y_axis());
        let p = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v = m * p;
        assert_approx_eq!(v.x, 2.366025404);
        assert_approx_eq!(v.y, 2.0);
        assert_approx_eq!(v.z, 2.098076211);
        let m = Mat3::rotation_angle_axis(Rad::from(Deg(30.0)), Vec3::z_axis());
        let p = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v = m * p;
        assert_approx_eq!(v.x, -0.133974596);
        assert_approx_eq!(v.y, 2.232050808);
        assert_approx_eq!(v.z, 3.0);
    }
}

#[cfg(test)]
mod tests_mat4 {
    use super::Mat4;
    use vector::Vec4;

    #[test]
    fn test_add() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let m2 = Mat4(Vec4 { x: 15.1, y: 14.2, z: 13.3, w: 12.4 },
                      Vec4 { x: 11.5, y: 10.6, z: 9.7, w: 8.8 },
                      Vec4 { x: 7.9, y: 6.0, z: 5.1, w: 4.2 },
                      Vec4 { x: 3.3, y: 2.4, z: 1.5, w: 0.6 });
        let m3 = m1 + m2;
        assert_approx_eq!(m3.0.x, 16.1);
        assert_approx_eq!(m3.0.y, 16.2);
        assert_approx_eq!(m3.0.z, 16.3);
        assert_approx_eq!(m3.0.w, 16.4);
        assert_approx_eq!(m3.1.x, 16.5);
        assert_approx_eq!(m3.1.y, 16.6);
        assert_approx_eq!(m3.1.z, 16.7);
        assert_approx_eq!(m3.1.w, 16.8);
        assert_approx_eq!(m3.2.x, 16.9);
        assert_approx_eq!(m3.2.y, 16.0);
        assert_approx_eq!(m3.2.z, 16.1);
        assert_approx_eq!(m3.2.w, 16.2);
        assert_approx_eq!(m3.3.x, 16.3);
        assert_approx_eq!(m3.3.y, 16.4);
        assert_approx_eq!(m3.3.z, 16.5);
        assert_approx_eq!(m3.3.w, 16.6);
    }

    #[test]
    fn test_sub() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let m2 = Mat4(Vec4 { x: 15.1, y: 14.2, z: 13.3, w: 12.4 },
                      Vec4 { x: 11.5, y: 10.6, z: 9.7, w: 8.8 },
                      Vec4 { x: 7.9, y: 6.0, z: 5.1, w: 4.2 },
                      Vec4 { x: 3.3, y: 2.4, z: 1.5, w: 0.6 });
        let m3 = m1 - m2;
        assert_approx_eq!(m3.0.x, -14.1);
        assert_approx_eq!(m3.0.y, -12.2);
        assert_approx_eq!(m3.0.z, -10.3);
        assert_approx_eq!(m3.0.w, -8.4);
        assert_approx_eq!(m3.1.x, -6.5);
        assert_approx_eq!(m3.1.y, -4.6);
        assert_approx_eq!(m3.1.z, -2.7);
        assert_approx_eq!(m3.1.w, -0.8);
        assert_approx_eq!(m3.2.x, 1.1);
        assert_approx_eq!(m3.2.y, 4.0);
        assert_approx_eq!(m3.2.z, 5.9);
        assert_approx_eq!(m3.2.w, 7.8);
        assert_approx_eq!(m3.3.x, 9.7);
        assert_approx_eq!(m3.3.y, 11.6);
        assert_approx_eq!(m3.3.z, 13.5);
        assert_approx_eq!(m3.3.w, 15.4);
    }

    #[test]
    fn test_mul_scalar() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let m2 = m1 * 0.25;
        assert_eq!(m2.0.x, 0.25);
        assert_eq!(m2.0.y, 0.5);
        assert_eq!(m2.0.z, 0.75);
        assert_eq!(m2.0.w, 1.0);
        assert_eq!(m2.1.x, 1.25);
        assert_eq!(m2.1.y, 1.5);
        assert_eq!(m2.1.z, 1.75);
        assert_eq!(m2.1.w, 2.0);
        assert_eq!(m2.2.x, 2.25);
        assert_eq!(m2.2.y, 2.5);
        assert_eq!(m2.2.z, 2.75);
        assert_eq!(m2.2.w, 3.0);
        assert_eq!(m2.3.x, 3.25);
        assert_eq!(m2.3.y, 3.5);
        assert_eq!(m2.3.z, 3.75);
        assert_eq!(m2.3.w, 4.0);
    }

    #[test]
    fn test_mul_vec4() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v2 = m1 * v1;
        assert_eq!(v2.x, 90.0);
        assert_eq!(v2.y, 100.0);
        assert_eq!(v2.z, 110.0);
        assert_eq!(v2.w, 120.0);
    }

    #[test]
    fn test_mul_pos3() {
        use vector::Pos3;
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let v1 = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2: Vec4<_> = m1 * v1;
        assert_eq!(v2.x, 51.0);
        assert_eq!(v2.y, 58.0);
        assert_eq!(v2.z, 65.0);
        assert_eq!(v2.w, 72.0);
    }

    #[test]
    fn test_mul_mat() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let m2 = Mat4(Vec4 { x: 16.0, y: 15.0, z: 14.0, w: 13.0 },
                      Vec4 { x: 12.0, y: 11.0, z: 10.0, w: 9.0 },
                      Vec4 { x: 8.0, y: 7.0, z: 6.0, w: 5.0 },
                      Vec4 { x: 4.0, y: 3.0, z: 2.0, w: 1.0 });
        let m3 = m1 * m2;
        assert_eq!(m3.0.x, 386.0);
        assert_eq!(m3.0.y, 444.0);
        assert_eq!(m3.0.z, 502.0);
        assert_eq!(m3.0.w, 560.0);
        assert_eq!(m3.1.x, 274.0);
        assert_eq!(m3.1.y, 316.0);
        assert_eq!(m3.1.z, 358.0);
        assert_eq!(m3.1.w, 400.0);
        assert_eq!(m3.2.x, 162.0);
        assert_eq!(m3.2.y, 188.0);
        assert_eq!(m3.2.z, 214.0);
        assert_eq!(m3.2.w, 240.0);
        assert_eq!(m3.3.x, 50.0);
        assert_eq!(m3.3.y, 60.0);
        assert_eq!(m3.3.z, 70.0);
        assert_eq!(m3.3.w, 80.0);
    }

    #[test]
    fn test_neg() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let m2 = -m1;
        assert_eq!(m2.0.x, -1.0);
        assert_eq!(m2.0.y, -2.0);
        assert_eq!(m2.0.z, -3.0);
        assert_eq!(m2.0.w, -4.0);
        assert_eq!(m2.1.x, -5.0);
        assert_eq!(m2.1.y, -6.0);
        assert_eq!(m2.1.z, -7.0);
        assert_eq!(m2.1.w, -8.0);
        assert_eq!(m2.2.x, -9.0);
        assert_eq!(m2.2.y, -10.0);
        assert_eq!(m2.2.z, -11.0);
        assert_eq!(m2.2.w, -12.0);
        assert_eq!(m2.3.x, -13.0);
        assert_eq!(m2.3.y, -14.0);
        assert_eq!(m2.3.z, -15.0);
        assert_eq!(m2.3.w, -16.0);
    }

    #[test]
    fn test_transpose() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let m2 = m1.transpose();
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 5.0);
        assert_eq!(m2.0.z, 9.0);
        assert_eq!(m2.0.w, 13.0);
        assert_eq!(m2.1.x, 2.0);
        assert_eq!(m2.1.y, 6.0);
        assert_eq!(m2.1.z, 10.0);
        assert_eq!(m2.1.w, 14.0);
        assert_eq!(m2.2.x, 3.0);
        assert_eq!(m2.2.y, 7.0);
        assert_eq!(m2.2.z, 11.0);
        assert_eq!(m2.2.w, 15.0);
        assert_eq!(m2.3.x, 4.0);
        assert_eq!(m2.3.y, 8.0);
        assert_eq!(m2.3.z, 12.0);
        assert_eq!(m2.3.w, 16.0);
    }

    #[test]
    fn test_inverse() {
        let m1 = Mat4(Vec4 { x: 2.0, y: 10.0, z: 0.0, w: 5.0 },
                      Vec4 { x: 1.0, y: -1.0, z: 1.0, w: 0.0 },
                      Vec4 { x: 4.0, y: 7.0, z: 2.0, w: -1.0 },
                      Vec4 { x: 5.0, y: 4.0, z: 3.0, w: 3.0 });
        let m2 = m1.inverse();
        assert_eq!(m2.0.x, -34.0 / 32.0);
        assert_eq!(m2.0.y, -155.0 / 32.0);
        assert_eq!(m2.0.z, -5.0 / 32.0);
        assert_eq!(m2.0.w, 55.0 / 32.0);
        assert_eq!(m2.1.x, 8.0 / 32.0);
        assert_eq!(m2.1.y, 28.0 / 32.0);
        assert_eq!(m2.1.z, 4.0 / 32.0);
        assert_eq!(m2.1.w, -12.0 / 32.0);
        assert_eq!(m2.2.x, 42.0 / 32.0);
        assert_eq!(m2.2.y, 215.0 / 32.0);
        assert_eq!(m2.2.z, 9.0 / 32.0);
        assert_eq!(m2.2.w, -67.0 / 32.0);
        assert_eq!(m2.3.x, 4.0 / 32.0);
        assert_eq!(m2.3.y, 6.0 / 32.0);
        assert_eq!(m2.3.z, -6.0 / 32.0);
        assert_eq!(m2.3.w, 2.0 / 32.0);
    }

    #[test]
    fn test_identity() {
        let m: Mat4<f64> = Mat4::identity();
        assert_eq!(m.0.x, 1.0);
        assert_eq!(m.0.y, 0.0);
        assert_eq!(m.0.z, 0.0);
        assert_eq!(m.0.w, 0.0);
        assert_eq!(m.1.x, 0.0);
        assert_eq!(m.1.y, 1.0);
        assert_eq!(m.1.z, 0.0);
        assert_eq!(m.1.w, 0.0);
        assert_eq!(m.2.x, 0.0);
        assert_eq!(m.2.y, 0.0);
        assert_eq!(m.2.z, 1.0);
        assert_eq!(m.2.w, 0.0);
        assert_eq!(m.3.x, 0.0);
        assert_eq!(m.3.y, 0.0);
        assert_eq!(m.3.z, 0.0);
        assert_eq!(m.3.w, 1.0);
    }
}

#[cfg(test)]
mod tests_tfm3 {
    use super::Tfm3;
    use vector::Vec3;

    #[test]
    fn test_add() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let m2 = Tfm3(Vec3 { x: 11.1, y: 10.2, z: 9.3 },
                      Vec3 { x: 8.4, y: 7.5, z: 6.6 },
                      Vec3 { x: 5.7, y: 4.8, z: 3.9 },
                      Vec3 { x: 2.0, y: 1.1, z: 0.2 });
        let m3 = m1 + m2;
        assert_approx_eq!(m3.0.x, 12.1);
        assert_approx_eq!(m3.0.y, 12.2);
        assert_approx_eq!(m3.0.z, 12.3);
        assert_approx_eq!(m3.1.x, 12.4);
        assert_approx_eq!(m3.1.y, 12.5);
        assert_approx_eq!(m3.1.z, 12.6);
        assert_approx_eq!(m3.2.x, 12.7);
        assert_approx_eq!(m3.2.y, 12.8);
        assert_approx_eq!(m3.2.z, 12.9);
        assert_approx_eq!(m3.3.x, 12.0);
        assert_approx_eq!(m3.3.y, 12.1);
        assert_approx_eq!(m3.3.z, 12.2);
    }

    #[test]
    fn test_sub() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let m2 = Tfm3(Vec3 { x: 11.1, y: 10.2, z: 9.3 },
                      Vec3 { x: 8.4, y: 7.5, z: 6.6 },
                      Vec3 { x: 5.7, y: 4.8, z: 3.9 },
                      Vec3 { x: 2.0, y: 1.1, z: 0.2 });
        let m3 = m1 - m2;
        assert_approx_eq!(m3.0.x, -10.1);
        assert_approx_eq!(m3.0.y, -8.2);
        assert_approx_eq!(m3.0.z, -6.3);
        assert_approx_eq!(m3.1.x, -4.4);
        assert_approx_eq!(m3.1.y, -2.5);
        assert_approx_eq!(m3.1.z, -0.6);
        assert_approx_eq!(m3.2.x, 1.3);
        assert_approx_eq!(m3.2.y, 3.2);
        assert_approx_eq!(m3.2.z, 5.1);
        assert_approx_eq!(m3.3.x, 8.0);
        assert_approx_eq!(m3.3.y, 9.9);
        assert_approx_eq!(m3.3.z, 11.8);
    }

    #[test]
    fn test_mul_scalar() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let m2 = m1 * 0.25;
        assert_eq!(m2.0.x, 0.25);
        assert_eq!(m2.0.y, 0.5);
        assert_eq!(m2.0.z, 0.75);
        assert_eq!(m2.1.x, 1.0);
        assert_eq!(m2.1.y, 1.25);
        assert_eq!(m2.1.z, 1.5);
        assert_eq!(m2.2.x, 1.75);
        assert_eq!(m2.2.y, 2.0);
        assert_eq!(m2.2.z, 2.25);
        assert_eq!(m2.3.x, 2.5);
        assert_eq!(m2.3.y, 2.75);
        assert_eq!(m2.3.z, 3.0);
    }

    #[test]
    fn test_mul_vec3() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };;
        let v2: Vec3<_> = m1 * v1;
        assert_eq!(v2.x, 30.0);
        assert_eq!(v2.y, 36.0);
        assert_eq!(v2.z, 42.0);
    }

    #[test]
    fn test_mul_vec4() {
        use vector::Vec4;
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v2: Vec3<_> = m1 * v1;
        assert_eq!(v2.x, 70.0);
        assert_eq!(v2.y, 80.0);
        assert_eq!(v2.z, 90.0);
    }

    #[test]
    fn test_mul_pos3() {
        use vector::Pos3;
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let v1 = Pos3 { x: 1.0, y: 2.0, z: 3.0 };;
        let v2: Pos3<_> = m1 * v1;
        assert_eq!(v2.x, 40.0);
        assert_eq!(v2.y, 47.0);
        assert_eq!(v2.z, 54.0);
    }

    #[test]
    fn test_mul_mat() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let m2 = Tfm3(Vec3 { x: -1.0, y: 3.0, z: 4.0 },
                      Vec3 { x: 10.0, y: -5.0, z: 1.0 },
                      Vec3 { x: 3.0, y: 0.0, z: 7.0 },
                      Vec3 { x: 0.0, y: 1.0, z: 4.0 });
        let m3 = m1 * m2;
        assert_eq!(m3.0.x, 39.0);
        assert_eq!(m3.0.y, 45.0);
        assert_eq!(m3.0.z, 51.0);
        assert_eq!(m3.1.x, -3.0);
        assert_eq!(m3.1.y, 3.0);
        assert_eq!(m3.1.z, 9.0);
        assert_eq!(m3.2.x, 52.0);
        assert_eq!(m3.2.y, 62.0);
        assert_eq!(m3.2.z, 72.0);
        assert_eq!(m3.3.x, 42.0);
        assert_eq!(m3.3.y, 48.0);
        assert_eq!(m3.3.z, 54.0);
    }

    #[test]
    fn test_neg() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let m2 = -m1;
        assert_eq!(m2.0.x, -1.0);
        assert_eq!(m2.0.y, -2.0);
        assert_eq!(m2.0.z, -3.0);
        assert_eq!(m2.1.x, -4.0);
        assert_eq!(m2.1.y, -5.0);
        assert_eq!(m2.1.z, -6.0);
        assert_eq!(m2.2.x, -7.0);
        assert_eq!(m2.2.y, -8.0);
        assert_eq!(m2.2.z, -9.0);
        assert_eq!(m2.3.x, -10.0);
        assert_eq!(m2.3.y, -11.0);
        assert_eq!(m2.3.z, -12.0);
    }

    #[test]
    fn test_inverse() {
        let a = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                     Vec3 { x: 40.0, y: 5.0, z: 6.0 },
                     Vec3 { x: 7.0, y: 9.0, z: 9.0 },
                     Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let b = a.inverse();
        assert_approx_eq!(b.0.x, -0.02727);
        assert_approx_eq!(b.0.y, 0.02727);
        assert_approx_eq!(b.0.z, -0.00909);
        assert_approx_eq!(b.1.x, -0.96364);
        assert_approx_eq!(b.1.y, -0.03636);
        assert_approx_eq!(b.1.z, 0.34545);
        assert_approx_eq!(b.2.x, 0.98485);
        assert_approx_eq!(b.2.y, 0.01515);
        assert_approx_eq!(b.2.z, -0.22727);
        assert_approx_eq!(b.3.x, -0.94545);
        assert_approx_eq!(b.3.y, -0.05455);
        assert_approx_eq!(b.3.z, -0.98182);
    }

    #[test]
    fn test_identity() {
        let m: Tfm3<f64> = Tfm3::identity();
        assert_eq!(m.0.x, 1.0);
        assert_eq!(m.0.y, 0.0);
        assert_eq!(m.0.z, 0.0);
        assert_eq!(m.1.x, 0.0);
        assert_eq!(m.1.y, 1.0);
        assert_eq!(m.1.z, 0.0);
        assert_eq!(m.2.x, 0.0);
        assert_eq!(m.2.y, 0.0);
        assert_eq!(m.2.z, 1.0);
        assert_eq!(m.3.x, 0.0);
        assert_eq!(m.3.y, 0.0);
        assert_eq!(m.3.z, 0.0);
    }
}

#[cfg(test)]
mod tests_conversion {
    use super::{Mat3, Mat4, Tfm3};
    use vector::{Vec3, Vec4};

    #[test]
    fn test_mat4_to_mat3() {
        let m1 = Mat4(Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 },
                      Vec4 { x: 5.0, y: 6.0, z: 7.0, w: 8.0 },
                      Vec4 { x: 9.0, y: 10.0, z: 11.0, w: 12.0 },
                      Vec4 { x: 13.0, y: 14.0, z: 15.0, w: 16.0 });
        let m2 = Mat3::from(m1);
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 2.0);
        assert_eq!(m2.0.z, 3.0);
        assert_eq!(m2.1.x, 5.0);
        assert_eq!(m2.1.y, 6.0);
        assert_eq!(m2.1.z, 7.0);
        assert_eq!(m2.2.x, 9.0);
        assert_eq!(m2.2.y, 10.0);
        assert_eq!(m2.2.z, 11.0);
    }

    #[test]
    fn test_tfm3_to_mat3() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let m2 = Mat3::from(m1);
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 2.0);
        assert_eq!(m2.0.z, 3.0);
        assert_eq!(m2.1.x, 4.0);
        assert_eq!(m2.1.y, 5.0);
        assert_eq!(m2.1.z, 6.0);
        assert_eq!(m2.2.x, 7.0);
        assert_eq!(m2.2.y, 8.0);
        assert_eq!(m2.2.z, 9.0);
    }

    #[test]
    fn test_mat3_to_mat4() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = Mat4::from(m1);
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 2.0);
        assert_eq!(m2.0.z, 3.0);
        assert_eq!(m2.0.w, 0.0);
        assert_eq!(m2.1.x, 4.0);
        assert_eq!(m2.1.y, 5.0);
        assert_eq!(m2.1.z, 6.0);
        assert_eq!(m2.1.w, 0.0);
        assert_eq!(m2.2.x, 7.0);
        assert_eq!(m2.2.y, 8.0);
        assert_eq!(m2.2.z, 9.0);
        assert_eq!(m2.2.w, 0.0);
        assert_eq!(m2.3.x, 0.0);
        assert_eq!(m2.3.y, 0.0);
        assert_eq!(m2.3.z, 0.0);
        assert_eq!(m2.3.w, 1.0);
    }

    #[test]
    fn test_mat3_to_tfm3() {
        let m1 = Mat3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 });
        let m2 = Tfm3::from(m1);
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 2.0);
        assert_eq!(m2.0.z, 3.0);
        assert_eq!(m2.1.x, 4.0);
        assert_eq!(m2.1.y, 5.0);
        assert_eq!(m2.1.z, 6.0);
        assert_eq!(m2.2.x, 7.0);
        assert_eq!(m2.2.y, 8.0);
        assert_eq!(m2.2.z, 9.0);
        assert_eq!(m2.3.x, 0.0);
        assert_eq!(m2.3.y, 0.0);
        assert_eq!(m2.3.z, 0.0);
    }

    #[test]
    fn test_tfm3_to_mat4() {
        let m1 = Tfm3(Vec3 { x: 1.0, y: 2.0, z: 3.0 },
                      Vec3 { x: 4.0, y: 5.0, z: 6.0 },
                      Vec3 { x: 7.0, y: 8.0, z: 9.0 },
                      Vec3 { x: 10.0, y: 11.0, z: 12.0 });
        let m2 = Mat4::from(m1);
        assert_eq!(m2.0.x, 1.0);
        assert_eq!(m2.0.y, 2.0);
        assert_eq!(m2.0.z, 3.0);
        assert_eq!(m2.0.w, 0.0);
        assert_eq!(m2.1.x, 4.0);
        assert_eq!(m2.1.y, 5.0);
        assert_eq!(m2.1.z, 6.0);
        assert_eq!(m2.1.w, 0.0);
        assert_eq!(m2.2.x, 7.0);
        assert_eq!(m2.2.y, 8.0);
        assert_eq!(m2.2.z, 9.0);
        assert_eq!(m2.2.w, 0.0);
        assert_eq!(m2.3.x, 10.0);
        assert_eq!(m2.3.y, 11.0);
        assert_eq!(m2.3.z, 12.0);
        assert_eq!(m2.3.w, 1.0);
    }

    #[test]
    fn test_mat4_to_tfm3() {
    }
}
