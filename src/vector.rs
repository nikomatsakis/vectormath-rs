#![deny(missing_docs)]
//! Vector

use float::Float;
use float::{Min, Max};
use std::ops::{Add, Sub, Mul, Div, Neg};
use std::convert::From;

/// 3-D Vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{v} = x \mathbf{i} + y \mathbf{j} + z \mathbf{k}
///              = \begin{bmatrix} x \\ y \\ z  \end{bmatrix}
/// $$</div>
#[derive(Debug, Copy, Clone)]
pub struct Vec3<T> {
    /// 1st component
    pub x: T,
    /// 2nd component
    pub y: T,
    /// 3rd component
    pub z: T
}

/// 3-D Position
///
/// # Definition
///
/// <div>$$
///   \mathbf{p} = x \mathbf{i} + y \mathbf{j} + z \mathbf{k}
///              = \begin{bmatrix} x \\ y \\ z \end{bmatrix}
/// $$</div>
#[derive(Debug, Copy, Clone)]
pub struct Pos3<T> {
    /// 1st component
    pub x: T,
    /// 2nd component
    pub y: T,
    /// 3rd component
    pub z: T
}

/// 4-D Vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{v} = \begin{bmatrix} x \\ y \\ z \\ w \end{bmatrix}
/// $$</div>
#[derive(Debug, Copy, Clone)]
pub struct Vec4<T> {
    /// 1st component
    pub x: T,
    /// 2nd component
    pub y: T,
    /// 3rd component
    pub z: T,
    /// 4th component
    pub w: T
}

/// Addition
///
/// # Definition
///
/// <div>$$
///   \mathbf{v}_1 + \mathbf{v}_2 = (x_1 + x_2) \mathbf{i} + (y_1 + y_2) \mathbf{j} + (z_1 + z_2) \mathbf{k}
/// $$</div>
impl<T> Add for Vec3<T> where T: Float {
    type Output = Vec3<T>;
    fn add(self, _rhs: Vec3<T>) -> Vec3<T> {
        Vec3 { x: (self.x + _rhs.x),
               y: (self.y + _rhs.y),
               z: (self.z + _rhs.z) }
    }
}

/// Addition
///
/// # Definition
///
/// <div>$$
///   \mathbf{v}_1 + \mathbf{v}_2 = \begin{bmatrix} x_1 + x_2 \\ y_1 + y_2 \\ z_1 + z_2 \\ w_1 + w_2 \end{bmatrix}
/// $$</div>
impl<T> Add for Vec4<T> where T: Float {
    type Output = Vec4<T>;
    fn add(self, _rhs: Vec4<T>) -> Vec4<T> {
        Vec4 { x: (self.x + _rhs.x),
               y: (self.y + _rhs.y),
               z: (self.z + _rhs.z),
               w: (self.w + _rhs.w) }
    }
}

/// Point Movement
///
/// # Definition
///
/// <div>$$
///   \mathbf{p} + \mathbf{v} = (x_p + x_v) \mathbf{i} + (y_p + y_v) \mathbf{j} + (z_p + z_v) \mathbf{k}
/// $$</div>
impl<T> Add<Vec3<T>> for Pos3<T> where T: Float {
    type Output = Pos3<T>;
    fn add(self, _rhs: Vec3<T>) -> Pos3<T> {
        Pos3 { x: (self.x + _rhs.x),
               y: (self.y + _rhs.y),
               z: (self.z + _rhs.z) }
    }
}

/// Subtraction
///
/// # Definition
///
/// <div>$$
///   \mathbf{v}_1 - \mathbf{v}_2 = (x_1 - x_2) \mathbf{i} + (y_1 - y_2) \mathbf{j} + (z_1 - z_2) \mathbf{k}
/// $$</div>
impl<T> Sub for Vec3<T> where T: Float {
    type Output = Vec3<T>;
    fn sub(self, _rhs: Vec3<T>) -> Vec3<T> {
        Vec3 { x: (self.x - _rhs.x),
               y: (self.y - _rhs.y),
               z: (self.z - _rhs.z) }
    }
}

/// Subtraction
///
/// # Definition
///
/// <div>$$
///   \mathbf{v}_1 - \mathbf{v}_2 = \begin{bmatrix} x_1 - x_2 \\ y_1 - y_2 \\ z_1 - z_2 \\ w_1 - w_2 \end{bmatrix}
/// $$</div>
impl<T> Sub for Vec4<T> where T: Float {
    type Output = Vec4<T>;
    fn sub(self, _rhs: Vec4<T>) -> Vec4<T> {
        Vec4 { x: (self.x - _rhs.x),
               y: (self.y - _rhs.y),
               z: (self.z - _rhs.z),
               w: (self.w - _rhs.w) }
    }
}

/// Point Movement
///
/// # Definition
///
/// <div>$$
///   \mathbf{p} - \mathbf{v} = (x_p - x_v) \mathbf{i} + (y_p - y_v) \mathbf{j} + (z_p - z_v) \mathbf{k}
/// $$</div>
impl<T> Sub<Vec3<T>> for Pos3<T> where T: Float {
    type Output = Pos3<T>;
    fn sub(self, _rhs: Vec3<T>) -> Pos3<T> {
        Pos3 { x: (self.x - _rhs.x),
               y: (self.y - _rhs.y),
               z: (self.z - _rhs.z) }
    }
}

/// Point Difference
///
/// # Definition
///
/// <div>$$
///   \mathbf{p}_1 - \mathbf{p}_2
///       = \Delta \mathbf{p}
///       = (x_1 - x_2) \mathbf{i} + (y_1 - y_2) \mathbf{j} + (z_1 - z_2) \mathbf{k}
/// $$</div>
impl<T> Sub for Pos3<T> where T: Float {
    type Output = Vec3<T>;
    fn sub(self, _rhs: Pos3<T>) -> Vec3<T> {
        Vec3 { x: (self.x - _rhs.x),
               y: (self.y - _rhs.y),
               z: (self.z - _rhs.z) }
    }
}

/// Multiplication by scalar
///
/// # Definition
///
/// <div>$$
///   \mathbf{v} \cdot s = (s x) \mathbf{i} + (s y) \mathbf{j} + (s z) \mathbf{k}, \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Mul<T> for Vec3<T> where T: Float {
    type Output = Vec3<T>;
    fn mul(self, s: T) -> Vec3<T> {
        Vec3 { x: (self.x * s),
               y: (self.y * s),
               z: (self.z * s) }
    }
}

/// Multiplication by scalar
///
/// # Definition
///
/// <div>$$
///   \mathbf{v} \cdot s = \begin{bmatrix} s x \\ s y \\ s z \\ s w \end{bmatrix}, \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Mul<T> for Vec4<T> where T: Float {
    type Output = Vec4<T>;
    fn mul(self, s: T) -> Vec4<T> {
        Vec4 { x: (self.x * s),
               y: (self.y * s),
               z: (self.z * s),
               w: (self.w * s) }
    }
}

/// Multiplication by scalar
///
/// # Definition
///
/// <div>$$
///   \mathbf{p} \cdot s = (s x) \mathbf{i} + (s y) \mathbf{j} + (s z) \mathbf{k}, \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Mul<T> for Pos3<T> where T: Float {
    type Output = Pos3<T>;
    fn mul(self, s: T) -> Pos3<T> {
        Pos3 { x: (self.x * s),
               y: (self.y * s),
               z: (self.z * s) }
    }
}

/// Division by scalar
///
/// # Definition
///
/// <div>$$
///   \frac{\mathbf{v}}{s} = \frac{x}{s} \mathbf{i} + \frac{y}{s} \mathbf{j} + \frac{z}{s} \mathbf{k}, \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Div<T> for Vec3<T> where T: Float {
    type Output = Vec3<T>;
    fn div(self, s: T) -> Vec3<T> {
        Vec3 { x: (self.x / s),
               y: (self.y / s),
               z: (self.z / s) }
    }
}

/// Division by scalar
///
/// # Definition
///
/// <div>$$
///   \frac{\mathbf{v}}{s} = \begin{bmatrix} \frac{x}{s} \\ \frac{y}{s} \\ \frac{z}{s} \\ \frac{w}{s} \end{bmatrix}, \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Div<T> for Vec4<T> where T: Float {
    type Output = Vec4<T>;
    fn div(self, s: T) -> Vec4<T> {
        Vec4 { x: (self.x / s),
               y: (self.y / s),
               z: (self.z / s),
               w: (self.w / s) }
    }
}

/// Division by scalar
///
/// # Definition
///
/// <div>$$
///   \frac{\mathbf{p}}{s} = \frac{x}{s} \mathbf{i} + \frac{y}{s} \mathbf{j} + \frac{z}{s} \mathbf{k}, \\
///   \textrm{where } s \in \mathbb{R}
/// $$</div>
impl<T> Div<T> for Pos3<T> where T: Float {
    type Output = Pos3<T>;
    fn div(self, s: T) -> Pos3<T> {
        Pos3 { x: (self.x / s),
               y: (self.y / s),
               z: (self.z / s) }
    }
}

/// Negation
///
/// # Definition
///
/// <div>$$
///   -\mathbf{v} = (-x) \mathbf{i} + (-y) \mathbf{j} + (-z) \mathbf{k}
/// $$</div>
impl<T> Neg for Vec3<T> where T: Float {
    type Output = Vec3<T>;
    fn neg(self) -> Vec3<T> {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

/// Negation
///
/// # Definition
///
/// <div>$$
///   -\mathbf{v} = \begin{bmatrix} -x \\ -y \\ -z \\ -w \end{bmatrix}
/// $$</div>
impl<T> Neg for Vec4<T> where T: Float {
    type Output = Vec4<T>;
    fn neg(self) -> Vec4<T> {
        Vec4 { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

/// Negation
///
/// # Definition
///
/// <div>$$
///   -\mathbf{v} = (-x) \mathbf{i} + (-y) \mathbf{j} + (-z) \mathbf{k}
/// $$</div>
impl<T> Neg for Pos3<T> where T: Float {
    type Output = Pos3<T>;
    fn neg(self) -> Pos3<T> {
        Pos3 { x: -self.x, y: -self.y, z: -self.z }
    }
}

//
// Functions
//

impl<T: Float> Vec3<T> {
    /// Dot Product
    ///
    /// # Definition
    ///
    /// <div>$$
    /// \mathbf{a} \cdot \mathbf{b} =
    ///   \begin{bmatrix} a_x & a_y & a_z \end{bmatrix}
    ///   \begin{bmatrix} b_x \\ b_y \\ b_z \end{bmatrix}
    ///   = a_x b_x + a_y b_y + a_z b_z
    /// $$</div>
    pub fn dot(self, rhs: Self) -> T {
        let mx = self.x * rhs.x;
        let my = self.y * rhs.y;
        let mz = self.z * rhs.z;
        (mx + my + mz)
    }

    /// Cross Product
    ///
    /// # Definition
    ///
    /// <div>$$
    /// \mathbf{a} \times \mathbf{b} =
    ///   \begin{vmatrix}
    ///     \mathbf{i} && \mathbf{j} && \mathbf{k} \\
    ///     a_x && a_y && a_z \\
    ///     b_x && b_y && b_z
    ///   \end{vmatrix}
    ///   = (a_y b_z - a_z b_y) \mathbf{i} +
    ///     (a_z b_x - a_x b_z) \mathbf{j} +
    ///     (a_x b_y - a_y b_x) \mathbf{k}
    /// $$</div>
    pub fn cross(self, rhs: Self) -> Self {
        let tx = self.y * rhs.z - self.z * rhs.y;
        let ty = self.z * rhs.x - self.x * rhs.z;
        let tz = self.x * rhs.y - self.y * rhs.x;
        Vec3 { x: tx, y: ty, z: tz }
    }

    /// Magnitude
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   |\mathbf{v}| = \sqrt{\mathbf{v} \cdot \mathbf{v}}
    /// $$</div>
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }

    /// Squared Magnitude
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   |\mathbf{v}|^2 = \mathbf{v} \cdot \mathbf{v}
    /// $$</div>
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    /// Normalization
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \hat{\mathbf{v}} = \frac{\mathbf{v}}{|\mathbf{v}|}
    /// $$</div>
    pub fn normalize(self) -> Vec3<T> {
        self * self.length_squared().rsqrt()
    }

    /// Linear Interpolation
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{v} = \mathrm{lerp}(\mathbf{v}_0, \mathbf{v}_1; t)
    ///              = \mathbf{v}_0 + t (\mathbf{v}_1 - \mathbf{v}_0), \\
    ///   \textrm{where } t \in \mathbb{R}
    /// $$</div>
    pub fn lerp(self, dest: Vec3<T>, t: T) -> Vec3<T> {
        self + (dest - self) * t
    }
}

impl<T: Float> Vec4<T> {
    /// Dot Product
    ///
    /// # Definition
    ///
    /// <div>$$
    /// \mathbf{a} \cdot \mathbf{b} =
    ///   \begin{bmatrix} a_x & a_y & a_z & a_w \end{bmatrix}
    ///   \begin{bmatrix} b_x \\ b_y \\ b_z \\ b_w \end{bmatrix}
    ///   = a_x b_x + a_y b_y + a_z b_z + a_w b_w
    /// $$</div>
    pub fn dot(self, rhs: Self) -> T {
        let mx = self.x * rhs.x;
        let my = self.y * rhs.y;
        let mz = self.z * rhs.z;
        let mw = self.w * rhs.w;
        (mx + my + mz + mw)
    }

    /// Magnitude
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   |\mathbf{v}| = \sqrt{\mathbf{v} \cdot \mathbf{v}}
    /// $$</div>
    pub fn length(self) -> T {
        self.length_squared().sqrt()
    }

    /// Squared Magnitude
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   |\mathbf{v}|^2 = \mathbf{v} \cdot \mathbf{v}
    /// $$</div>
    pub fn length_squared(self) -> T {
        self.dot(self)
    }

    /// Normalization
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \hat{\mathbf{v}} = \frac{\mathbf{v}}{|\mathbf{v}|}
    /// $$</div>
    pub fn normalize(self) -> Vec4<T> {
        self * self.length_squared().rsqrt()
    }

    /// Linear Interpolation
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{v} = \mathrm{lerp}(\mathbf{v}_0, \mathbf{v}_1; t)
    ///              = \mathbf{v}_0 + t (\mathbf{v}_1 - \mathbf{v}_0), \\
    ///   \textrm{where } t \in \mathbb{R}
    /// $$</div>
    pub fn lerp(self, dest: Vec4<T>, t: T) -> Vec4<T> {
        self + (dest - self) * t
    }
}

impl<T: Float> Pos3<T> {
    /// Linear Interpolation
    ///
    /// # Definition
    ///
    /// <div>$$
    ///   \mathbf{p} = \mathrm{lerp}(\mathbf{p}_0, \mathbf{p}_1; t)
    ///              = \mathbf{p}_0 + t (\mathbf{p}_1 - \mathbf{p}_0), \\
    ///   \textrm{where } t \in \mathbb{R}
    /// $$</div>
    pub fn lerp(self, dest: Pos3<T>, t: T) -> Pos3<T> {
        self + (dest - self) * t
    }
}

/// Component-wise minimum
///
/// # Definition
///
/// <div>$$
///   \mathrm{min}(\mathbf{a},\mathbf{b}) = \begin{bmatrix}
///           \mathrm{min}(a_x, b_x) \\
///           \mathrm{min}(a_y, b_y) \\
///           \mathrm{min}(a_z, b_z)
///       \end{bmatrix}
/// $$</div>
impl<T: Float> Min for Vec3<T> {
    fn min(self, _rhs: Self) -> Self {
        let mx = self.x.min(_rhs.x);
        let my = self.y.min(_rhs.y);
        let mz = self.z.min(_rhs.z);
        Vec3 { x: mx, y: my, z: mz }
    }
}

/// Component-wise minimum
///
/// # Definition
///
/// <div>$$
///   \mathrm{min}(\mathbf{a},\mathbf{b}) = \begin{bmatrix}
///           \mathrm{min}(a_x, b_x) \\
///           \mathrm{min}(a_y, b_y) \\
///           \mathrm{min}(a_z, b_z) \\
///           \mathrm{min}(a_w, b_w)
///       \end{bmatrix}
/// $$</div>
impl<T: Float> Min for Vec4<T> {
    fn min(self, _rhs: Self) -> Self {
        let mx = self.x.min(_rhs.x);
        let my = self.y.min(_rhs.y);
        let mz = self.z.min(_rhs.z);
        let mw = self.w.min(_rhs.w);
        Vec4 { x: mx, y: my, z: mz, w: mw }
    }
}

/// Component-wise minimum
///
/// # Definition
///
/// <div>$$
///   \mathrm{min}(\mathbf{a},\mathbf{b}) = \begin{bmatrix}
///           \mathrm{min}(a_x, b_x) \\
///           \mathrm{min}(a_y, b_y) \\
///           \mathrm{min}(a_z, b_z)
///       \end{bmatrix}
/// $$</div>
impl<T: Float> Min for Pos3<T> {
    fn min(self, _rhs: Self) -> Self {
        let mx = self.x.min(_rhs.x);
        let my = self.y.min(_rhs.y);
        let mz = self.z.min(_rhs.z);
        Pos3 { x: mx, y: my, z: mz }
    }
}

/// Component-wise maximum
///
/// # Definition
///
/// <div>$$
///   \mathrm{max}(\mathbf{a},\mathbf{b}) = \begin{bmatrix}
///           \mathrm{max}(a_x, b_x) \\
///           \mathrm{max}(a_y, b_y) \\
///           \mathrm{max}(a_z, b_z)
///       \end{bmatrix}
/// $$</div>
impl<T: Float> Max for Vec3<T> {
    fn max(self, _rhs: Self) -> Self {
        let mx = self.x.max(_rhs.x);
        let my = self.y.max(_rhs.y);
        let mz = self.z.max(_rhs.z);
        Vec3 { x: mx, y: my, z: mz }
    }
}

/// Component-wise maximum
///
/// # Definition
///
/// <div>$$
///   \mathrm{max}(\mathbf{a},\mathbf{b}) = \begin{bmatrix}
///           \mathrm{max}(a_x, b_x) \\
///           \mathrm{max}(a_y, b_y) \\
///           \mathrm{max}(a_z, b_z) \\
///           \mathrm{max}(a_w, b_w)
///       \end{bmatrix}
/// $$</div>
impl<T: Float> Max for Vec4<T> {
    fn max(self, _rhs: Self) -> Self {
        let mx = self.x.max(_rhs.x);
        let my = self.y.max(_rhs.y);
        let mz = self.z.max(_rhs.z);
        let mw = self.w.max(_rhs.w);
        Vec4 { x: mx, y: my, z: mz, w: mw }
    }
}

/// Component-wise maximum
///
/// # Definition
///
/// <div>$$
///   \mathrm{max}(\mathbf{a},\mathbf{b}) = \begin{bmatrix}
///           \mathrm{max}(a_x, b_x) \\
///           \mathrm{max}(a_y, b_y) \\
///           \mathrm{max}(a_z, b_z)
///       \end{bmatrix}
/// $$</div>
impl<T: Float> Max for Pos3<T> {
    fn max(self, _rhs: Self) -> Self {
        let mx = self.x.max(_rhs.x);
        let my = self.y.max(_rhs.y);
        let mz = self.z.max(_rhs.z);
        Pos3 { x: mx, y: my, z: mz }
    }
}

/// Cast from 3-D Position to 3-D Vector
impl<T> From<Pos3<T>> for Vec3<T> {
    fn from(p: Pos3<T>) -> Vec3<T> {
        Vec3 { x: p.x, y: p.y, z: p.z }
    }
}

/// Cast from 4-D Vector to 3-D Vector
impl<T> From<Vec4<T>> for Vec3<T> {
    fn from(v: Vec4<T>) -> Vec3<T> {
        Vec3 { x: v.x, y: v.y, z: v.z }
    }
}

/// Cast from 3-D Vector to 4-D Vector
///
/// # Definition
///
/// <div>$$
///   \mathbf{v} = \begin{bmatrix} x \\ y \\ z \\ 0 \end{bmatrix}
/// $$</div>
impl<T> From<Vec3<T>> for Vec4<T> where T: Float {
    fn from(v: Vec3<T>) -> Vec4<T> {
        Vec4 { x: v.x, y: v.y, z: v.z, w: T::zero() }
    }
}

/// Cast from 3-D Position to 4-D Vector
///
/// ## Definition
///
/// <div>$$
///   \mathbf{v} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
/// $$</div>
impl<T> From<Pos3<T>> for Vec4<T> where T: Float {
    fn from(p: Pos3<T>) -> Vec4<T> {
        Vec4 { x: p.x, y: p.y, z: p.z, w: T::one() }
    }
}

/// Cast from 3-D Vector to 3-D Position
impl<T> From<Vec3<T>> for Pos3<T> {
    fn from(v: Vec3<T>) -> Pos3<T> {
        Pos3 { x: v.x, y: v.y, z: v.z }
    }
}

/// Project from 4-D Vector to 3-D Position
///
/// # Definition
///
/// <div>$$
///   \mathbf{p} = \frac{ x \mathbf{i} + y \mathbf{j} + z \mathbf{k} }{ w }
/// $$</div>
impl<T> From<Vec4<T>> for Pos3<T> where T: Float {
    fn from(v: Vec4<T>) -> Pos3<T> {
        Pos3 { x: v.x / v.w, y: v.y / v.w, z: v.z / v.w }
    }
}

//
// Constructors
//

impl<T: Float> Vec3<T> {
    /// X Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
    /// $$</div>
    pub fn x_axis() -> Vec3<T> { Vec3{ x: T::one(), y: T::zero(), z: T::zero() } }

    /// Y Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
    /// $$</div>
    pub fn y_axis() -> Vec3<T> { Vec3{ x: T::zero(), y: T::one(), z: T::zero() } }

    /// Z Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
    /// $$</div>
    pub fn z_axis() -> Vec3<T> { Vec3{ x: T::zero(), y: T::zero(), z: T::one() } }
}

impl<T: Float> Vec4<T> {
    /// X Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}
    /// $$</div>
    pub fn x_axis() -> Vec4<T> { Vec4{ x: T::one(), y: T::zero(), z: T::zero(), w: T::zero() } }

    /// Y Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}
    /// $$</div>
    pub fn y_axis() -> Vec4<T> { Vec4{ x: T::zero(), y: T::one(), z: T::zero(), w: T::zero() } }

    /// Z Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}
    /// $$</div>
    pub fn z_axis() -> Vec4<T> { Vec4{ x: T::zero(), y: T::zero(), z: T::one(), w: T::zero() } }

    /// W Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}
    /// $$</div>
    pub fn w_axis() -> Vec4<T> { Vec4{ x: T::zero(), y: T::zero(), z: T::zero(), w: T::one() } }
}

impl<T: Float> Pos3<T> {
    /// X Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
    /// $$</div>
    pub fn x_axis() -> Pos3<T> { Pos3{ x: T::one(), y: T::zero(), z: T::zero() } }

    /// Y Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}
    /// $$</div>
    pub fn y_axis() -> Pos3<T> { Pos3{ x: T::zero(), y: T::one(), z: T::zero() } }

    /// Z Axis
    ///
    /// <div>$$
    ///   \mathbf{v} = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
    /// $$</div>
    pub fn z_axis() -> Pos3<T> { Pos3{ x: T::zero(), y: T::zero(), z: T::one() } }
}

#[cfg(test)]
mod test_checktype {
    use super::*;
    use std::ops::{Add, Sub, Mul, Div, Neg};
    use float::{Min, Max};

    impl TVec3<f32> for Vec3<f32> {}
    impl TVec4<f32> for Vec4<f32> {}
    impl TPos3<f32> for Pos3<f32> {}

    trait TVec3<T>:
        Copy
        + Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + Mul<T, Output = Self>
        + Div<T, Output = Self>
        + Neg<Output = Self>
        + Min + Max
    {
    }

    trait TVec4<T>:
        Copy
        + Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + Mul<T, Output = Self>
        + Div<T, Output = Self>
        + Neg<Output = Self>
        + Min + Max
    {
    }

    trait TPos3<T>:
        Copy
        + Add<Vec3<T>, Output = Self>
        + Sub<Vec3<T>, Output = Self>
        + Sub<Self, Output = Vec3<T>>
        + Mul<T, Output = Self>
        + Div<T, Output = Self>
        + Neg<Output = Self>
        + Min + Max
    {
    }
}

#[cfg(test)]
mod tests_vec3 {
    use super::Vec3;

    #[test]
    fn test_add() {
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2 = Vec3 { x: 0.5, y: -1.5, z: 4.0 };
        let v3 = v1 + v2;
        assert_eq!(v3.x, 1.5);
        assert_eq!(v3.y, 0.5);
        assert_eq!(v3.z, 7.0);
    }

    #[test]
    fn test_sub() {
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2 = Vec3 { x: 0.5, y: -1.5, z: 4.0 };
        let v3 = v1 - v2;
        assert_eq!(v3.x, 0.5);
        assert_eq!(v3.y, 3.5);
        assert_eq!(v3.z, -1.0);
    }

    #[test]
    fn test_mul() {
        let v = Vec3 { x: 0.5, y: -1.5, z: 2.0 };
        let v2 = v * 2.0_f32;
        assert_eq!(v2.x, 1.0);
        assert_eq!(v2.y, -3.0);
        assert_eq!(v2.z, 4.0);
    }

    #[test]
    fn test_div() {
        let v = Vec3 { x: 0.5, y: -1.5, z: 2.0 };
        let v2 = v / 2.0_f32;
        assert_eq!(v2.x, 0.25);
        assert_eq!(v2.y, -0.75);
        assert_eq!(v2.z, 1.0);
    }

    #[test]
    fn test_neg() {
        let v1 = Vec3 { x: 1.0, y: 0.5, z: -3.0 };
        let v2 = -v1;
        assert_eq!(v2.x, -1.0);
        assert_eq!(v2.y, -0.5);
        assert_eq!(v2.z, 3.0);
    }

    #[test]
    fn test_length() {
        let v1 = Vec3::<f32> { x: 2.0, y: 1.0, z: 2.0 };
        let s = v1.length();
        assert_eq!(s, 3.0);
    }

    #[test]
    fn test_lengthsqr() {
        let v1 = Vec3::<f32> { x: 2.0, y: 1.0, z: 2.0 };
        let s = v1.length_squared();
        assert_eq!(s, 9.0);
    }

    #[test]
    fn test_normalize() {
        let a = Vec3 { x: 2.0, y: -1.0, z: 2.0 };
        let b = a.normalize();
        assert_approx_eq!(b.x, 2.0 / 3.0);
        assert_approx_eq!(b.y, -1.0 / 3.0);
        assert_approx_eq!(b.z, 2.0 / 3.0);
    }

    #[test]
    fn test_dot() {
        let a = Vec3 { x: 1.0, y: 0.5, z: -1.5 };
        let b = Vec3 { x: 2.0, y: -2.0, z: 1.0 };
        let dp = a.dot(b);
        assert_eq!(dp, -0.5);
    }

    #[test]
    fn test_cross() {
        let a = Vec3 { x: 1.0, y: 1.0, z: -1.0 };
        let b = Vec3 { x: -1.0, y: 1.0, z: 1.0 };
        let xp = a.cross(b);
        assert_eq!(xp.x, 2.0);
        assert_eq!(xp.y, 0.0);
        assert_eq!(xp.z, 2.0);
    }

    #[test]
    fn test_min() {
        use float::Min;
        let a = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let b = Vec3 { x: 0.5, y: 4.0, z: -1.0 };
        let vmin = a.min(b);
        assert_eq!(vmin.x, 0.5);
        assert_eq!(vmin.y, 2.0);
        assert_eq!(vmin.z, -1.0);
    }

    #[test]
    fn test_max() {
        use float::Max;
        let a = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let b = Vec3 { x: 0.5, y: 4.0, z: -1.0 };
        let vmax = a.max(b);
        assert_eq!(vmax.x, 1.0);
        assert_eq!(vmax.y, 4.0);
        assert_eq!(vmax.z, 3.0);
    }

    #[test]
    fn test_lerp() {
        let a = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let b = Vec3 { x: 0.0, y: -5.0, z: 7.0 };
        let c = a.lerp(b, 0.25);
        assert_eq!(c.x, 0.75);
        assert_eq!(c.y, 0.25);
        assert_eq!(c.z, 4.0);
    }

    #[test]
    fn test_x_axis() {
        let v: Vec3<f64> = Vec3::x_axis();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn test_y_axis() {
        let v: Vec3<f64> = Vec3::y_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 1.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn test_z_axis() {
        let v: Vec3<f64> = Vec3::z_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 1.0);
    }
}

#[cfg(test)]
mod tests_vec4 {
    use super::Vec4;

    #[test]
    fn test_add() {
        let a = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = Vec4 { x: 0.5, y: -3.5, z: 0.0, w: -6.0 };
        let c = a + b;
        assert_eq!(c.x, 1.5);
        assert_eq!(c.y, -1.5);
        assert_eq!(c.z, 3.0);
        assert_eq!(c.w, -2.0);
    }

    #[test]
    fn test_sub() {
        let a = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = Vec4 { x: 0.5, y: -3.5, z: 0.0, w: 5.0 };
        let c = a - b;
        assert_eq!(c.x, 0.5);
        assert_eq!(c.y, 5.5);
        assert_eq!(c.z, 3.0);
        assert_eq!(c.w, -1.0);
    }

    #[test]
    fn test_mul() {
        let a = Vec4 { x: 1.0, y: -2.0, z: 3.5, w: -0.5 };
        let b = a * 2.0_f32;
        assert_eq!(b.x, 2.0);
        assert_eq!(b.y, -4.0);
        assert_eq!(b.z, 7.0);
        assert_eq!(b.w, -1.0);
    }

    #[test]
    fn test_div() {
        let a = Vec4 { x: 1.0, y: -2.0, z: 3.5, w: -0.5 };
        let b = a / 2.0_f32;
        assert_eq!(b.x, 0.5);
        assert_eq!(b.y, -1.0);
        assert_eq!(b.z, 1.75);
        assert_eq!(b.w, -0.25);
    }

    #[test]
    fn test_neg() {
        let a = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = -a;
        assert_eq!(b.x, -1.0);
        assert_eq!(b.y, -2.0);
        assert_eq!(b.z, -3.0);
        assert_eq!(b.w, -4.0);
    }

    #[test]
    fn test_length() {
        let a = Vec4 { x: 1.0, y: 2.0, z: 2.0, w: 4.0 };
        let l = a.length();
        assert_eq!(l, 5.0);
    }

    #[test]
    fn test_lengthsqr() {
        let a = Vec4 { x: 1.0, y: 2.0, z: 2.0, w: 4.0 };
        let l = a.length_squared();
        assert_eq!(l, 25.0);
    }

    #[test]
    fn test_normalize() {
        let a = Vec4 { x: 1.0, y: 2.0, z: 2.0, w: 4.0 };
        let b = a.normalize();
        assert_eq!(b.x, 0.2);
        assert_eq!(b.y, 0.4);
        assert_eq!(b.z, 0.4);
        assert_eq!(b.w, 0.8);
    }

    #[test]
    fn test_dot() {
        let a = Vec4 { x: 1.0, y: 0.5, z: -1.5, w: 100.0 };
        let b = Vec4 { x: 2.0, y: -2.0, z: 1.0, w: 0.0 };
        let dp = a.dot(b);
        assert_eq!(dp, -0.5);
    }

    #[test]
    fn test_min() {
        use float::Min;
        let a = Vec4 { x: 2.0, y: 0.5, z: -1.0, w: 1.0 };
        let b = Vec4 { x: 1.0, y: 1.0, z: -2.0, w: -1.0 };
        let vmin = a.min(b);
        assert_eq!(vmin.x, 1.0);
        assert_eq!(vmin.y, 0.5);
        assert_eq!(vmin.z, -2.0);
        assert_eq!(vmin.w, -1.0);
    }

    #[test]
    fn test_max() {
        use float::Max;
        let a = Vec4 { x: 2.0, y: 0.5, z: -1.0, w: 1.0 };
        let b = Vec4 { x: 1.0, y: 1.0, z: -2.0, w: -1.0 };
        let vmax = a.max(b);
        assert_eq!(vmax.x, 2.0);
        assert_eq!(vmax.y, 1.0);
        assert_eq!(vmax.z, -1.0);
        assert_eq!(vmax.w, 1.0);
    }

    #[test]
    fn test_lerp() {
        let a = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: -4.0 };
        let b = Vec4 { x: 0.0, y: -5.0, z: 7.0, w: -5.0 };
        let c = a.lerp(b, 0.25);
        assert_eq!(c.x, 0.75);
        assert_eq!(c.y, 0.25);
        assert_eq!(c.z, 4.0);
        assert_eq!(c.w, -4.25);
    }

    #[test]
    fn test_x_axis() {
        let v: Vec4<f64> = Vec4::x_axis();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
        assert_eq!(v.w, 0.0);
    }

    #[test]
    fn test_y_axis() {
        let v: Vec4<f64> = Vec4::y_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 1.0);
        assert_eq!(v.z, 0.0);
        assert_eq!(v.w, 0.0);
    }

    #[test]
    fn test_z_axis() {
        let v: Vec4<f64> = Vec4::z_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 1.0);
        assert_eq!(v.w, 0.0);
    }

    #[test]
    fn test_w_axis() {
        let v: Vec4<f64> = Vec4::w_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
        assert_eq!(v.w, 1.0);
    }
}

#[cfg(test)]
mod tests_pos3 {
    use super::Pos3;

    #[test]
    fn test_add() {
        use super::Vec3;
        let v1 = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2 = Vec3 { x: 0.5, y: -1.5, z: 4.0 };
        let v3: Pos3<f32> = v1 + v2;
        assert_eq!(v3.x, 1.5);
        assert_eq!(v3.y, 0.5);
        assert_eq!(v3.z, 7.0);
    }

    #[test]
    fn test_sub_v3() {
        use super::Vec3;
        let v1 = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2 = Vec3 { x: 0.5, y: -1.5, z: 4.0 };
        let v3: Pos3<f32> = v1 - v2;
        assert_eq!(v3.x, 0.5);
        assert_eq!(v3.y, 3.5);
        assert_eq!(v3.z, -1.0);
    }

    #[test]
    fn test_sub_p3() {
        use super::Vec3;
        let v1 = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2 = Pos3 { x: 0.5, y: -1.5, z: 4.0 };
        let v3: Vec3<f32> = v1 - v2;
        assert_eq!(v3.x, 0.5);
        assert_eq!(v3.y, 3.5);
        assert_eq!(v3.z, -1.0);
    }

    #[test]
    fn test_mul() {
        let v = Pos3 { x: 0.5, y: -1.5, z: 2.0 };
        let v2 = v * 2.0_f32;
        assert_eq!(v2.x, 1.0);
        assert_eq!(v2.y, -3.0);
        assert_eq!(v2.z, 4.0);
    }

    #[test]
    fn test_div() {
        let v = Pos3 { x: 0.5, y: -1.5, z: 2.0 };
        let v2 = v / 2.0_f32;
        assert_eq!(v2.x, 0.25);
        assert_eq!(v2.y, -0.75);
        assert_eq!(v2.z, 1.0);
    }

    #[test]
    fn test_neg() {
        let v1 = Pos3 { x: 1.0, y: 0.5, z: -3.0 };
        let v2 = -v1;
        assert_eq!(v2.x, -1.0);
        assert_eq!(v2.y, -0.5);
        assert_eq!(v2.z, 3.0);
    }

    #[test]
    fn test_min() {
        use float::Min;
        let a = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let b = Pos3 { x: 0.5, y: 4.0, z: -1.0 };
        let vmin = a.min(b);
        assert_eq!(vmin.x, 0.5);
        assert_eq!(vmin.y, 2.0);
        assert_eq!(vmin.z, -1.0);
    }

    #[test]
    fn test_max() {
        use float::Max;
        let a = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let b = Pos3 { x: 0.5, y: 4.0, z: -1.0 };
        let vmax = a.max(b);
        assert_eq!(vmax.x, 1.0);
        assert_eq!(vmax.y, 4.0);
        assert_eq!(vmax.z, 3.0);
    }

    #[test]
    fn test_lerp() {
        let a = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let b = Pos3 { x: 0.0, y: -5.0, z: 7.0 };
        let c: Pos3<_> = a.lerp(b, 0.25);
        assert_eq!(c.x, 0.75);
        assert_eq!(c.y, 0.25);
        assert_eq!(c.z, 4.0);
    }

    #[test]
    fn test_x_axis() {
        let v: Pos3<f64> = Pos3::x_axis();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn test_y_axis() {
        let v: Pos3<f64> = Pos3::y_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 1.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn test_z_axis() {
        let v: Pos3<f64> = Pos3::z_axis();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 1.0);
    }
}

#[cfg(test)]
mod tests_conversion {
    use super::{Vec3, Vec4, Pos3};

    #[test]
    fn test_vec3_to_vec4() {
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2: Vec4<f32> = From::from(v1);
        assert_eq!(v2.x, 1.0);
        assert_eq!(v2.y, 2.0);
        assert_eq!(v2.z, 3.0);
        assert_eq!(v2.w, 0.0);
    }

    #[test]
    fn test_vec4_to_vec3() {
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v2: Vec3<f32> = From::from(v1);
        assert_eq!(v2.x, 1.0);
        assert_eq!(v2.y, 2.0);
        assert_eq!(v2.z, 3.0);
    }

    #[test]
    fn test_pos3_to_vec3() {
        let v1 = Pos3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2: Vec3<f32> = From::from(v1);
        assert_eq!(v2.x, 1.0);
        assert_eq!(v2.y, 2.0);
        assert_eq!(v2.z, 3.0);
    }

    #[test]
    fn test_pos3_to_vec4() {
        let v1 = Pos3 { x: 10.0, y: 20.0, z: 30.0 };
        let v2: Vec4<f32> = From::from(v1);
        assert_eq!(v2.x, 10.0);
        assert_eq!(v2.y, 20.0);
        assert_eq!(v2.z, 30.0);
        assert_eq!(v2.w, 1.0);
    }

    #[test]
    fn test_vec3_to_pos3() {
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2: Pos3<f32> = From::from(v1);
        assert_eq!(v2.x, 1.0);
        assert_eq!(v2.y, 2.0);
        assert_eq!(v2.z, 3.0);
    }

    #[test]
    fn test_vec4_to_pos3() {
        let v1 = Vec4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let v2: Pos3<f32> = From::from(v1);
        assert_eq!(v2.x, 0.25);
        assert_eq!(v2.y, 0.5);
        assert_eq!(v2.z, 0.75);
    }
}
