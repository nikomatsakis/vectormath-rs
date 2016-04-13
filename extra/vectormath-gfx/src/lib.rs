//! Math Library for Computer Graphics

#[macro_use]
extern crate vectormath;

pub trait Float : vectormath::float::Float + PartialOrd + PartialEq {
}

impl Float for f32 {}
impl Float for f64 {}

pub use vectormath::{Rad, Deg, Vec3, Vec4, Pos3, Quat, Mat3, Mat4, Tfm3};
pub use vectormath::{min, max};

pub mod plane;
pub mod frustum;
pub mod camera;
