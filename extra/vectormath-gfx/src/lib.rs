//! Math Library for Computer Graphics

#[macro_use]
extern crate vectormath;

pub use vectormath::{Rad, Deg, Vec3, Vec4, Pos3, Quat, Mat3, Mat4, Tfm3};
pub use vectormath::{min, max};

pub mod plane;
pub mod frustum;
pub mod camera;
