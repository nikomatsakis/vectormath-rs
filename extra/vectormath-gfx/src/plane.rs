//! Plane

use vectormath::float::Float;
use ::Pos3;

/// Plane equation: ax + by + cz + d = 0
#[derive(Debug, Copy, Clone)]
pub struct Plane<T>(pub T, pub T, pub T, pub T);

impl<T: Float> Plane<T> {
    /// Compute signed distance to a point
    pub fn distance_to(self, pt: Pos3<T>) -> T {
        self.0 * pt.x + self.1 * pt.y + self.2 * pt.z + self.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::Pos3;

    #[test]
    fn test_distance() {
        let pl = Plane(1.0, 2.0, 3.0, 4.0);
        assert_eq!(pl.distance_to(Pos3 { x: 0.5, y: 2.5, z: 1.5 }), 14.0);
        assert_eq!(pl.distance_to(Pos3 { x: 1.0, y: 0.5, z: -2.0 }), 0.0);
        assert_eq!(pl.distance_to(Pos3 { x: -1.0, y: -1.0, z: -1.0 }), -2.0);
    }
}
