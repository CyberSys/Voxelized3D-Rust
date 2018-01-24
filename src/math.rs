use generic_array::ArrayLength;
use typenum::{Prod};
use alga::linear::FiniteDimInnerSpace;
use alga::general::SupersetOf;
use std::fmt::Debug;
use std;
use typenum;
use generic_array;
use rand::Rng;
use alga::general::Real;
use matrix::*;
use generic_array::*;

#[derive(Clone, Debug)]
pub struct Triangle2<T : Real + Copy>{
    pub p1: Vect2<T>,
    pub p2: Vect2<T>,
    pub p3: Vect2<T>,
}

#[derive(Clone, Debug)]
pub struct Triangle3<T : Real + Copy>{
    pub p1: Vect3<T>,
    pub p2: Vect3<T>,
    pub p3: Vect3<T>,
}

#[derive(Clone, Debug)]
pub struct Line2<T : Real + Copy> {
    pub start : Vect2<T>,
    pub end : Vect2<T>,
}

#[derive(Clone, Debug)]
pub struct Line3<T : Real + Copy> {
    pub start : Vect3<T>,
    pub end : Vect3<T>,
}

#[derive(Clone, Debug)]
pub struct Plane<T : Real + Copy> {
    pub point : Vect3<T>,
    pub normal : Vect3<T>,
}

//axis aligned
#[derive(Clone, Debug)]
pub struct Square2<T : Real>{
    pub center : Vect2<T>,
    pub extent : T,
}

//axis aligned
#[derive(Clone, Debug)]
pub struct Square3<T : Real>{
    pub center : Vect3<T>,
    pub extent : T,
}

#[derive(Clone, Debug)]
pub struct Sphere<T : Real>{
    pub center : Vect3<T>,
    pub rad : T,
}






pub type DenFn2<T> = Box<Fn(Vect2<T>) -> T>;
pub type DenFn3<T> = Box<Fn(Vect3<T>) -> T>;


pub fn intersection2<T : Real>(a : DenFn2<T>, b : DenFn2<T>) -> DenFn2<T>{
    Box::new(move |x|{Real::max(a(x), b(x))})
}

pub fn union2<T : Real>(a : DenFn2<T>, b : DenFn2<T>) -> DenFn2<T>{
    Box::new(move |x| {Real::min(a(x), b(x))})
}

pub fn difference2<T : Real>(a : DenFn2<T>, b : DenFn2<T>) -> DenFn2<T>{
    Box::new(move |x| {Real::max(a(x), -b(x))})
}

pub fn intersection3<T : Real>(a : DenFn3<T>, b : DenFn3<T>) -> DenFn3<T>{
    Box::new(move |x|{Real::max(a(x), b(x))})
}

pub fn union3<T : Real>(a : DenFn3<T>, b : DenFn3<T>) -> DenFn3<T>{
    Box::new(move |x| {Real::min(a(x), b(x))})
}

pub fn difference3<T : Real>(a : DenFn3<T>, b : DenFn3<T>) -> DenFn3<T>{
    Box::new(move |x| {Real::max(a(x), -b(x))})
}

pub fn mk_circle2<T : Real + Copy>(center : Vect2<T>, rad : T) -> DenFn2<T>{
    Box::new(move |x|{
        let dist = &x - &center;
        dist.dot(&dist) - rad * rad
    })
}

pub fn mk_half_plane2_left<T : Real + Copy>(x : T) -> DenFn2<T>{
    Box::new(move |p|{p.x() - x})
}

pub fn mk_half_plane2_right<T : Real + Copy>(x : T) -> DenFn2<T>{
    Box::new(move |p|{x - p.x()})
}


pub fn mk_half_plane2_lower<T : Real + Copy>(y : T) -> DenFn2<T>{
    Box::new(move |p|{p.y() - y})
}

pub fn mk_half_plane2_upper<T : Real + Copy>(y : T) -> DenFn2<T>{
    Box::new(move |p|{y - p.y()})
}



pub fn union3_mat<T : Real>(a : DenFn3<T>, b : DenFn3<T>) -> DenFn3<T>{
    Box::new(move |x| {Real::min(a(x), b(x))})
}

pub fn difference3_mat<T : Real>(a : DenFn3<T>, b : DenFn3<T>) -> DenFn3<T>{
    Box::new(move |x| {Real::max(a(x), -b(x))})
}

pub fn mk_half_space_x_neg<T : Real + Copy>(x : T) -> DenFn3<T>{
    Box::new(move |p|{p.x() - x})
}

pub fn mk_half_space_x_pos<T : Real + Copy>(x : T) -> DenFn3<T>{
    Box::new(move |p|{x - p.x()})
}

pub fn mk_half_space_y_neg<T : Real + Copy>(y : T) -> DenFn3<T>{
    Box::new(move |p|{p.y() - y})
}

pub fn mk_half_space_y_pos<T : Real + Copy>(y : T) -> DenFn3<T>{
    Box::new(move |p|{y - p.y()})
}

pub fn mk_half_space_z_neg<T : Real + Copy>(z : T) -> DenFn3<T>{
    Box::new(move |p|{p.z() - z})
}

pub fn mk_half_space_z_pos<T : Real + Copy>(z : T) -> DenFn3<T>{
    Box::new(move |p|{z - p.z()})
}


pub fn mk_rectangle2<T : Real + Copy>(center : Vect2<T>, extent : Vect2<T>) -> DenFn2<T> {
    let right = mk_half_plane2_right(center.x() - extent.x());
    let left = mk_half_plane2_left(center.x() + extent.x());

    let lower = mk_half_plane2_lower(center.y() + extent.y());
    let upper = mk_half_plane2_upper(center.y() - extent.y());

    let i1 = intersection2(left, right);
    let i2 = intersection2(upper, lower);

    intersection2(i1, i2)
}

pub fn mk_aabb<T : Real + Copy>(center : Vect3<T>, extent : Vect3<T>) -> DenFn3<T> {
    let x_neg = mk_half_space_x_neg(center.x() + extent.x());
    let x_pos = mk_half_space_x_pos(center.x() - extent.x());

    let y_neg = mk_half_space_y_neg(center.y() + extent.y());
    let y_pos = mk_half_space_y_pos(center.y() - extent.y());

    let z_neg = mk_half_space_z_neg(center.z() + extent.z());
    let z_pos = mk_half_space_z_pos(center.z() - extent.z());

    let ix = intersection3(x_neg, x_pos);
    let iy = intersection3(y_neg, y_pos);
    let iz = intersection3(z_neg, z_pos);

    let ixy = intersection3(ix, iy);

    intersection3(ixy, iz)
}

pub fn mk_sphere<T : Real + Copy>(sphere : Sphere<T>) -> DenFn3<T>{
    Box::new(move |x|{
        let dist = &x - &sphere.center;
        dist.dot(&dist) - sphere.rad * sphere.rad
    })
}


pub fn mk_sphere_displacement<'f, T : Real + Copy>(sphere : Sphere<T>, f : Box<Fn(Vect3<T>) -> T>) -> DenFn3<T>{
    Box::new(move |x|{
        let dist = &x - &sphere.center;
        dist.dot(&dist) - sphere.rad * sphere.rad * f(dist.normalize())
    })
}

pub fn distance_point2_line2<T : Real>(point2 : &Vect2<T>, line2 : &Line2<T>) -> T{
    let d = &line2.start - &line2.end;
    let norm = d.normalize();
    let n = Vect2::new(-norm.y(), norm.x());
    let vec = point2 - &line2.start;
    Real::abs(n.dot(&vec))
}

pub fn distance_point3_plane<T : Real>(point3 : &Vect3<T>, plane : &Plane<T>) -> T{
    let vec = point3 - &plane.point;
    Real::abs(plane.normal.dot(&vec))
}

pub fn point3_inside_square3_inclusive<T : Real>(point3 : &Vect3<T>, square3 : Square3<T>) -> bool{
    point3.x() <= square3.center.x() + square3.extent &&
    point3.x() >= square3.center.x() - square3.extent &&

    point3.y() <= square3.center.y() + square3.extent &&
    point3.y() >= square3.center.y() - square3.extent &&

    point3.z() <= square3.center.z() + square3.extent &&
    point3.z() >= square3.center.z() - square3.extent
}

pub fn vec3f_vec3d(a : Vect3<f64>) -> Vect3<f32>{
    Vect3::new(a.x() as f32, a.y() as f32, a.z() as f32)
}

//column-major
pub fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> [f32;16]{
    [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left),
     0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom),
     0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near),
     0.0, 0.0, 0.0, 1.0
    ]
}

//column-major
pub fn perspective(fovy : f32, aspect : f32, near : f32, far : f32) -> Mat4<f32>{
    let top = near * (std::f32::consts::PI / 180.0 * fovy / 2.0).tan();
    let bottom = -top;
    let right = top * aspect;
    let left = -right;
    Mat4{ar : arr![f32;2.0 * near / (right - left), 0.0, (right + left) / (right - left), 0.0,
                 0.0, 2.0 * near / (top - bottom), (top + bottom) / (top - bottom), 0.0,
                 0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near),
                 0.0, 0.0, -1.0, 0.0]}
}

//column-major
pub fn view_dir(pos : Vect3<f32>, look : Vect3<f32>, up : Vect3<f32>) -> Mat4<f32>{
    let za = -look;
    let xa = (&up).cross(&za);
    let ya = (&za).cross(&xa);

    Mat{ar : arr![f32;xa.x(), ya.x(), za.x(), 0.0,
                 xa.y(), ya.y(), za.y(), 0.0,
                 xa.z(), ya.z(), za.z(), 0.0,
                 -xa.dot(&pos), -ya.dot(&pos), -za.dot(&pos), 1.0]}.transpose()
}

//column-major, radians, axis must be a unit vector
pub fn rotation3(axis : &Vect3<f32>, angle : f32) -> Mat4<f32> {

    let cosphi = angle.cos();
    let sinphi = angle.sin();
    let ux = axis.x();
    let uy = axis.y();
    let uz = axis.z();

    Mat{ar :
        arr![f32;
            cosphi + ux * ux * (1.0 - cosphi), ux * uy * (1.0 - cosphi) - uz * sinphi, ux * uz * (1.0 - cosphi) + uy * sinphi, 0.0,
            uy * ux * (1.0 - cosphi) + uz * sinphi, cosphi + uy * uy * (1.0 - cosphi), uy * uz * (1.0 - cosphi) - ux * sinphi, 0.0,
            uz * ux * (1.0 - cosphi) - uy * sinphi, uz * uy * (1.0 - cosphi) + ux * sinphi, cosphi + uz * uz * (1.0 - cosphi), 0.0,
            0.0, 0.0, 0.0, 1.0
        ]
    }
}