use typenum::*;
use std::ops::*;
use generic_array::*;
use alga::general::*;
use typenum::consts::*;

#[derive(Clone, Debug)]
pub struct Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{

        pub ar : GenericArray<T, typenum::Prod<N,M>>,
}


impl<'a, T : Clone,N,M> Not for &'a Mat<T,N,M> where N : Mul<M>, Prod<N,M> : ArrayLength<T>{
    type Output = Mat<T,N,M>;
    fn not(self) -> Self::Output{
        Mat{ar : self.ar.clone()}
    }
}

pub type Vect<T, N> = Mat<T,N,U1>;
pub type Vect2<T> = Vect<T,U2>;
pub type Vect3<T> = Vect<T,U3>;
pub type Vect4<T> = Vect<T,U4>;
pub type Mat4<T> = Mat<T,U4,U4>;


impl <T : Identity<Additive>, N, M> Mat<T,N,M> where
    N : Mul<M>,
    typenum::Prod<N,M> : ArrayLength<T>,{
    

    fn new_empty() -> Mat<T,N,M>{
        let ar = GenericArray::generate(|_| T::identity());
        Mat{ar}
    }
}

impl<T : Copy> Mat4<T>{
    pub fn new(slice : [T;16]) -> Vect2<T>{
        Vect::<T, U2>{ar : GenericArray::<T, U2>::clone_from_slice(&slice)}
    }

}

impl<T : Copy> Vect2<T>{
    pub fn new(x : T, y : T) -> Vect2<T>{
        Vect::<T, U2>{ar : GenericArray::<T, U2>::clone_from_slice(&[x,y])}
    }

    pub fn x(&self) -> T{
        self.ar[0]
    }

    pub fn y(&self) -> T{
        self.ar[1]
    }
}

impl<T : Copy> Vect3<T>{
    pub fn new(x : T, y : T, z : T) -> Vect3<T>{
        Vect::<T, U3>{ar : GenericArray::<T, U3>::clone_from_slice(&[x,y,z])}
    }
    

    pub fn x(&self) -> T{
        self.ar[0]
    }

    pub fn y(&self) -> T{
        self.ar[1]
    }

    pub fn z(&self) -> T{
        self.ar[2]
    }

    
}

impl<T : Copy + Real> Vect3<T>{
    
    pub fn w_one(&self) -> Vect4<T>{
        Vect4::from_xyz(self, T::one())
    } 

    
}


impl<T : Copy> Vect4<T>{
    pub fn new(x : T, y : T, z : T, w : T) -> Vect4<T>{
        Vect::<T, U4>{ar : GenericArray::<T, U4>::clone_from_slice(&[x,y,z,z])}
    }

    pub fn from_xyz(xyz : &Vect3<T>, w : T) -> Vect4<T>{
        Vect::<T, U4>{ar : GenericArray::<T, U4>::clone_from_slice(&[xyz.x(),xyz.y(),xyz.z(),w])}
    }
    
    pub fn xyz(&self) -> Vect3<T>{
        Vect3::new(self.x(), self.y(), self.z())
    }


    pub fn x(&self) -> T{
        self.ar[0]
    }

    pub fn y(&self) -> T{
        self.ar[1]
    }

    pub fn z(&self) -> T{
        self.ar[2]
    }

    pub fn w(&self) -> T{
        self.ar[3]
    }
}

impl<T : Copy + Mul<Output=T> + Add<Output=T> + Sub<Output=T>> Vect3<T>{

    pub fn cross(&self, other : &Vect3<T>) -> Vect3<T>{
        Vect3::new(self.y() * other.z() - other.y() * self.z(), other.x() * self.z() -self.x() * other.z(), self.x() * other.y() - other.x() * self.y())
    }
}

impl <T, N, M> Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{
    

    pub fn get(&self, i : usize) -> &T{
        &self.ar[i]
    }

    
}

impl <T : Real + Copy, N, M : Unsigned> Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>, M : Mul<U1>, Prod<M,U1> : ArrayLength<T>{
    

    pub fn row(&self, i : usize) -> Vect<T,M>{
        let mut ar = GenericArray::<T, Prod<M,U1>>::generate(|_| T::zero());
        let m = M::to_usize();
        for j in 0..m{
            ar[j] = self.ar[i * m + j];
        }

        Mat{ar}

    }
}

impl <T : Real + Copy, N : Unsigned, M : Unsigned> Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>, N : Mul<U1>, Prod<N,U1> : ArrayLength<T>{
    

    pub fn column(&self, j : usize) -> Vect<T,N>{
        let mut ar = GenericArray::<T, Prod<N,U1>>::generate(|_| T::zero());
        let n = N::to_usize();
        let m = M::to_usize();
        for i in 0..n{
            ar[i] = self.ar[i * m + j];
        }

        Mat{ar}

    }
}

impl <T : Clone + Copy + Real + Mul<T>, N, M> Mul<T> for Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{
    
    type Output = Mat<T,N,M>;

    fn mul(self, k : T) -> Mat<T,N,M> {
        let ar = self.ar.map(|x| x * k);
        Mat{ar}
    }

}

impl <T : Copy + Real, N : Unsigned, M : Unsigned, P : Unsigned> Mul<Mat<T,M,P>> for Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,
     M : Mul<P>,
      Prod<M,P> : ArrayLength<T>,
       N : Mul<P>, Prod<N,P> : ArrayLength<T>,
       M : Mul<U1>, Prod<M,U1> : ArrayLength<T>,
       N : Mul<U1>, Prod<N,U1> : ArrayLength<T>{
    
    type Output = Mat<T,N,P>;

    fn mul(self, other : Mat<T,M,P>) -> Mat<T,N,P> {
        let mut ar = GenericArray::<T, Prod<N,P>>::generate(|_| T::zero());

        let n = N::to_usize();
        let p = P::to_usize();

        for i in 0..n{
            for j in 0..p{
                ar[i * p + j] = self.row(i).dot(&other.column(j));
            }
        }

        Mat{ar}
    }

}


impl <'a, 'b, T : Copy + Real, N : Unsigned, M : Unsigned, P : Unsigned> Mul<&'b Mat<T,M,P>> for &'a Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,
     M : Mul<P>,
      Prod<M,P> : ArrayLength<T>,
       N : Mul<P>, Prod<N,P> : ArrayLength<T>,
       M : Mul<U1>, Prod<M,U1> : ArrayLength<T>,
       N : Mul<U1>, Prod<N,U1> : ArrayLength<T>{
    
    type Output = Mat<T,N,P>;

    fn mul(self, other : & 'b Mat<T,M,P>) -> Mat<T,N,P> {
        let mut ar = GenericArray::<T, Prod<N,P>>::generate(|_| T::zero());

        let n = N::to_usize();
        let p = P::to_usize();

        for i in 0..n{
            for j in 0..p{
                ar[i * p + j] = self.row(i).dot(&other.column(j));
            }
        }

        Mat{ar}
    }

}

impl <'a, T : Copy + Real, N, M> Mul<T> for &'a Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{
    
    type Output = Mat<T,N,M>;

    fn mul(self, k : T) -> Mat<T,N,M> {
        let ar = self.ar.map_ref(|x| x.clone() * k);
        Mat{ar}
    }

}

impl <T : Copy + Real, N, M> Div<T> for Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{
    
    type Output = Mat<T,N,M>;

    fn div(self, k : T) -> Mat<T,N,M> {
        let ar = self.ar.map(|x| x / k);
        Mat{ar}
    }
    
}

impl <'b, T : Copy + Real, N, M> Div<T> for &'b Mat<T,N,M> where
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{
    
    type Output = Mat<T,N,M>;

    fn div(self, k : T) -> Mat<T,N,M> {
        let ar = self.ar.map_ref(|x| x.clone() / k);
        Mat{ar}
    }
    
}

impl<T,N> Index<usize> for Vect<T,N> where
    N : Mul<U1>,
    Prod<N,U1> : ArrayLength<T>,{

    type Output = T;

    fn index(&self, i : usize) -> &T{
        &self.ar[i]
    }

}


impl<T,N> IndexMut<usize> for Vect<T,N> where
    N : Mul<U1>,
    Prod<N,U1> : ArrayLength<T>,{


    fn index_mut(&mut self, i : usize) -> &mut T{
        &mut self.ar[i]
    }

}

impl<T,N,M> Index<(usize, usize)> for Mat<T,N,M> where
    N : Unsigned,
    M : Unsigned,
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{

    type Output = T;

    fn index(&self, i : (usize, usize)) -> &T{
        &self.ar[i.0 * N::to_usize() + i.1]
    }

}

impl<T,N,M> IndexMut<(usize, usize)> for Mat<T,N,M> where
    N : Unsigned,
    M : Unsigned,
    N : Mul<M>,
    Prod<N,M> : ArrayLength<T>,{


    fn index_mut(&mut self, i : (usize, usize)) -> &mut T{
        &mut self.ar[i.0 * N::to_usize() + i.1]
    }

}

impl<'a,
     'b,
     T : Add<Output=T> + Copy,
     N,
     M>
     
     Add<& 'b Mat<T,N,M>>
    
for & 'a Mat<T,N,M> where N : Mul<M>, Prod<N,M> : ArrayLength<T>{
    
    type Output = Mat<T,N,M>;

    fn add(self, other : & 'b Mat<T,N,M>) -> Mat<T,N,M>{
        Mat{ar : GenericArray::<T, Prod<N, M>>::
              generate(&|i| self.get(i).clone() + other.get(i).clone())}
    }

}

impl<
     T : Add<Output=T> + Copy,
     N,
     M>
     
     Add<Mat<T,N,M>>
    
for Mat<T,N,M> where N : Mul<M>, Prod<N,M> : ArrayLength<T>{
    
    type Output = Mat<T,N,M>;

    fn add(self, other : Mat<T,N,M>) -> Mat<T,N,M>{
        Mat{ar : GenericArray::<T, Prod<N, M>>::
              generate(&|i| self.get(i).clone() + other.get(i).clone())}
    }

}

impl<'a,
     'b,
     T : Sub<Output=T> + Copy,
     N,
     M>
     
     Sub<& 'b Mat<T,N,M>>
    
for & 'a Mat<T,N,M> where N : Mul<M>, Prod<N,M> : ArrayLength<T>{
    
    type Output = Mat<T,N,M>;

    fn sub(self, other : & 'b Mat<T,N,M>) -> Mat<T,N,M>{
        Mat{ar : GenericArray::<T, Prod<N, M>>::
              generate(&|i| self.get(i).clone() - other.get(i).clone())}
    }


}

impl<
     T : Sub<Output=T> + Copy,
     N,
     M>
     
     Sub<Mat<T,N,M>>
    
for Mat<T,N,M> where N : Mul<M>, Prod<N,M> : ArrayLength<T>{
    
    type Output = Mat<T,N,M>;

    fn sub(self, other : Mat<T,N,M>) -> Mat<T,N,M>{
        Mat{ar : GenericArray::<T, Prod<N, M>>::
              generate(&|i| self.get(i).clone() - other.get(i).clone())}
    }


}

impl<
     T : Mul<Output=T> + Add<Output=T> + Copy + AbstractMonoid<Additive>,
     N>
     
Vect<T,N> where N : Mul<U1> + Unsigned, Prod<N,U1> : ArrayLength<T>{
    

    pub fn dot(&self, other : &Vect<T,N>) -> T{
        let mut res = T::identity();
        for i in 0..<N as Unsigned>::to_usize(){
            res = res + self.ar[i] * other.ar[i];
        }

        res
    }

}

impl<
     T : Neg<Output=T> + Copy,
     N, M>

     Neg for
     
Mat<T,N,M> where N : Mul<M> + Unsigned, Prod<N,M> : ArrayLength<T>{
    

    type Output = Mat<T,N,M>;

    fn neg(self) -> Mat<T,N,M> {
        let ar = self.ar.map(|x| -x);
        Mat{ar}
    }

}

impl<
     T : Copy + Real,
     N : Unsigned, M : Unsigned>

     
Mat<T,N,M> where N : Mul<M> + Unsigned, Prod<N,M> : ArrayLength<T>, M : Mul<N> + Unsigned, Prod<M,N> : ArrayLength<T>{
    

   pub fn transpose(&self) -> Mat<T,M,N>{
       let mut ar = GenericArray::<T, Prod<M,N>>::generate(|_| T::zero());
       for i in 0..N::to_usize(){
           for j in 0..M::to_usize(){
               ar[j * N::to_usize() + i] = self.ar[i * M::to_usize() + j]
           }
       }

       Mat{ar}
   }

}

impl<
     T : Real + Copy,
     N>

     
Vect<T,N> where N : Mul<U1> + Unsigned, Prod<N,U1> : ArrayLength<T>{
    
    pub fn norm(&self) -> T{
        let mut acc = T::zero();
        for el in self.ar.iter(){
            acc = acc + el.clone() * el.clone();
        }

        acc.sqrt()
    }

    pub fn normalize(&self) -> Vect<T,N> {
        self * (T::one() / self.norm())
    }

}

macro_rules! vec2 {
    ( $x:expr , $y:expr) => {
        {
            Vect2::new($x, $y)
        }
    };
}

macro_rules! vec3 {
    ( $x:expr , $y:expr, $z:expr ) => {
        {
            Vect3::new($x, $y, $z)
        }
    };
}

pub fn test_matrices(){
    let a = Vect::<_, U3>{ar : arr!(i32; 1,0,0)};
    let b = Vect::<_, U3>{ar : arr!(i32; -1,0,0)};
    let c = (&a) + (&b);

    let mut d = vec3!(1,2,3);

    let i = vec3!(1,0,0);
    let j = vec3!(0,1,0);
    
    d[1] = 0;
    d[(0,1)] = 0;

    let a1 = a;

    println!("{:?} + {:?} = {:?}", a, b, c);
    println!("{}", (&d).dot(&d));
    println!("{:?}", (&i).cross(&j));

    //TODO macros for creation
}