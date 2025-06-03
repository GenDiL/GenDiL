// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <functional>
#include "gendil/Utilities/types.hpp"
#include <iostream>
namespace gendil {

// Unpreconditionned conjugate gradient algorithm
template<
   typename Operator,
   typename Rhs,
   typename Vector >
GENDIL_HOST_DEVICE
auto ConjugateGradient(
   const Operator & A,
   const Rhs & rhs,
   const Integer & max_iters,
   const Real & tol,
   Vector & x )
{
   using Result = std::tuple< bool, Integer, Real >; // Success, iteration, tolerance error
   using TmpVector = Vector; // TODO: Make some trait
   // TODO: Allocate all vectors here? Provide allocator? Some kind of memory space?
   
   Integer iters;
   TmpVector tmp, z;

   Real tol_error;

   TmpVector residual = rhs;
   A( x, tmp );
   residual -= tmp; // initial residual

   Real rhs_norm = Norml2( rhs );
   if( rhs_norm == 0 )
   {
      x = 0.0;
      iters = 0;
      tol_error = 0.0;
      return Result{ true, iters, tol_error };
   }
   Real residual_norm = Norml2( residual );
   if ( residual_norm < tol * rhs_norm )
   {
      iters = 0;
      tol_error = Sqrt( residual_norm / rhs_norm );
      return Result{ true, iters, tol_error };
   }

   TmpVector p = residual; // initial search direction

   Real abs_new = Dot( residual, p );  // the square of the absolute value of r scaled by invM
   Integer i = 0;
   while( i < max_iters )
   {
      A( p, tmp );

      Real alpha = abs_new / Dot( p, tmp );       // the amount we travel on dir
      x += alpha * p;                             // update solution
      residual -= alpha * tmp;                    // update residual

      residual_norm = Norml2( residual );
      if( residual_norm < tol * rhs_norm )
      {
         tol_error = Sqrt( residual_norm / rhs_norm );
         iters = i;
         return Result{ true, iters, tol_error };
      }

      z = residual; // approximately solve for "A z = residual"

      Real abs_old = abs_new;
      abs_new = Dot( residual, z );     // update the absolute value of r
      Real beta = abs_new / abs_old;    // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                 // update search direction
      i++;
   }
   tol_error = Sqrt( residual_norm / rhs_norm );
   iters = i;
   return Result{ false, iters, tol_error };
}

// template<
//    typename KernelContext,
//    typename Operator,
//    typename Vector >
// GENDIL_HOST_DEVICE
// auto ConjugateGradient(
//    const KernelContext & kernel_conf,
//    const Operator & A,
//    const Vector & rhs,
//    const Integer & max_iters,
//    const Real & tol,
//    Vector & x )
// {
//    using Result = std::tuple< bool, Integer, Real >; // Success, iteration, tolerance error
//    using TmpVector = Vector; // TODO: Make some trait
//    // TODO: Allocate all vectors here? Provide allocator? Some kind of memory space?
   
//    Integer iters;
//    TmpVector tmp, z;

//    Real tol_error;

//    TmpVector residual = rhs;
//    A( x, tmp );
//    residual = residual - tmp; // initial residual

//    Real rhs_norm = Norml2( kernel_conf, rhs );
//    if( rhs_norm == 0 )
//    {
//       x = 0.0;
//       iters = 0;
//       tol_error = 0.0;
//       return Result{ true, iters, tol_error };
//    }
//    Real residual_norm = Norml2( kernel_conf, residual );
//    if( residual_norm < tol * rhs_norm )
//    {
//       iters = 0;
//       tol_error = Sqrt( residual_norm / rhs_norm );
//       return Result{ true, iters, tol_error };
//    }

//    TmpVector p = residual; // initial search direction

//    Real abs_new = Dot( kernel_conf, residual, p );  // the square of the absolute value of r scaled by invM
//    Integer i = 0;
//    while( i < max_iters )
//    {
//       A( p, tmp );

//       Real alpha = abs_new / Dot( kernel_conf, p, tmp );       // the amount we travel on dir
//       x = x + alpha * p;                             // update solution
//       residual = residual - alpha * tmp;                    // update residual

//       residual_norm = Norml2( kernel_conf, residual );
//       if( residual_norm < tol * rhs_norm )
//       {
//          tol_error = Sqrt( residual_norm / rhs_norm );
//          iters = i;
//          return Result{ true, iters, tol_error };
//       }

//       z = residual; // approximately solve for "A z = residual"

//       Real abs_old = abs_new;
//       abs_new = Dot( kernel_conf, residual, z );     // update the absolute value of r
//       Real beta = abs_new / abs_old;    // calculate the Gram-Schmidt value used to create the new search direction
//       p = z + beta * p;                 // update search direction
//       i++;
//    }
//    tol_error = Sqrt( residual_norm / rhs_norm );
//    iters = i;
//    return Result{ false, iters, tol_error };
// }

// Preconditionned conjugate gradient algorithm
// template<
//    typename Operator,
//    typename Rhs,
//    typename Preconditioner,
//    typename Vector >
// GENDIL_HOST_DEVICE
// auto ConjugateGradient(
//    const Operator & A,
//    const Rhs & rhs,
//    const Preconditioner & P,
//    const Integer & max_iters,
//    const Real & tol,
//    Vector & x )
// {
//    using Result = std::tuple< bool, Integer, Real >; // Success, iteration, tolerance error
//    using TmpVector = Vector; // TODO: Make some trait
//    // TODO: Allocate all vectors here? Provide allocator? Some kind of memory space?
   
//    Integer iters;
//    TmpVector tmp, z;

//    Real tol_error;

//    TmpVector residual = rhs;
//    A( x, tmp );
//    residual -= tmp; // initial residual

//    Real rhs_norm = Norml2( rhs );
//    if( rhs_norm == 0 )
//    {
//       x = 0.0;
//       iters = 0;
//       tol_error = 0.0;
//       return Result{ true, iters, tol_error };
//    }
//    Real residual_norm = Norml2( residual );
//    if ( residual_norm < tol * rhs_norm )
//    {
//       iters = 0;
//       tol_error = Sqrt( residual_norm / rhs_norm );
//       return Result{ true, iters, tol_error };
//    }

//    TmpVector p;
//    P( residual, p );      // initial search direction

//    Real abs_new = Dot( residual, p );  // the square of the absolute value of r scaled by invM
//    Integer i = 0;
//    while( i < max_iters )
//    {
//       A( p, tmp );

//       Real alpha = abs_new / Dot( p, tmp );        // the amount we travel on dir
//       x += alpha * p;                             // update solution
//       residual -= alpha * tmp;                    // update residual

//       residual_norm = Norml2( residual );
//       if( residual_norm < tol * rhs_norm )
//       {
//          tol_error = Sqrt( residual_norm / rhs_norm );
//          return Result{ true, iters, tol_error };
//       }

//       P( residual, z );                // approximately solve for "A z = residual"

//       Real abs_old = abs_new;
//       abs_new = Dot( residual, z );     // update the absolute value of r
//       Real beta = abs_new / abs_old;     // calculate the Gram-Schmidt value used to create the new search direction
//       p = z + beta * p;                // update search direction
//       i++;
//    }
//    tol_error = Sqrt( residual_norm / rhs_norm );
//    iters = i;
//    return Result{ false, iters, tol_error };
// }

/**
 * @brief Solve A x = rhs using the Conjugate Gradient method.
 *
 * This implementation uses pre‑allocated work vectors and an injected
 * dot‑product function, so it remains agnostic to any parallel/MPI details.
 *
 * @tparam Vector    Mutable vector type (e.g. `std::vector<Real>`)
 *
 * @param[in]  A         Linear operator functor: tmp = A * p
 * @param[in]  rhs       Right‑hand side vector
 * @param[in]  dot       Dot‑product function (`std::function<Real(const Vector&,const Vector&)>`)
 *                       Should perform any necessary global reductions.
 * @param[in]  max_iters Maximum number of iterations
 * @param[in]  tol       Convergence tolerance (relative to ‖rhs‖)
 * @param[in,out] x      On entry: initial guess. On exit: approximate solution.
 * @param[in,out] tmp    Pre‑allocated workspace of same size as x
 * @param[in,out] z      Pre‑allocated workspace of same size as x
 * @param[in,out] residual Pre‑allocated workspace of same size as x
 * @param[in,out] p      Pre‑allocated workspace of same size as x
 *
 * @return `std::tuple<bool, Integer, Real>`  
 *         - bool:    `true` if converged, `false` if max_iters reached  
 *         - Integer: number of iterations performed  
 *         - Real:    final relative residual `sqrt(‖r‖² / ‖rhs‖²)`
 */
template<
   typename Operator,
   typename RHSType,
   typename DotType,
   typename Vector
>
GENDIL_HOST_DEVICE
auto ConjugateGradient(
   // std::function< void(const Vector &, Vector &) > &  A,
   // std::function< void(const Vector &, Vector &) > &  P,
   const Operator & A,
   const RHSType & rhs,
   // std::function< Real(const Vector &, const Vector &) > & dot,
   const DotType & dot,
   const Integer max_iters,
   const Real tol,
   Vector & x,
   // scratch memory inputs
   Vector & tmp,
   Vector & z,
   Vector & residual,
   Vector & p )
{
   using Result = std::tuple<bool, Integer, Real>; // TODO use a struct!

   // Helper: 2‑norm via injected dot
   auto norm2 = [&](Vector const & v){
      return Sqrt( dot(v, v) );
   };

   // 1) r = rhs - A*x
   A(x, tmp);
   residual = rhs;
   residual -= tmp;

   // 2) Compute ‖rhs‖ for relative tolerance
   const Real rhs_n = norm2( residual );
   if ( rhs_n == 0.0 ) {
      // trivial solution if rhs is zero
      x = 0.0;
      return Result{ true, 0, 0.0 };
   }

   // 3) Check initial convergence
   Real r_n = norm2( residual );
   if (r_n < tol * rhs_n) {
      return Result{ true, 0, Sqrt(r_n/rhs_n) };
   }

   // 4) p = r
   // P( residual, p );
   p = residual;
   Real abs_new = dot( residual, p );  // (r, r)

   Integer iter = 0;
   while ( iter < max_iters )
   {
      // 5) tmp = A * p
      A(p, tmp);

      // 6) α = (r, r) / (p, A p)
      Real alpha = abs_new / dot( p, tmp );

      // 7) x += α p;  r -= α tmp
      Axpy( alpha, p, x);
      Axpy(-alpha, tmp, residual);

      // 8) Convergence check: ‖r‖ < tol * ‖rhs‖
      r_n = norm2( residual );
      if ( r_n < tol * rhs_n ) {
         return Result{ true, iter+1, Sqrt(r_n/rhs_n) };
      }

      // 9) z = r  (for future preconditioning)
      // P( residual, z );
      z = residual;

      // 10) β = (r_new, z) / (r_old, r_old)
      Real abs_old = abs_new;
      abs_new = dot(residual, z);
      Real beta  = abs_new / abs_old;

      // 11) p = z + β p
      p *= beta;
      p += z;

      ++iter;
   }

   // 12) No convergence within max_iters
   return Result{ false, iter, Sqrt(r_n/rhs_n) };
}

template<
   typename KernelContext,
   typename Operator,
   typename RHSVector,
   typename Vector >
GENDIL_HOST_DEVICE
auto ConjugateGradient(
   const KernelContext & kernel_conf,
   const Operator & A,
   const RHSVector & rhs,
   const Integer & max_iters,
   const Real & tol,
   Vector & x )
{
   // std::function< void(const Vector &, Vector &) > A_op = [&]( const Vector & x, Vector & y )
   auto A_op = [&]( const Vector & x, Vector & y )
   {
      A( x, y );
   };

   // std::function< Real(const Vector &, const Vector &) > dot = [&]( const Vector & u, const Vector & v )
   auto dot = [&]( const Vector & u, const Vector & v )
   {
      return Dot( kernel_conf, u, v );
   };

   Vector tmp, z, residual, p;

   return ConjugateGradient( A_op, rhs, dot, max_iters, tol, x, tmp, z, residual, p );
}

}
