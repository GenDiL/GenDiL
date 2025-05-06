// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

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

template<
   typename KernelContext,
   typename Operator,
   typename Vector >
GENDIL_HOST_DEVICE
auto ConjugateGradient(
   const KernelContext & kernel_conf,
   const Operator & A,
   const Vector & rhs,
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
   residual = residual - tmp; // initial residual

   Real rhs_norm = Norml2( kernel_conf, rhs );
   if( rhs_norm == 0 )
   {
      x = 0.0;
      iters = 0;
      tol_error = 0.0;
      return Result{ true, iters, tol_error };
   }
   Real residual_norm = Norml2( kernel_conf, residual );
   if( residual_norm < tol * rhs_norm )
   {
      iters = 0;
      tol_error = Sqrt( residual_norm / rhs_norm );
      return Result{ true, iters, tol_error };
   }

   TmpVector p = residual; // initial search direction

   Real abs_new = Dot( kernel_conf, residual, p );  // the square of the absolute value of r scaled by invM
   Integer i = 0;
   while( i < max_iters )
   {
      A( p, tmp );

      Real alpha = abs_new / Dot( kernel_conf, p, tmp );       // the amount we travel on dir
      x = x + alpha * p;                             // update solution
      residual = residual - alpha * tmp;                    // update residual

      residual_norm = Norml2( kernel_conf, residual );
      if( residual_norm < tol * rhs_norm )
      {
         tol_error = Sqrt( residual_norm / rhs_norm );
         iters = i;
         return Result{ true, iters, tol_error };
      }

      z = residual; // approximately solve for "A z = residual"

      Real abs_old = abs_new;
      abs_new = Dot( kernel_conf, residual, z );     // update the absolute value of r
      Real beta = abs_new / abs_old;    // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                 // update search direction
      i++;
   }
   tol_error = Sqrt( residual_norm / rhs_norm );
   iters = i;
   return Result{ false, iters, tol_error };
}

// Preconditionned conjugate gradient algorithm
template<
   typename Operator,
   typename Rhs,
   typename Preconditioner,
   typename Vector >
GENDIL_HOST_DEVICE
auto ConjugateGradient(
   const Operator & A,
   const Rhs & rhs,
   const Preconditioner & P,
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

   TmpVector p;
   P( residual, p );      // initial search direction

   Real abs_new = Dot( residual, p );  // the square of the absolute value of r scaled by invM
   Integer i = 0;
   while( i < max_iters )
   {
      A( p, tmp );

      Real alpha = abs_new / Dot( p, tmp );        // the amount we travel on dir
      x += alpha * p;                             // update solution
      residual -= alpha * tmp;                    // update residual

      residual_norm = Norml2( residual );
      if( residual_norm < tol * rhs_norm )
      {
         tol_error = Sqrt( residual_norm / rhs_norm );
         return Result{ true, iters, tol_error };
      }

      P( residual, z );                // approximately solve for "A z = residual"

      Real abs_old = abs_new;
      abs_new = Dot( residual, z );     // update the absolute value of r
      Real beta = abs_new / abs_old;     // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                // update search direction
      i++;
   }
   tol_error = Sqrt( residual_norm / rhs_norm );
   iters = i;
   return Result{ false, iters, tol_error };
}

}
