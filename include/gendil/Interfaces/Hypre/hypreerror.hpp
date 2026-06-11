// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Interfaces/Hypre/hypretypes.hpp"
#include "gendil/Utilities/debug.hpp"

#include <string>

namespace gendil
{

inline std::string DescribeHypreError( const HYPRE_Int error )
{
   char message[256] = {};
   HYPRE_DescribeError( error, message );
   return std::string( message );
}

inline bool IsOnlyHypreConvergenceError( const HYPRE_Int error )
{
   return error != 0 &&
          HYPRE_CheckError( error, HYPRE_ERROR_CONV ) &&
          ( error & ~HYPRE_ERROR_CONV ) == 0;
}

inline void CheckHypreError(
   const HYPRE_Int error,
   const char * context )
{
   if ( error == 0 )
   {
      return;
   }

   const std::string message =
      std::string( context ) + ": " + DescribeHypreError( error );
   GENDIL_VERIFY( false, message.c_str() );
}

inline bool CheckHypreSolveError(
   const HYPRE_Int error,
   const char * context )
{
   if ( error == 0 )
   {
      return true;
   }

   if ( IsOnlyHypreConvergenceError( error ) )
   {
      HYPRE_ClearAllErrors();
      return false;
   }

   CheckHypreError( error, context );
   return false;
}

inline void RequireHypreInitialized( const char * context )
{
   GENDIL_VERIFY(
      HYPRE_Initialized(),
      context );
}

} // namespace gendil
