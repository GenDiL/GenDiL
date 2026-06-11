// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Interfaces/Hypre/hypreerror.hpp"

namespace gendil
{

namespace details
{

inline Integer hypre_session_ref_count = 0;
inline bool hypre_session_initialized_hypre = false;

} // namespace details

struct HypreSession
{
   HypreSession()
   {
      if ( details::hypre_session_ref_count == 0 && !HYPRE_Initialized() )
      {
         CheckHypreError(
            HYPRE_Initialize(),
            "HYPRE_Initialize failed" );
         details::hypre_session_initialized_hypre = true;
      }
      ++details::hypre_session_ref_count;
   }

   HypreSession( const HypreSession & ) = delete;
   HypreSession & operator=( const HypreSession & ) = delete;
   HypreSession( HypreSession && ) = delete;
   HypreSession & operator=( HypreSession && ) = delete;

   ~HypreSession()
   {
      if ( details::hypre_session_ref_count == 0 )
      {
         return;
      }

      --details::hypre_session_ref_count;
      if ( details::hypre_session_ref_count == 0 &&
           details::hypre_session_initialized_hypre &&
           HYPRE_Initialized() )
      {
         CheckHypreError(
            HYPRE_Finalize(),
            "HYPRE_Finalize failed" );
         details::hypre_session_initialized_hypre = false;
      }
   }
};

} // namespace gendil
