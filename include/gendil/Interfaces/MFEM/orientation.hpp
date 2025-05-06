// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"

namespace gendil {

/**
 * @brief Structure to represent the orientation between two cells using MFEM information.
 * 
 * @tparam Dim The dimension of the mfem::Mesh (1 to 3).
 */
template < Integer Dim >
struct mfem_orientation
{
   int element_face_id;
   int neighbor_face_id;
   int face_orientation;
};

// !FIXME is there two possible cases?
Permutation<1> TranslateMFEMOrientation( mfem_orientation<1> const & orientation )
{
   // We start from 1 to be able to use -
   Permutation<1> basis{ { 1 } };

   return basis;
}

/**
 * @brief Returns the face orientation based on the element orientation and the mfem_element_face_id.
 * The element coordinate system is the reference coordinate system. The face coordinate system is
 * ( eta, normal ), eta being the tangential coordinate system, the normal is the outward normal on the face.
 * 
 * @param element_orientation 
 * @param mfem_element_face_id 
 * @return Permutation<2> 
 */
Permutation<2> ToFaceOrientation( Permutation<2> element_orientation, int mfem_element_face_id )
{
   Permutation<2> face_orientation = element_orientation;
   switch (mfem_element_face_id)
   {
   case 0: // BOTTOM
      face_orientation( 0 ) = element_orientation( 0 );
      face_orientation( 1 ) = -element_orientation( 1 );
      break;
   case 1: // RIGHT
      face_orientation( 0 ) = element_orientation( 1 );
      face_orientation( 1 ) = element_orientation( 0 );
      break;
   case 2: // TOP
      face_orientation( 0 ) = -element_orientation( 0 );
      face_orientation( 1 ) = element_orientation( 1 );
      break;
   case 3: // LEFT
      face_orientation( 0 ) = -element_orientation( 1 );
      face_orientation( 1 ) = -element_orientation( 0 );
      break;
   
   default:
      // error
      break;
   }
   return face_orientation;
}

/**
 * @brief Returns the neighbor face orientation based on the face orientation.
 * 
 * @param face_orientation 
 * @param mfem_face_orientation 
 * @return Permutation<2> 
 */
Permutation<2> ToNeighborFaceOrientation( Permutation<2> face_orientation, int mfem_face_orientation )
{
   Permutation<2> neighbor_face_orientation = face_orientation;
   // Check mfem geom.cpp file for definition of these face orientations.
   switch (mfem_face_orientation)
   {
   case 0:
      neighbor_face_orientation( 0 ) = face_orientation( 0 );
      neighbor_face_orientation( 1 ) = face_orientation( 1 );
      break;
   case 1:
      neighbor_face_orientation( 0 ) = -face_orientation( 0 ); // !VERIFY
      neighbor_face_orientation( 1 ) = -face_orientation( 1 ); // !VERIFY
      break;
   
   default:
      // error
      break;
   }
   return neighbor_face_orientation;
}

/**
 * @brief Returns the element orientation based on a face orientation and the mfem_face_id.
 * 
 * @param face_orientation 
 * @param mfem_face_id 
 * @return Permutation<2> 
 */
Permutation<2> ToElementOrientation( Permutation<2> face_orientation, int mfem_face_id )
{
   Permutation<2> orientation = face_orientation;
   switch (mfem_face_id)
   {
   case 0: // BOTTOM
      orientation( 0 ) = face_orientation( 0 );
      orientation( 1 ) = -face_orientation( 1 );
      break;
   case 1: // RIGHT
      orientation( 0 ) = face_orientation( 1 );
      orientation( 1 ) = face_orientation( 0 );
      break;
   case 2: // TOP
      orientation( 0 ) = -face_orientation( 0 );
      orientation( 1 ) = face_orientation( 1 );
      break;
   case 3: // LEFT
      orientation( 0 ) = -face_orientation( 1 );
      orientation( 1 ) = -face_orientation( 0 );
      break;
   
   default:
      break;
   }
   return orientation;
}

Permutation<2> TranslateMFEMOrientation( mfem_orientation<2> const & orientation )
{
   // We start from 1 to be able to use -
   Permutation<2> basis{ { 1, 2 } };

   // return basis;
   return   ToElementOrientation(
               ToNeighborFaceOrientation(
                  ToFaceOrientation( basis, orientation.element_face_id ),
                  orientation.face_orientation ),
               orientation.neighbor_face_id );
}

/**
 * @brief Returns the face orientation based on the element orientation and the mfem_element_face_id.
 * The element coordinate system is the reference coordinate system. The face coordinate system is ( eta, theta, normal ),
 * where eta ^ theta = normal, (eta, theta) being the tangential coordinate system, the normal is the outward normal on the face.
 * 
 * @param element_orientation 
 * @param mfem_element_face_id 
 * @return Permutation<3> 
 */
Permutation<3> ToFaceOrientation( Permutation<3> element_orientation, int mfem_element_face_id )
{
   Permutation<3> face_orientation = element_orientation;
   switch (mfem_element_face_id)
   {
   case 0: // BOTTOM
      face_orientation( 0 ) = element_orientation( 0 );
      face_orientation( 1 ) = -element_orientation( 1 );
      face_orientation( 2 ) = -element_orientation( 2 );
      break;
   case 1: // FRONT
      face_orientation( 0 ) = element_orientation( 0 );
      face_orientation( 1 ) = element_orientation( 2 );
      face_orientation( 2 ) = -element_orientation( 1 );
      break;
   case 2: // RIGHT
      face_orientation( 0 ) = element_orientation( 1 );
      face_orientation( 1 ) = element_orientation( 2 );
      face_orientation( 2 ) = element_orientation( 0 );
      break;
   case 3: // BACK
      face_orientation( 0 ) = -element_orientation( 0 );
      face_orientation( 1 ) = element_orientation( 2 );
      face_orientation( 2 ) = element_orientation( 1 );
      break;
   case 4: // LEFT
      face_orientation( 0 ) = -element_orientation( 1 );
      face_orientation( 1 ) = element_orientation( 2 );
      face_orientation( 2 ) = -element_orientation( 0 );
      break;
   case 5: // TOP
      face_orientation( 0 ) = element_orientation( 0 );
      face_orientation( 1 ) = element_orientation( 1 );
      face_orientation( 2 ) = element_orientation( 2 );
      break;
   
   default:
      // error
      break;
   }
   return face_orientation;
}

/**
 * @brief Returns the neighbor face orientation based on the face orientation.
 * 
 * @param face_orientation 
 * @param mfem_face_orientation 
 * @return Permutation<3> 
 */
Permutation<3> ToNeighborFaceOrientation( Permutation<3> face_orientation, int mfem_face_orientation )
{
   Permutation<3> neighbor_face_orientation = face_orientation;
   // Check mfem geom.cpp file for definition of these face orientations.
   switch (mfem_face_orientation)
   {
   case 0:
      neighbor_face_orientation( 0 ) = face_orientation( 0 );
      neighbor_face_orientation( 1 ) = face_orientation( 1 );
      neighbor_face_orientation( 2 ) = face_orientation( 2 );
      break;
   case 1:
      neighbor_face_orientation( 0 ) = face_orientation( 1 );
      neighbor_face_orientation( 1 ) = face_orientation( 0 );
      neighbor_face_orientation( 2 ) = -face_orientation( 2 );
      break;
   case 2:
      neighbor_face_orientation( 0 ) = face_orientation( 1 );
      neighbor_face_orientation( 1 ) = -face_orientation( 0 );
      neighbor_face_orientation( 2 ) = face_orientation( 2 );
      break;
   case 3:
      neighbor_face_orientation( 0 ) = -face_orientation( 0 );
      neighbor_face_orientation( 1 ) = face_orientation( 1 );
      neighbor_face_orientation( 2 ) = -face_orientation( 2 );
      break;
   case 4:
      neighbor_face_orientation( 0 ) = -face_orientation( 0 );
      neighbor_face_orientation( 1 ) = -face_orientation( 1 );
      neighbor_face_orientation( 2 ) = face_orientation( 2 );
      break;
   case 5:
      neighbor_face_orientation( 0 ) = -face_orientation( 1 );
      neighbor_face_orientation( 1 ) = -face_orientation( 0 );
      neighbor_face_orientation( 2 ) = -face_orientation( 2 );
      break;
   case 6:
      neighbor_face_orientation( 0 ) = -face_orientation( 1 );
      neighbor_face_orientation( 1 ) = face_orientation( 0 );
      neighbor_face_orientation( 2 ) = face_orientation( 2 );
      break;
   case 7:
      neighbor_face_orientation( 0 ) = face_orientation( 0 );
      neighbor_face_orientation( 1 ) = -face_orientation( 1 );
      neighbor_face_orientation( 2 ) = -face_orientation( 2 );
      break;
   
   default:
      // error
      break;
   }
   return neighbor_face_orientation;
}

/**
 * @brief Returns the element orientation based on a face orientation and the mfem_face_id.
 * 
 * @param face_orientation 
 * @param mfem_face_id 
 * @return Permutation<3> 
 */
Permutation<3> ToElementOrientation( Permutation<3> face_orientation, int mfem_face_id )
{
   Permutation<3> orientation = face_orientation;
   switch (mfem_face_id)
   {
   case 0: // BOTTOM
      orientation( 0 ) = face_orientation( 0 );
      orientation( 1 ) = -face_orientation( 1 );
      orientation( 2 ) = -face_orientation( 2 );
      break;
   case 1: // FRONT
      orientation( 0 ) = face_orientation( 0 );
      orientation( 1 ) = -face_orientation( 2 );
      orientation( 2 ) = face_orientation( 1 );
      break;
   case 2: // RIGHT
      orientation( 0 ) = face_orientation( 2 );
      orientation( 1 ) = face_orientation( 0 );
      orientation( 2 ) = face_orientation( 1 );
      break;
   case 3: // BACK
      orientation( 0 ) = -face_orientation( 0 );
      orientation( 1 ) = face_orientation( 2 );
      orientation( 2 ) = face_orientation( 1 );
      break;
   case 4: // LEFT
      orientation( 0 ) = -face_orientation( 2 );
      orientation( 1 ) = -face_orientation( 0 );
      orientation( 2 ) = face_orientation( 1 );
      break;
   case 5: // TOP
      orientation( 0 ) = face_orientation( 0 );
      orientation( 1 ) = face_orientation( 1 );
      orientation( 2 ) = face_orientation( 2 );
      break;
   
   default:
      break;
   }
   return orientation;
}

/**
 * @brief Returns neighboring element orientation based of the face configuration:
 * ( mfem_element_face_id, mfem_neighbor_face_id, mfem_face_orientation ).
 * 
 * @param mfem_element_face_id 
 * @param mfem_neighbor_face_id 
 * @param mfem_face_orientation 
 * @return Permutation<3> 
 */
// Permutation<3> TranslateMFEMOrientation( int mfem_element_face_id, int mfem_neighbor_face_id, int mfem_face_orientation )
Permutation<3> TranslateMFEMOrientation( mfem_orientation<3> const & orientation )
{
   // std::cout
   //    << "face elem 0: " << mfem_element_face_id
   //    << ", face elem 1: " << mfem_neighbor_face_id
   //    << ", face orientation: " << mfem_face_orientation
   //    << std::endl;
   // We start from 1 to be able to use -
   Permutation<3> basis{ { 1, 2, 3 } };
   // std::cout << "To face: " << ToFaceOrientation( basis, mfem_neighbor_face_id );
   // std::cout << "To neighbor face: " << ToNeighborFaceOrientation( ToFaceOrientation( basis, mfem_neighbor_face_id ), mfem_face_orientation );
   // std::cout << "To element: ";

   return   ToElementOrientation(
               ToNeighborFaceOrientation(
                  ToFaceOrientation( basis, orientation.element_face_id ),
                  orientation.face_orientation ),
               orientation.neighbor_face_id );
   // return   ToElementOrientation(
   //             ToNeighborFaceOrientation(
   //                ToFaceOrientation( basis, mfem_neighbor_face_id ),
   //                mfem_face_orientation ),
   //             mfem_element_face_id );
}

}

#endif
