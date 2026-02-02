// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

/**
 * @brief Face connectivity between two Cartesian meshes along a fixed interface.
 *
 * This type connects the face `LocalFaceIndex` of the "minus" mesh to the
 * opposite face on the "plus" mesh. The interface is orthogonal to
 * `axis = LocalFaceIndex % Dim`; tangential directions (all d ≠ axis)
 * must have identical grid sizes on both meshes.
 *
 * @tparam Dim            Topological dimension of the cells (e.g., 2 or 3).
 * @tparam LocalFaceIndex Compile-time local face id in `[0, 2*Dim)`.
 *                        `0..Dim-1` denote faces with reference normal `-e_axis`,
 *                        `Dim..2*Dim-1` denote faces with `+e_axis`.
 *
 * @par Data members
 *  - `sizes_minus[d]` : number of cells of the "minus" mesh along dimension @p d
 *  - `sizes_plus[d]`  : number of cells of the "plus"  mesh along dimension @p d
 *  - `num_faces`      : number of interface faces = ∏_{d≠axis} sizes_minus[d]
 *
 * @note The interface selection (which side is “minus/plus”) is determined by
 *       `LocalFaceIndex`. For `sign > 0` (a `+e_axis` face), the minus cells
 *       lie at the last slice along @p axis and the plus cells at the first slice;
 *       for `sign < 0` the roles are reversed.
 */
template <Integer Dim, Integer LocalFaceIndex>
struct CartesianIntermeshFaceConnectivity
{
   static_assert(0 <= LocalFaceIndex && LocalFaceIndex < 2*Dim,
                  "LocalFaceIndex out of range");

   using geometry         = HyperCube<Dim>;
   using orientation_type = IdentityOrientation<Dim>;

   // Derive interface axis and sign from the minus local face index
   static constexpr Integer axis = LocalFaceIndex % Dim;
   static constexpr int     sign = (LocalFaceIndex < Dim ? -1 : +1);

   // Minus = the provided local face; Plus = its opposite face
   static constexpr Integer minus_local_face_index = LocalFaceIndex;
   static constexpr Integer plus_local_face_index  =
      (LocalFaceIndex < Dim ? LocalFaceIndex + Dim : LocalFaceIndex - Dim);

   using minus_normal_type = CanonicalVector<Dim, axis,  sign>;
   using plus_normal_type  = CanonicalVector<Dim, axis, -sign>;

   using face_info_type =
      ConformingCellFaceView<
      geometry,
      std::integral_constant<Integer, minus_local_face_index>,
      std::integral_constant<Integer, plus_local_face_index>,
      orientation_type,
      minus_normal_type,
      plus_normal_type>;

   // Grid sizes: minus = first mesh, plus = second mesh (must match tangentially)
   std::array<GlobalIndex, Dim> sizes_minus{}; // #cells per dim in minus mesh
   std::array<GlobalIndex, Dim> sizes_plus{};  // #cells per dim in plus  mesh
   GlobalIndex num_faces{0};                   // product over d≠axis of tangential sizes

   /**
   * @brief Construct the inter-mesh connector from per-dimension grid sizes.
   *
   * Initializes the connector and checks that tangential sizes (all @p d ≠ @ref axis)
   * match between @p sizes_minus and @p sizes_plus. The number of interface faces
   * is computed as the product of tangential sizes.
   *
   * @param[in] sm  Grid sizes of the "minus" mesh (cells per dimension).
   * @param[in] sp  Grid sizes of the "plus"  mesh (cells per dimension).
   *
   * @pre For all @p d ≠ @ref axis: `sm[d] == sp[d]`.
   * @note The extent along @ref axis may differ between the two meshes.
   * @warning Uses `assert` for the tangential equality; replace with your
   *          preferred `GENDIL_ASSERT` if needed.
   *
   * @par Complexity
   * O(Dim) for the checks and the product.
   */
   GENDIL_HOST_DEVICE
   CartesianIntermeshFaceConnectivity(std::array<GlobalIndex, Dim> sm,
                                       std::array<GlobalIndex, Dim> sp)
   : sizes_minus(sm), sizes_plus(sp)
   {
      // Tangential sizes must match (geometry-conforming interface)
      for (int d = 0; d < Dim; ++d) {
         if (d == axis) continue;
         // GENDIL_VERIFY(sizes_minus[d] == sizes_plus[d]);
         assert(sizes_minus[d] == sizes_plus[d]);
      }
      num_faces = 1;
      for (int d = 0; d < Dim; ++d)
         if (d != axis) num_faces *= sizes_minus[d];
   }

   /**
   * @brief Return the number of interface faces.
   * @return `∏_{d≠axis} sizes_minus[d]`.
   *
   * @par Complexity
   * O(1).
   */
   GENDIL_HOST_DEVICE
   GlobalIndex GetNumberOfFaces() const { return num_faces; }

   /**
    * @brief Map a linear face index to the underlying minus/plus cell pair.
    *
    * Enumerates tangential positions on the interface (setting the extent along
    * @ref axis to 1) and decodes @p face_index into multi-indices. It then
    * selects the boundary slice along @ref axis on each side according to @ref sign:
    *  - if `sign > 0` (face normal `+e_axis`): minus at last slice, plus at first;
    *  - if `sign < 0` (face normal `-e_axis`): minus at first slice, plus at last.
    * The corresponding linear cell indices are returned as a
    * `face_info_type{ {minus_linear}, {plus_linear} }`.
    *
    * @param[in] face_index Linear index in `[0, GetNumberOfFaces())` over tangential faces.
    * @return A `face_info_type` describing `{minus_cell, plus_cell}` and reference normals.
    *
    * @par Complexity
    * O(Dim) for index decode/encode.
    *
    * @par Example
    * @code
    * // 3D, interface orthogonal to x (axis=0), connect plus-x faces:
    * using Conn = CartesianIntermeshFaceConnectivity<3, 3>;
    * Conn conn({Nx_minus, Ny, Nz}, {Nx_plus, Ny, Nz});
    * for (GlobalIndex f = 0; f < conn.GetNumberOfFaces(); ++f) {
    *   auto info = conn.GetGlobalFaceInfo(f);
    *   // info.minus() and info.plus() carry the boundary cell ids and normals
    * }
    * @endcode
    */
   GENDIL_HOST_DEVICE
   face_info_type GetGlobalFaceInfo(const GlobalIndex& face_index) const
   {
      // Enumerate tangential positions (axis extent = 1)
      auto face_sizes = sizes_minus;
      face_sizes[axis] = 1;
      const auto tangential = GetStructuredSubIndices(face_index, face_sizes);

      // Pick boundary cells along the interface axis according to sign
      auto idx_minus = tangential;
      auto idx_plus  = tangential;

      if (sign > 0) {
         // minus face is +axis → minus at last slice, plus at first slice
         idx_minus[axis] = sizes_minus[axis] - 1;
         idx_plus [axis] = 0;
      } else {
         // minus face is −axis → minus at first slice, plus at last slice
         idx_minus[axis] = 0;
         idx_plus [axis] = sizes_plus[axis] - 1;
      }

      const GlobalIndex minus_linear = ComputeLinearIndex(idx_minus, sizes_minus);
      const GlobalIndex plus_linear  = ComputeLinearIndex(idx_plus,  sizes_plus);

      // Return as {minus, plus} with normals (minus_normal, plus_normal)
      return face_info_type{ { minus_linear }, { plus_linear } };
   }
};

} // namespace gendil
