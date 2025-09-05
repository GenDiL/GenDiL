// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Nonconforming Cartesian inter-mesh face connector (coarse ↔ fine).
 *
 * This header provides
 * `NonconformingCartesianIntermeshFaceConnectivity<Dim, LocalFaceIndex>`, which
 * enumerates and maps interface faces between two axis-aligned Cartesian meshes
 * when the "plus" side is a *tangential refinement* of the "minus" side.
 *
 * Key ideas:
 *  - The interface is orthogonal to `axis = LocalFaceIndex % Dim`.
 *  - Faces `0..Dim-1` carry reference normal `-e_axis` (“minus faces”),
 *    faces `Dim..2*Dim-1` carry `+e_axis` (“plus faces”).
 *  - Tangential directions (all `d != axis`) assume integer refinement ratios
 *    `refine[d] = sizes_plus[d] / sizes_minus[d]` (with `refine[axis] = 1`).
 *  - Each *fine* tangential position corresponds to one global face pair.
 *
 * The returned face info is expressed with `FaceView`/`GlobalFaceInfo` and
 * carries reference normals and conformity maps to handle subface geometry.
 *
 * All functions are header-only and marked @c GENDIL_HOST_DEVICE.
 */

#include "gendil/prelude.hpp"

namespace gendil {

/**
 * @brief Inter-mesh face connectivity for a nonconforming Cartesian interface.
 *
 * Connects the face `LocalFaceIndex` of the coarse (“minus”) mesh to the
 * opposite face on the refined (“plus”) mesh. Enumeration proceeds over all
 * *fine* tangential positions, producing one `{minus_cell, plus_cell}` pair
 * per fine subface on the interface plane.
 *
 * @tparam Dim            Topological dimension (e.g., 2 or 3).
 * @tparam LocalFaceIndex Compile-time local face id in `[0, 2*Dim)`.
 *                        `0..Dim-1` denote faces with normal `-e_axis`,
 *                        `Dim..2*Dim-1` denote faces with `+e_axis`.
 *
 * @par Associated types and constants
 *  - `using geometry         = HyperCube<Dim>;`
 *  - `using orientation_type = IdentityOrientation<Dim>;`
 *  - `static constexpr Integer minus_local_face_index` — provided face id
 *  - `static constexpr Integer plus_local_face_index`  — opposite face id
 *  - `using minus_normal_type = CanonicalVector<Dim, axis,  sign>;`
 *  - `using plus_normal_type  = CanonicalVector<Dim, axis, -sign>;`
 *  - `using minus_conformity_type = NonconformingHyperCubeFaceMap<Dim>;`
 *  - `using plus_conformity_type  = ConformingFaceMap<Dim>;`
 *  - `using minus_view_type = FaceView<..., minus_conformity_type>;`
 *  - `using plus_view_type  = FaceView<...,  plus_conformity_type>;`
 *  - `using face_info_type  = GlobalFaceInfo<minus_view_type, plus_view_type>;`
 *
 * @par Data members
 *  - `sizes_minus[d]` : coarse (#cells per dimension)
 *  - `sizes_plus[d]`  : fine   (#cells per dimension)
 *  - `refine[d]`      : integer refinement ratios (tangential), `refine[axis]=1`
 *  - `num_pairs`      : number of fine subfaces across the interface
 *                       = ∏_{d≠axis} sizes_plus[d]
 *
 * @note The *plus* mesh is assumed to be a tangential refinement of the *minus*
 *       mesh (integer ratios). The extent along @ref axis may differ freely.
 */
template <Integer Dim, Integer LocalFaceIndex>
struct NonconformingCartesianIntermeshFaceConnectivity {
  static_assert(0 <= LocalFaceIndex && LocalFaceIndex < 2*Dim, "LocalFaceIndex out of range");

  using geometry         = HyperCube<Dim>;
  using orientation_type = IdentityOrientation<Dim>;

  // Derive axis/sign from the local-face index
  static constexpr Integer axis = LocalFaceIndex % Dim;
  static constexpr int     sign = (LocalFaceIndex < Dim ? -1 : +1);

  // Minus = the face indicated by LocalFaceIndex; Plus = the opposite face
  static constexpr Integer minus_local_face_index = LocalFaceIndex;
  static constexpr Integer plus_local_face_index  =
      (LocalFaceIndex < Dim ? LocalFaceIndex + Dim : LocalFaceIndex - Dim);

  using minus_normal_type = CanonicalVector<Dim, axis,  sign>;
  using plus_normal_type  = CanonicalVector<Dim, axis, -sign>;

  // Conformity types: minus (coarse) is a subface; plus (fine) is identity
  // (Keep <Dim> to match your current face-coordinate convention.)
  using minus_conformity_type = NonconformingHyperCubeFaceMap<Dim>;
  using plus_conformity_type  = ConformingFaceMap<Dim>;

  using minus_view_type = FaceView<
      std::integral_constant<Integer, minus_local_face_index>,
      geometry, orientation_type, minus_normal_type, minus_conformity_type>;
  using plus_view_type  = FaceView<
      std::integral_constant<Integer, plus_local_face_index>,
      geometry, orientation_type, plus_normal_type,  plus_conformity_type>;

  using face_info_type = GlobalFaceInfo<minus_view_type, plus_view_type>;

  // Grid sizes: minus = coarse side, plus = fine side
  std::array<GlobalIndex, Dim> sizes_minus{};
  std::array<GlobalIndex, Dim> sizes_plus{};
  // Tangential refinement ratios r[d] = sizes_plus[d] / sizes_minus[d] (for d!=axis), r[axis]=1
  std::array<Integer, Dim> refine{};
  GlobalIndex num_pairs{0}; // number of fine subfaces across the interface plane

  /**
   * @brief Construct from coarse/fine per-dimension grid sizes.
   *
   * Initializes refinement ratios in all tangential directions and computes
   * the total number of fine subfaces on the interface.
   *
   * @param[in] coarse  Grid sizes of the coarse (“minus”) mesh.
   * @param[in] fine    Grid sizes of the refined (“plus”) mesh.
   *
   * @pre For all `d != axis`, `fine[d]` is an integer multiple of `coarse[d]`.
   * @note This constructor does not currently `assert` the divisibility by default;
   *       uncomment the `GENDIL_ASSERT` lines if you want runtime checks.
   *
   * @par Complexity
   * O(Dim).
   */
  GENDIL_HOST_DEVICE
  NonconformingCartesianIntermeshFaceConnectivity(std::array<GlobalIndex, Dim> coarse,
                                                  std::array<GlobalIndex, Dim> fine)
    : sizes_minus(coarse), sizes_plus(fine)
  {
    for (int d=0; d<Dim; ++d) {
      if (d == axis) { refine[d] = 1; continue; }
      // Require integer refinement tangentially (fine is a refinement of coarse)
      // GENDIL_ASSERT(sizes_plus[d] % sizes_minus[d] == 0);
      refine[d] = static_cast<Integer>(sizes_plus[d] / sizes_minus[d]);
      // GENDIL_ASSERT(refine[d] >= 1);
    }
    // One face per fine tangential position
    num_pairs = 1;
    for (int d=0; d<Dim; ++d) if (d != axis) num_pairs *= sizes_plus[d];
  }

  /**
   * @brief Number of face pairs (one per fine tangential position).
   * @return `∏_{d≠axis} sizes_plus[d]`.
   *
   * @par Complexity
   * O(1).
   */
  GENDIL_HOST_DEVICE
  GlobalIndex GetNumberOfFaces() const { return num_pairs; }

  /**
   * @brief Map a linear interface face index to `{minus_cell, plus_cell}` with subface map.
   *
   * The linear @p face_index is decoded over the *fine* tangential grid.
   * For each tangential dimension `d != axis`, the fine coordinate is split into:
   *  - a coarse tangential coordinate: `coarse_t[d] = fine_t[d] / refine[d]`,
   *  - a subface offset inside the coarse face: `sub_ofs[d] = fine_t[d] % refine[d]`.
   *
   * The boundary cells along the interface axis are then selected according to
   * @ref sign:
   *  - if `sign > 0` (reference normal `+e_axis`): minus at last slice, plus at first;
   *  - if `sign < 0` (reference normal `-e_axis`): minus at first slice, plus at last.
   *
   * A `NonconformingHyperCubeFaceMap` is built on the minus side to describe
   * the subface origin/size in face coordinates (`[0,1]^{Dim-1}` tangentially,
   * `[0,1]` along the axis direction):
   *  - `o[d] = sub_ofs[d] / refine[d]`, `s[d] = 1 / refine[d]` for `d != axis`,
   *  - `o[axis] = 0`, `s[axis] = 1`.
   * The plus side uses an identity `ConformingFaceMap`.
   *
   * @param[in] face_index Linear index over fine tangential positions (`0..GetNumberOfFaces()-1`).
   * @return A `face_info_type` carrying `minus_view` (coarse with subface map) and
   *         `plus_view` (fine with identity map), along with reference normals.
   *
   * @par Complexity
   * O(Dim) for index decode/encode and map construction.
   *
   * @par Example
   * @code
   * using Conn = NonconformingCartesianIntermeshFaceConnectivity<3,3>;
   * Conn conn({Nx_c, Ny_c, Nz_c}, {Nx_f, Ny_f, Nz_f}); // axis = 0
   * for (GlobalIndex f = 0; f < conn.GetNumberOfFaces(); ++f) {
   *   auto info = conn.GetGlobalFaceInfo(f);
   *   // info.minus() has a NonconformingHyperCubeFaceMap locating the subface on the coarse side
   *   // info.plus()  is the fine-side boundary cell with identity map
   * }
   * @endcode
   */
  GENDIL_HOST_DEVICE
  face_info_type GetGlobalFaceInfo(const GlobalIndex& face_index) const
  {
    // Enumerate *fine* tangential positions (each is a subface on the coarse side)
    std::array<GlobalIndex, Dim> fine_face_sizes = sizes_plus;
    fine_face_sizes[axis] = 1;

    const auto fine_t = GetStructuredSubIndices(face_index, fine_face_sizes);

    // Coarse tangential coords and subface offsets
    std::array<GlobalIndex, Dim> coarse_t{};
    std::array<Integer,     Dim> sub_ofs{};
    for (int d=0; d<Dim; ++d) {
      if (d == axis) { coarse_t[d] = 0; sub_ofs[d] = 0; continue; }
      coarse_t[d] = fine_t[d] / static_cast<GlobalIndex>(refine[d]);
      sub_ofs[d]  = static_cast<Integer>(fine_t[d] % refine[d]);
    }

    // Minus (coarse) cell index: boundary cell along 'axis' depending on sign
    auto idx_minus = coarse_t;
    idx_minus[axis] = (sign > 0 ? sizes_minus[axis]-1 : 0);
    const GlobalIndex minus_linear = ComputeLinearIndex(idx_minus, sizes_minus);

    // Plus (fine) cell index: opposite boundary cell along 'axis'
    auto idx_plus = fine_t;
    idx_plus[axis] = (sign > 0 ? 0 : sizes_plus[axis]-1);
    const GlobalIndex plus_linear  = ComputeLinearIndex(idx_plus, sizes_plus);

    // Build minus subface map in face coordinates: origin/size per tangential dir; axis dir is [0,1]
    Point<Dim> o{};
    std::array<Real, Dim> s{};
    for (int d=0; d<Dim; ++d) {
      if (d == axis) { o[d] = Real(0); s[d] = Real(1); }
      else {
        const Real rr = Real(refine[d]);
        const Real kk = Real(sub_ofs[d]);
        o[d] = kk / rr;     // start offset inside the coarse face
        s[d] = Real(1) / rr; // subface size fraction
      }
    }
    minus_conformity_type conf_minus{ o, s };
    plus_conformity_type  conf_plus{};

    // Assemble views
    minus_view_type minus{ minus_linear, {}, {}, conf_minus };
    plus_view_type  plus { plus_linear,  {}, {}, conf_plus  };

    return face_info_type{ minus, plus };
  }
};

} // namespace gendil
