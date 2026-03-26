// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Generic iteration utilities and concept for global face meshes.
 *
 * This header defines:
 *  - The @ref gendil::mesh::GlobalFaceMeshConnectivity concept describing the
 *    minimal interface a "global face mesh" must provide.
 *  - Lightweight iteration helpers over global faces and over a set of
 *    face meshes, with execution delegated to a user-provided KernelConfiguration.
 *
 * The design keeps iteration policy separate from mesh connectivity:
 * users supply a KernelConfiguration with a `BlockLoop(count, body)` static method
 * to select the execution backend (e.g., serial, OpenMP, CUDA/HIP).
 *
 */

#include "gendil/Utilities/types.hpp"

namespace gendil {

namespace mesh {

/**
 * @brief Concept for global face mesh connectivity.
 *
 * A type `G` models GlobalFaceMeshConnectivity if it provides:
 *  - `GlobalIndex GetNumberOfFaces() const;`
 *  - `auto GetGlobalFaceInfo(GlobalIndex fi) const;`  (return type is user-defined)
 *
 * Semantics:
 *  - `GetNumberOfFaces()` returns the total number of global faces.
 *  - `GetGlobalFaceInfo(fi)` returns mesh-defined information for the global face
 *    with index `fi` (e.g., endpoints, adjacent elements, orientation, boundary tags).
 *    The exact type is left to the mesh implementation; this concept only checks
 *    for the presence of the call.
 *
 * This concept is intentionally minimal to accommodate multiple mesh backends.
 */
template <typename G>
concept GlobalFaceMeshConnectivity =
   // std::is_object_v<G> &&
   requires (const G& g, GlobalIndex fi)
   {
      // Total number of global faces (runtime count is fine)
      { g.GetNumberOfFaces() } -> std::convertible_to<GlobalIndex>;

      // Access global face info by global face index
      { g.GetGlobalFaceInfo(fi) };
   };

/**
 * @brief Iterate over all global faces of a mesh with a user-supplied kernel configuration.
 *
 * Executes `body(face_id)` for each global face `face_id` in `[0, GetNumberOfFaces())`,
 * delegating the loop policy to `KernelConfiguration::BlockLoop(count, body)`.
 *
 * @tparam KernelConfiguration  Policy type exposing
 *         `static void BlockLoop(GlobalIndex count, F&& body)`.
 *         This selects the execution backend (CPU, GPU, MFEM forall, etc.).
 * @tparam GFMesh               A type modeling @ref GlobalFaceMeshConnectivity.
 * @tparam Lambda               Callable satisfying `std::invocable<Lambda, GlobalIndex>`.
 *
 * @param[in] gfmesh  Global face mesh connectivity object.
 * @param[in] body    Functor/lambda invoked with the global face index.
 *
 * @pre `gfmesh` satisfies GlobalFaceMeshConnectivity.
 * @pre `body` must be compatible with the execution space implied by KernelConfiguration
 *      (e.g., host/device annotation if launching on GPU).
 *
 * @warning The function does not perform bounds checking inside `body`.
 *
 * @par Complexity
 * Linear in the number of faces: `O(gfmesh.GetNumberOfFaces())`.
 *
 * @par Example
 * @code
 * struct CPUSeq {
 *   static void BlockLoop(GlobalIndex n, auto&& f) {
 *     for (GlobalIndex i = 0; i < n; ++i) f(i);
 *   }
 * };
 *
 * GlobalFaceIterator<CPUSeq>(gfmesh, [&](GlobalIndex fi){
 *   const auto info = gfmesh.GetGlobalFaceInfo(fi);
 *   // ... do work ...
 * });
 * @endcode
 */
template <typename KernelConfiguration, GlobalFaceMeshConnectivity GFMesh, typename Lambda>
requires std::invocable<Lambda, GlobalIndex>
inline void GlobalFaceIterator(const GFMesh& gfmesh, Lambda&& body)
{
   const GlobalIndex nfaces = gfmesh.GetNumberOfFaces();
   KernelConfiguration::BlockLoop(nfaces, std::forward<Lambda>(body));
}

/**
 * @brief Apply a callable to each face mesh in a tuple (compile-time unrolled).
 *
 * Invokes `body(gfmesh_i)` for each entry of the `gfmeshes` tuple. This is a
 * convenient way to perform the same operation across several mesh backends or
 * partitions without writing boilerplate.
 *
 * @tparam Lambda     Callable type; must be invocable with `const GFMesh&`.
 * @tparam GFMeshes   Parameter pack of types modeling
 *                    @ref GlobalFaceMeshConnectivity.
 *
 * @param[in] gfmeshes  Tuple of global face mesh connectivity objects.
 * @param[in] body      Functor/lambda called as `body(gfmesh)` for each tuple entry.
 *
 * @note Uses `ConstexprLoop<N>` to unroll at compile time.
 *
 * @par Example
 * @code
 * auto meshes = std::make_tuple(meshA, meshB);
 * ForEachFaceMesh(meshes, [&](const auto& m){
 *   GlobalFaceIterator<CPUSeq>(m, do_face);
 * });
 * @endcode
 */
template <typename Lambda, GlobalFaceMeshConnectivity ... GFMeshes>
inline void ForEachFaceMesh(const std::tuple< GFMeshes ... >& gfmeshes, Lambda&& body)
{
   ConstexprLoop< sizeof...(GFMeshes) >( [&]( auto I )
   {
      const auto& gfmesh = std::get<I>(gfmeshes);
      body( gfmesh );
   });
}

/**
 * @brief Apply a callable to a single face mesh (uniform API with the tuple overload).
 *
 * This overload mirrors the tuple-based version to enable generic code that can
 * accept either a single mesh or a tuple of meshes.
 *
 * @tparam GFMesh  A type modeling @ref GlobalFaceMeshConnectivity.
 * @tparam Lambda  Callable type; must be invocable with `const GFMesh&`.
 *
 * @param[in] gfmesh  Global face mesh connectivity object.
 * @param[in] body    Functor/lambda called as `body(gfmesh)`.
 *
 * @par Example
 * @code
 * ForEachFaceMesh(mesh, [&](const auto& m){
 *   GlobalFaceIterator<CPUSeq>(m, do_face);
 * });
 * @endcode
 */
template <GlobalFaceMeshConnectivity GFMesh, typename Lambda>
inline void ForEachFaceMesh(const GFMesh& gfmesh, Lambda&& body)
{
   body( gfmesh );
}

} // namespace mesh

} // namespace gendil
