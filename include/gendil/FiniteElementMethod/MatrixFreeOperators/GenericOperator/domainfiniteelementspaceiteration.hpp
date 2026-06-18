// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

#include <tuple>
#include <type_traits>

namespace gendil {

template<StaticString Name, size_t CellI, class CellSpace>
struct CellExecutionBatch
{
   static constexpr auto domain_name = Name;
   static constexpr size_t cell_batch_index = CellI;

   const CellSpace& cell_space;

   GENDIL_HOST_DEVICE
   constexpr const CellSpace& GetCellFiniteElementSpace() const
   {
      return cell_space;
   }
};

template<
   StaticString Name,
   size_t FaceI,
   class BoundaryFaceSpace>
struct BoundaryFaceExecutionBatch
{
   static constexpr auto domain_name = Name;
   static constexpr size_t face_batch_index = FaceI;

   const BoundaryFaceSpace& face_space;

   GENDIL_HOST_DEVICE
   constexpr const BoundaryFaceSpace& GetBoundaryFaceFiniteElementSpace() const
   {
      return face_space;
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetCellFiniteElementSpace() const
   {
      return face_space.GetMinusFiniteElementSpace();
   }
};

template<
   StaticString Name,
   size_t FaceI,
   class InteriorFaceSpace>
struct InteriorFaceExecutionBatch
{
   static constexpr auto domain_name = Name;
   static constexpr size_t face_batch_index = FaceI;

   const InteriorFaceSpace& face_space;

   GENDIL_HOST_DEVICE
   constexpr const InteriorFaceSpace& GetInteriorFaceFiniteElementSpace() const
   {
      return face_space;
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetMinusCellFiniteElementSpace() const
   {
      return face_space.GetMinusFiniteElementSpace();
   }

   GENDIL_HOST_DEVICE
   constexpr decltype(auto) GetPlusCellFiniteElementSpace() const
   {
      return face_space.GetPlusFiniteElementSpace();
   }
};

template<class T>
struct is_cell_execution_batch : std::false_type {};

template<StaticString Name, size_t CellI, class CellSpace>
struct is_cell_execution_batch<
   CellExecutionBatch<Name, CellI, CellSpace>> : std::true_type {};

template<class T>
inline constexpr bool is_cell_execution_batch_v =
   is_cell_execution_batch<std::remove_cvref_t<T>>::value;

template<class T>
struct is_boundary_face_execution_batch : std::false_type {};

template<
   StaticString Name,
   size_t FaceI,
   class BoundaryFaceSpace>
struct is_boundary_face_execution_batch<
   BoundaryFaceExecutionBatch<
      Name,
      FaceI,
      BoundaryFaceSpace>> : std::true_type {};

template<class T>
inline constexpr bool is_boundary_face_execution_batch_v =
   is_boundary_face_execution_batch<std::remove_cvref_t<T>>::value;

template<class T>
struct is_interior_face_execution_batch : std::false_type {};

template<
   StaticString Name,
   size_t FaceI,
   class InteriorFaceSpace>
struct is_interior_face_execution_batch<
   InteriorFaceExecutionBatch<
      Name,
      FaceI,
      InteriorFaceSpace>> : std::true_type {};

template<class T>
inline constexpr bool is_interior_face_execution_batch_v =
   is_interior_face_execution_batch<std::remove_cvref_t<T>>::value;

template<StaticString Name, class Space, class Fn>
void ForEachCellFiniteElementSpaceInSpace(
   const Space& space,
   Fn&& fn)
{
   using SpaceType = std::remove_cvref_t<Space>;
   if constexpr (is_mixed_finite_element_space_v<SpaceType>)
   {
      const auto& spaces = space.CellSpaces();
      using Spaces = std::remove_cvref_t<decltype(spaces)>;
      constexpr size_t NumSpaces = std::tuple_size_v<Spaces>;
      ConstexprLoop<NumSpaces>(
         [&] (auto index)
         {
            constexpr size_t I = decltype(index)::value;
            const auto& cell_space = std::get<I>(spaces);
            using CellSpace = std::remove_cvref_t<decltype(cell_space)>;
            fn(CellExecutionBatch<Name, I, CellSpace>{ cell_space });
         });
   }
   else if constexpr (is_cell_finite_element_space_v<SpaceType>)
   {
      fn(CellExecutionBatch<Name, 0, SpaceType>{ space });
   }
   else
   {
      static_assert(
         dependent_false_v<SpaceType>,
         "Cells<Name> requires CellIntegrationDomain<Space> to wrap a "
         "homogeneous cell finite element space or a MixedFiniteElementSpace.");
   }
}

template<StaticString Name, class Space, class Fn>
void ForEachCellFiniteElementSpaceInDomain(
   const CellIntegrationDomain<Space>& domain,
   Fn&& fn)
{
   ForEachCellFiniteElementSpaceInSpace<Name>(
      domain.space,
      std::forward<Fn>(fn));
}

template<StaticString Name, class Domain, class Fn>
void ForEachCellFiniteElementSpaceInDomain(
   const Domain&,
   Fn&&)
{
   using DomainType = std::remove_cvref_t<Domain>;
   static_assert(
      dependent_false_v<DomainType>,
      "Cells<Name> requires a normalized CellIntegrationDomain<Space> "
      "registered under Name in WeakFormContext.");
}

template<class WFContext, StaticString Name, class Fn>
void ForEachCellFiniteElementSpace(
   const WFContext& wf_ctx,
   Cells<Name>,
   Fn&& fn)
{
   using Context = std::remove_cvref_t<WFContext>;

   if constexpr (Context::template has_domain<Name>())
   {
      ForEachCellFiniteElementSpaceInDomain<Name>(
         wf_ctx.template domain<Name>(),
         std::forward<Fn>(fn));
   }
   else
   {
      static_assert(
         dependent_false_v<Context>,
         "Cells<Name> requires a cell integration domain registered under "
         "Name in WeakFormContext.");
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachInteriorFaceFiniteElementSpaceInDomain(
   const Domain& domain,
   Fn&& fn)
{
   using DomainType = std::remove_cvref_t<Domain>;

   if constexpr (is_interior_face_integration_domain_v<DomainType>)
   {
      using Space = std::remove_cvref_t<decltype(domain.space)>;
      if constexpr (is_mixed_finite_element_space_v<Space>)
      {
         const auto& spaces = domain.space.InteriorFaceSpaces();
         using Spaces = std::remove_cvref_t<decltype(spaces)>;
         constexpr size_t NumSpaces = std::tuple_size_v<Spaces>;
         ConstexprLoop<NumSpaces>(
            [&] (auto index)
            {
               constexpr size_t FaceI = decltype(index)::value;
               const auto& face_space = std::get<FaceI>(spaces);
               using FaceSpace = std::remove_cvref_t<decltype(face_space)>;

               fn(InteriorFaceExecutionBatch<
                  Name,
                  FaceI,
                  FaceSpace>{ face_space });
            });
      }
      else if constexpr (is_interior_face_finite_element_space_v<Space>)
      {
         fn(InteriorFaceExecutionBatch<
            Name,
            0,
            Space>{ domain.space });
      }
      else
      {
         static_assert(
            dependent_false_v<Space>,
            "InteriorFacets<Name> requires InteriorFaceIntegrationDomain<Space> "
            "to wrap an interior face finite element space or a "
            "MixedFiniteElementSpace.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<DomainType>,
         "InteriorFacets<Name> requires a normalized "
         "InteriorFaceIntegrationDomain<Space> registered under Name.");
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachLocalInteriorFaceFiniteElementSpaceInDomain(
   const Domain& domain,
   Fn&& fn)
{
   using DomainType = std::remove_cvref_t<Domain>;

   if constexpr (is_cell_integration_domain_v<DomainType>)
   {
      using Space = std::remove_cvref_t<decltype(domain.space)>;
      if constexpr (is_cell_finite_element_space_v<Space>)
      {
         fn(CellExecutionBatch<Name, 0, Space>{ domain.space });
      }
      else if constexpr (is_mixed_finite_element_space_v<Space>)
      {
         static_assert(
            dependent_false_v<Space>,
            "InteriorFacets<Name> over a mixed integration domain requires "
            "interior face finite element spaces registered under the same Name.");
      }
      else
      {
         static_assert(
            dependent_false_v<Space>,
            "InteriorFacets<Name> requires CellIntegrationDomain<Space> to wrap "
            "a homogeneous cell finite element space for local facet execution.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<DomainType>,
         "InteriorFacets<Name> requires a normalized homogeneous "
         "CellIntegrationDomain<Space> for local facet execution.");
   }
}

template<class WFContext, StaticString Name, class Fn>
void ForEachInteriorFaceFiniteElementSpace(
   const WFContext& wf_ctx,
   InteriorFacets<Name>,
   Fn&& fn)
{
   using Context = std::remove_cvref_t<WFContext>;

   if constexpr (Context::template has_interior_face_domain<Name>())
   {
      ForEachInteriorFaceFiniteElementSpaceInDomain<Name>(
         wf_ctx.template interior_face_domain<Name>(),
         std::forward<Fn>(fn));
   }
   else if constexpr (Context::template has_domain<Name>())
   {
      ForEachLocalInteriorFaceFiniteElementSpaceInDomain<Name>(
         wf_ctx.template domain<Name>(),
         std::forward<Fn>(fn));
   }
   else
   {
      static_assert(
         dependent_false_v<Context>,
         "InteriorFacets<Name> requires an interior face domain or a "
         "homogeneous cell integration domain registered under Name in "
         "WeakFormContext.");
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachBoundaryFaceFiniteElementSpaceInDomain(
   const Domain& domain,
   Fn&& fn)
{
   using DomainType = std::remove_cvref_t<Domain>;

   if constexpr (is_boundary_face_integration_domain_v<DomainType>)
   {
      using Space = std::remove_cvref_t<decltype(domain.space)>;
      if constexpr (is_mixed_finite_element_space_v<Space>)
      {
         const auto& spaces = domain.space.BoundaryFaceSpaces();
         using Spaces = std::remove_cvref_t<decltype(spaces)>;
         constexpr size_t NumSpaces = std::tuple_size_v<Spaces>;
         ConstexprLoop<NumSpaces>(
            [&] (auto index)
            {
               constexpr size_t FaceI = decltype(index)::value;
               const auto& face_space = std::get<FaceI>(spaces);
               using FaceSpace = std::remove_cvref_t<decltype(face_space)>;
               fn(BoundaryFaceExecutionBatch<
                  Name,
                  FaceI,
                  FaceSpace>{ face_space });
            });
      }
      else if constexpr (is_boundary_face_finite_element_space_v<Space>)
      {
         fn(BoundaryFaceExecutionBatch<
            Name,
            0,
            Space>{ domain.space });
      }
      else
      {
         static_assert(
            dependent_false_v<Space>,
            "BoundaryFacets<Name> requires BoundaryFaceIntegrationDomain<Space> "
            "to wrap a boundary face finite element space or a "
            "MixedFiniteElementSpace.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<DomainType>,
         "BoundaryFacets<Name> requires a normalized "
         "BoundaryFaceIntegrationDomain<Space> registered under Name.");
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachLocalBoundaryFaceFiniteElementSpaceInDomain(
   const Domain& domain,
   Fn&& fn)
{
   using DomainType = std::remove_cvref_t<Domain>;

   if constexpr (is_cell_integration_domain_v<DomainType>)
   {
      using Space = std::remove_cvref_t<decltype(domain.space)>;
      if constexpr (is_cell_finite_element_space_v<Space>)
      {
         fn(CellExecutionBatch<Name, 0, Space>{ domain.space });
      }
      else if constexpr (is_mixed_finite_element_space_v<Space>)
      {
         static_assert(
            dependent_false_v<Space>,
            "BoundaryFacets<Name> over a mixed integration domain requires "
            "boundary face finite element spaces registered under the same Name.");
      }
      else
      {
         static_assert(
            dependent_false_v<Space>,
            "BoundaryFacets<Name> requires CellIntegrationDomain<Space> to wrap "
            "a homogeneous cell finite element space for local facet execution.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<DomainType>,
         "BoundaryFacets<Name> requires a normalized homogeneous "
         "CellIntegrationDomain<Space> for local facet execution.");
   }
}

template<class WFContext, StaticString Name, class Fn>
void ForEachBoundaryFaceFiniteElementSpace(
   const WFContext& wf_ctx,
   BoundaryFacets<Name>,
   Fn&& fn)
{
   using Context = std::remove_cvref_t<WFContext>;

   if constexpr (Context::template has_boundary_face_domain<Name>())
   {
      ForEachBoundaryFaceFiniteElementSpaceInDomain<Name>(
         wf_ctx.template boundary_face_domain<Name>(),
         std::forward<Fn>(fn));
   }
   else if constexpr (Context::template has_domain<Name>())
   {
      ForEachLocalBoundaryFaceFiniteElementSpaceInDomain<Name>(
         wf_ctx.template domain<Name>(),
         std::forward<Fn>(fn));
   }
   else
   {
      static_assert(
         dependent_false_v<Context>,
         "BoundaryFacets<Name> requires a boundary face domain or a "
         "homogeneous cell integration domain registered under Name in "
         "WeakFormContext.");
   }
}

} // namespace gendil
