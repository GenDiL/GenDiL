// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/FiniteElementMethod/Restrictions/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <type_traits>

namespace gendil
{

struct IdentityBsrGather
{
   static constexpr bool is_identity = true;
};

struct IdentityBsrScatter
{
   static constexpr bool is_identity = true;
};

template < typename FiniteElementSpace >
struct DGGatherToBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

   void operator()( const Vector & x_fe, Vector & x_bsr ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, L2Restriction >,
         "DGGatherToBsr only supports L2Restriction finite element spaces." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();

      GENDIL_VERIFY(
         x_bsr.Size() == static_cast< size_t >( num_elements * block_size ),
         "DGGatherToBsr output vector has the wrong BSR size." );
      GENDIL_VERIFY(
         x_fe.Size() >= static_cast< size_t >(
            finite_element_space.restriction.shift +
            finite_element_space.GetNumberOfFiniteElementDofs() ),
         "DGGatherToBsr input vector is too small for the finite element space." );

      const Real * fe_data = x_fe.ReadHostData();
      Real * bsr_data = x_bsr.WriteHostData();

      for ( GlobalIndex element_index = 0;
            element_index < num_elements;
            ++element_index )
      {
         ForEachLocalDof(
            finite_element_space,
            [&] ( const auto component, const auto & indices )
            {
               const GlobalIndex bsr_index =
                  element_index * block_size +
                  FlattenLocalDof(
                     finite_element_space,
                     component,
                     indices );
               const GlobalIndex fe_index =
                  GlobalDofIndex(
                     finite_element_space,
                     component,
                     element_index,
                     indices );
               bsr_data[bsr_index] = fe_data[fe_index];
            });
      }
   }
};

template < typename FiniteElementSpace >
struct DGScatterFromBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

   void operator()( const Vector & y_bsr, Vector & y_fe ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, L2Restriction >,
         "DGScatterFromBsr only supports L2Restriction finite element spaces." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();

      GENDIL_VERIFY(
         y_bsr.Size() == static_cast< size_t >( num_elements * block_size ),
         "DGScatterFromBsr input vector has the wrong BSR size." );
      GENDIL_VERIFY(
         y_fe.Size() >= static_cast< size_t >(
            finite_element_space.restriction.shift +
            finite_element_space.GetNumberOfFiniteElementDofs() ),
         "DGScatterFromBsr output vector is too small for the finite element space." );

      const Real * bsr_data = y_bsr.ReadHostData();
      Real * fe_data = y_fe.WriteHostData();

      for ( GlobalIndex element_index = 0;
            element_index < num_elements;
            ++element_index )
      {
         ForEachLocalDof(
            finite_element_space,
            [&] ( const auto component, const auto & indices )
            {
               const GlobalIndex bsr_index =
                  element_index * block_size +
                  FlattenLocalDof(
                     finite_element_space,
                     component,
                     indices );
               const GlobalIndex fe_index =
                  GlobalDofIndex(
                     finite_element_space,
                     component,
                     element_index,
                     indices );
               fe_data[fe_index] = bsr_data[bsr_index];
            });
      }
   }
};

template < typename FiniteElementSpace >
struct CGGatherToBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

   void operator()( const Vector & x_fe, Vector & x_bsr ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, H1Restriction >,
         "CGGatherToBsr only supports H1Restriction finite element spaces." );
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "CGGatherToBsr currently supports scalar H1 finite element spaces only." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;

      GENDIL_VERIFY(
         x_bsr.Size() == static_cast< size_t >( expected_bsr_size ),
         "CGGatherToBsr output BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         x_fe.Size() >= static_cast< size_t >( finite_element_space.restriction.num_dofs ),
         "CGGatherToBsr input vector is smaller than the conforming H1 vector size." );

      const Real * fe_data = x_fe.ReadHostData();
      Real * bsr_data = x_bsr.WriteHostData();

      // This gathers into the raw element-block BSR vector. The wrapped BSR
      // matrix is not a true-DoF globally assembled sparse matrix.
      for ( GlobalIndex element_index = 0;
            element_index < num_elements;
            ++element_index )
      {
         ForEachScalarLocalDof(
            finite_element_space,
            [&] ( const auto component, const auto & indices )
            {
               const GlobalIndex local_id =
                  FlattenLocalDof( finite_element_space, component, indices );
               const GlobalIndex bsr_index =
                  element_index * block_size + local_id;
               const GlobalIndex fe_index =
                  GlobalDofIndex(
                     finite_element_space,
                     component,
                     element_index,
                     indices );
               bsr_data[bsr_index] = fe_data[fe_index];
            });
      }
   }
};

template < typename FiniteElementSpace >
struct CGScatterFromBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

   void operator()( const Vector & y_bsr, Vector & y_fe ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      static_assert(
         std::is_same_v< typename Space::restriction_type, H1Restriction >,
         "CGScatterFromBsr only supports H1Restriction finite element spaces." );
      static_assert(
         !is_vector_shape_functions_v< ShapeFunctions >,
         "CGScatterFromBsr currently supports scalar H1 finite element spaces only." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;

      GENDIL_VERIFY(
         y_bsr.Size() == static_cast< size_t >( expected_bsr_size ),
         "CGScatterFromBsr input BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         y_fe.Size() >= static_cast< size_t >( finite_element_space.restriction.num_dofs ),
         "CGScatterFromBsr output vector is smaller than the conforming H1 vector size." );

      const Real * bsr_data = y_bsr.ReadHostData();
      Real * fe_data = y_fe.WriteHostData();

      for ( GlobalIndex i = 0;
            i < static_cast< GlobalIndex >( finite_element_space.restriction.num_dofs );
            ++i )
      {
         fe_data[i] = 0.0;
      }

      // Serial scatter-add gives Set semantics without element-write races.
      // Element-parallel conforming scatter needs atomics or coloring.
      for ( GlobalIndex element_index = 0;
            element_index < num_elements;
            ++element_index )
      {
         ForEachScalarLocalDof(
            finite_element_space,
            [&] ( const auto component, const auto & indices )
            {
               const GlobalIndex local_id =
                  FlattenLocalDof( finite_element_space, component, indices );
               const GlobalIndex bsr_index =
                  element_index * block_size + local_id;
               const GlobalIndex fe_index =
                  GlobalDofIndex(
                     finite_element_space,
                     component,
                     element_index,
                     indices );
               fe_data[fe_index] += bsr_data[bsr_index];
            });
      }
   }
};

template < typename FiniteElementSpace >
struct VectorCGGatherToBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

   void operator()( const Vector & x_fe, Vector & x_bsr ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      using Restriction = typename Space::restriction_type;
      static_assert(
         is_vector_h1_restriction_v< Restriction >,
         "VectorCGGatherToBsr only supports VectorH1Restriction<NComp> finite element spaces." );
      static_assert(
         is_vector_shape_functions_v< ShapeFunctions >,
         "VectorCGGatherToBsr requires a vector finite element space." );
      static_assert(
         Restriction::num_comp == ShapeFunctions::vector_dim,
         "VectorH1Restriction<NComp> must match the vector finite element component count." );
      static_assert(
         VectorComponentDofShapesMatchFirst< ShapeFunctions >(),
         "VectorH1Restriction currently requires identical scalar component DoF shapes." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;
      const GlobalIndex expected_fe_size =
         static_cast< GlobalIndex >( Restriction::num_comp ) *
         static_cast< GlobalIndex >(
            finite_element_space.restriction.scalar_num_dofs );

      GENDIL_VERIFY(
         x_bsr.Size() == static_cast< size_t >( expected_bsr_size ),
         "VectorCGGatherToBsr output BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         x_fe.Size() >= static_cast< size_t >( expected_fe_size ),
         "VectorCGGatherToBsr input vector is smaller than the vector conforming H1 vector size." );

      const Real * fe_data = x_fe.ReadHostData();
      Real * bsr_data = x_bsr.WriteHostData();

      // Gather from component-major vector true DoFs into the element-block
      // BSR layout. The BSR local position is still the full vector local DoF.
      for ( GlobalIndex element_index = 0;
            element_index < num_elements;
            ++element_index )
      {
         ForEachLocalDof(
            finite_element_space,
            [&] ( const auto component, const auto & indices )
            {
               const GlobalIndex local_id =
                  FlattenLocalDof( finite_element_space, component, indices );
               const GlobalIndex bsr_index =
                  element_index * block_size + local_id;
               const GlobalIndex fe_index =
                  GlobalDofIndex(
                     finite_element_space,
                     component,
                     element_index,
                     indices );
               bsr_data[bsr_index] = fe_data[fe_index];
            });
      }
   }
};

template < typename FiniteElementSpace >
struct VectorCGScatterFromBsr
{
   static constexpr bool is_identity = false;

   FiniteElementSpace finite_element_space;

   void operator()( const Vector & y_bsr, Vector & y_fe ) const
   {
      using Space = std::remove_cvref_t< FiniteElementSpace >;
      using ShapeFunctions = typename Space::finite_element_type::shape_functions;
      using Restriction = typename Space::restriction_type;
      static_assert(
         is_vector_h1_restriction_v< Restriction >,
         "VectorCGScatterFromBsr only supports VectorH1Restriction<NComp> finite element spaces." );
      static_assert(
         is_vector_shape_functions_v< ShapeFunctions >,
         "VectorCGScatterFromBsr requires a vector finite element space." );
      static_assert(
         Restriction::num_comp == ShapeFunctions::vector_dim,
         "VectorH1Restriction<NComp> must match the vector finite element component count." );
      static_assert(
         VectorComponentDofShapesMatchFirst< ShapeFunctions >(),
         "VectorH1Restriction currently requires identical scalar component DoF shapes." );

      const GlobalIndex num_elements =
         finite_element_space.GetNumberOfFiniteElements();
      constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
      const GlobalIndex expected_bsr_size =
         num_elements * block_size;
      const GlobalIndex expected_fe_size =
         static_cast< GlobalIndex >( Restriction::num_comp ) *
         static_cast< GlobalIndex >(
            finite_element_space.restriction.scalar_num_dofs );

      GENDIL_VERIFY(
         y_bsr.Size() == static_cast< size_t >( expected_bsr_size ),
         "VectorCGScatterFromBsr input BSR vector size is inconsistent with the element-local DoF count." );
      GENDIL_VERIFY(
         y_fe.Size() >= static_cast< size_t >( expected_fe_size ),
         "VectorCGScatterFromBsr output vector is smaller than the vector conforming H1 vector size." );

      const Real * bsr_data = y_bsr.ReadHostData();
      Real * fe_data = y_fe.WriteHostData();

      for ( GlobalIndex i = 0; i < expected_fe_size; ++i )
      {
         fe_data[i] = 0.0;
      }

      // Match scalar CGScatterFromBsr: clear the true-DoF vector, then
      // scatter-add element contributions so shared H1 nodes accumulate.
      for ( GlobalIndex element_index = 0;
            element_index < num_elements;
            ++element_index )
      {
         ForEachLocalDof(
            finite_element_space,
            [&] ( const auto component, const auto & indices )
            {
               const GlobalIndex local_id =
                  FlattenLocalDof( finite_element_space, component, indices );
               const GlobalIndex bsr_index =
                  element_index * block_size + local_id;
               const GlobalIndex fe_index =
                  GlobalDofIndex(
                     finite_element_space,
                     component,
                     element_index,
                     indices );
               fe_data[fe_index] += bsr_data[bsr_index];
            });
      }
   }
};

template <
   typename FESpace,
   typename Restriction =
      typename std::remove_cvref_t< FESpace >::restriction_type >
struct DefaultBsrGatherFor
{
   static_assert(
      dependent_false_v< FESpace >,
      "DefaultBsrGatherFor supports only L2Restriction, scalar H1Restriction, and VectorH1Restriction finite element spaces." );
};

template < typename FESpace >
struct DefaultBsrGatherFor< FESpace, L2Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = DGGatherToBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
struct DefaultBsrGatherFor< FESpace, H1Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename space_type::finite_element_type::shape_functions;

   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "CGGatherToBsr currently supports scalar H1 finite element spaces only." );

   using type = CGGatherToBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace, size_t NComp >
struct DefaultBsrGatherFor< FESpace, VectorH1Restriction< NComp > >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = VectorCGGatherToBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
using default_bsr_gather_t = typename DefaultBsrGatherFor< FESpace >::type;

template <
   typename FESpace,
   typename Restriction =
      typename std::remove_cvref_t< FESpace >::restriction_type >
struct DefaultBsrScatterFor
{
   static_assert(
      dependent_false_v< FESpace >,
      "DefaultBsrScatterFor supports only L2Restriction, scalar H1Restriction, and VectorH1Restriction finite element spaces." );
};

template < typename FESpace >
struct DefaultBsrScatterFor< FESpace, L2Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = DGScatterFromBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
struct DefaultBsrScatterFor< FESpace, H1Restriction >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename space_type::finite_element_type::shape_functions;

   static_assert(
      !is_vector_shape_functions_v< ShapeFunctions >,
      "CGScatterFromBsr currently supports scalar H1 finite element spaces only." );

   using type = CGScatterFromBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace, size_t NComp >
struct DefaultBsrScatterFor< FESpace, VectorH1Restriction< NComp > >
{
   using space_type = std::remove_cvref_t< FESpace >;
   using type = VectorCGScatterFromBsr< space_type >;

   static type Make( const space_type & finite_element_space )
   {
      return type{ finite_element_space };
   }
};

template < typename FESpace >
using default_bsr_scatter_t = typename DefaultBsrScatterFor< FESpace >::type;

} // namespace gendil
