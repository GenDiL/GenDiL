// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-global-face-advection-operator skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{

template <
   typename BaseKernelPolicy,
   typename FaceReadPolicy,
   typename FaceWritePolicy >
struct KernelPolicyWithFacePolicies : public BaseKernelPolicy
{
   using face_read_dofs_policy = FaceReadPolicy;
   using face_write_dofs_policy = FaceWritePolicy;
};

template < typename BaseKernelPolicy >
using DefaultFacePolicies = BaseKernelPolicy;

template < typename BaseKernelPolicy >
using FullSharedFacePolicies =
   KernelPolicyWithFacePolicies<
      BaseKernelPolicy,
      FullSharedFaceReadDofsPolicy,
      FullSharedFaceWriteDofsPolicy >;

template < Integer Order, size_t... I >
auto MakeRepeatedFiniteElementOrders( std::index_sequence< I... > )
{
   return FiniteElementOrders< ( static_cast< void >( I ), Order )... >{};
}

template < Integer NumQuad1D, size_t... I >
auto MakeRepeatedIntegrationRuleNumPoints( std::index_sequence< I... > )
{
   return IntegrationRuleNumPoints<
      ( static_cast< void >( I ), NumQuad1D )... >{};
}

template < Integer Dim, Integer Order >
auto MakeRepeatedFiniteElementOrders()
{
   return MakeRepeatedFiniteElementOrders< Order >(
      std::make_index_sequence< Dim >{} );
}

template < Integer Dim, Integer NumQuad1D >
auto MakeRepeatedIntegrationRuleNumPoints()
{
   return MakeRepeatedIntegrationRuleNumPoints< NumQuad1D >(
      std::make_index_sequence< Dim >{} );
}

template < Integer Dim >
auto MakeCartesianMeshForExtents(
   const std::array< GlobalIndex, Dim > & extents )
{
   if constexpr ( Dim == 1 )
   {
      return Cartesian1DMesh( 1.0, extents[ 0 ] );
   }
   else if constexpr ( Dim == 2 )
   {
      return Cartesian2DMesh( 1.0, extents[ 0 ], extents[ 1 ] );
   }
   else
   {
      static_assert( Dim == 3 );
      return Cartesian3DMesh( 1.0, extents[ 0 ], extents[ 1 ], extents[ 2 ] );
   }
}

template < typename FaceMeshes >
GlobalIndex CountGlobalFaces( const FaceMeshes & face_meshes )
{
   GlobalIndex num_faces = 0;
   mesh::ForEachFaceMesh(
      face_meshes,
      [&] ( const auto & face_mesh )
      {
         num_faces += face_mesh.GetNumberOfFaces();
      } );
   return num_faces;
}

template < Integer Dim >
void PrintExtents( const std::array< GlobalIndex, Dim > & extents )
{
   std::cout << '{';
   for ( Integer d = 0; d < Dim; ++d )
   {
      if ( d > 0 )
      {
         std::cout << ',';
      }
      std::cout << extents[ d ];
   }
   std::cout << '}';
}

template < typename VectorType >
bool CheckVectorFinite( const char * label, const VectorType & values )
{
   bool success = true;
   for ( Integer i = 0; i < values.Size(); ++i )
   {
      if ( !std::isfinite( values[ i ] ) )
      {
         std::cout << "FAILED: " << label
                   << " has non-finite entry at index " << i << ".\n";
         success = false;
         break;
      }
   }
   return success;
}

template < Integer Dim >
struct AdvectionVelocity
{
   GENDIL_HOST_DEVICE
   void operator()(
      const std::array< Real, Dim > & X,
      Real ( & velocity )[ Dim ] ) const
   {
      for ( Integer d = 0; d < Dim; ++d )
      {
         velocity[ d ] =
            0.5 +
            0.125 * static_cast< Real >( d + 1 ) +
            0.25 * X[ d ];
      }
   }
};

template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename FaceMeshes,
   typename Rule,
   Integer Dim >
Vector ApplyGlobalFaceAdvection(
   const FiniteElementSpace & fe_space,
   const FaceMeshes & face_meshes,
   const Rule & integration_rule,
   const Vector & x,
   const Vector & baseline,
   AdvectionVelocity< Dim > adv )
{
   Vector y = baseline;

   auto op =
      MakeAdvectionFaceOperator< KernelPolicy >(
         fe_space,
         face_meshes,
         integration_rule,
         adv );
   op( x, y );
   GENDIL_DEVICE_SYNC;

   return y;
}

template <
   Integer Dim,
   typename KernelPolicy,
   Integer Order = 2,
   Integer NumQuad1D = Order + 2 >
Vector ApplyGlobalFaceAdvectionForExtents(
   const std::array< GlobalIndex, Dim > & extents,
   const Vector & x,
   const Vector & baseline )
{
   auto mesh = MakeCartesianMeshForExtents< Dim >( extents );
   auto face_meshes =
      MakeCartesianInteriorFaceConnectivity< Dim >( extents );

   auto orders = MakeRepeatedFiniteElementOrders< Dim, Order >();
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   auto num_quads =
      MakeRepeatedIntegrationRuleNumPoints< Dim, NumQuad1D >();
   auto integration_rule = MakeIntegrationRule( num_quads );

   return ApplyGlobalFaceAdvection< KernelPolicy >(
      fe_space,
      face_meshes,
      integration_rule,
      x,
      baseline,
      AdvectionVelocity< Dim >{} );
}

template < Integer Dim, Integer Order = 2 >
GlobalIndex GetNumberOfDofsForExtents(
   const std::array< GlobalIndex, Dim > & extents )
{
   auto mesh = MakeCartesianMeshForExtents< Dim >( extents );
   auto orders = MakeRepeatedFiniteElementOrders< Dim, Order >();
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );
   return fe_space.GetNumberOfFiniteElementDofs();
}

template < Integer Dim >
GlobalIndex GetNumberOfFacesForExtents(
   const std::array< GlobalIndex, Dim > & extents )
{
   auto face_meshes =
      MakeCartesianInteriorFaceConnectivity< Dim >( extents );
   return CountGlobalFaces( face_meshes );
}

bool CheckExpectedFaceCount(
   const char * label,
   const GlobalIndex actual_num_faces,
   const GlobalIndex expected_num_faces )
{
   if ( actual_num_faces != expected_num_faces )
   {
      std::cout << "FAILED: " << label
                << " expected " << expected_num_faces
                << " global faces, observed " << actual_num_faces
                << ".\n";
      return false;
   }
   return true;
}

template <
   Integer Dim,
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize,
   template < typename > typename FacePolicyTransform >
bool RunGlobalFaceAdvectionThreadedCase(
   const char * policy_label,
   const char * layout_label,
   const char * case_label,
   const std::array< GlobalIndex, Dim > & extents,
   const GlobalIndex expected_num_faces,
   const bool require_exact_face_count )
{
   using LegacyBase =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using Batch1Base =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using BatchNBase =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   using LegacyConfig = FacePolicyTransform< LegacyBase >;
   using DeviceBatch1 = FacePolicyTransform< Batch1Base >;
   using DeviceBatchN = FacePolicyTransform< BatchNBase >;

   const GlobalIndex actual_num_faces =
      GetNumberOfFacesForExtents< Dim >( extents );

   std::cout << "Running policy=" << policy_label
             << ", layout=" << layout_label
             << ", BatchSize=" << BatchSize
             << ", oracle=LegacyConfig+DeviceBatch1"
             << ", dim=" << Dim
             << ", mesh_extents=";
   PrintExtents( extents );
   std::cout << ", case=" << case_label
             << ", expected_faces=" << expected_num_faces
             << ", actual_faces=" << actual_num_faces << '\n';

   if ( require_exact_face_count &&
        !CheckExpectedFaceCount(
           case_label,
           actual_num_faces,
           expected_num_faces ) )
   {
      return false;
   }

   if ( !LaunchConfigurationFits< Layout, BatchSize >( layout_label ) )
   {
      return true;
   }

   const GlobalIndex ndofs = GetNumberOfDofsForExtents< Dim >( extents );
   Vector x(
      static_cast< Integer >( ndofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.3125 +
            0.043 * static_cast< Real >( i ) +
            0.078125 * static_cast< Real >( ( i * 11 ) % 19 );
      } );
   Vector baseline =
      MakeBaselineVector( static_cast< Integer >( ndofs ) );

   auto y_legacy =
      ApplyGlobalFaceAdvectionForExtents< Dim, LegacyConfig >(
         extents,
         x,
         baseline );
   auto y_batch1 =
      ApplyGlobalFaceAdvectionForExtents< Dim, DeviceBatch1 >(
         extents,
         x,
         baseline );
   auto y_batchn =
      ApplyGlobalFaceAdvectionForExtents< Dim, DeviceBatchN >(
         extents,
         x,
         baseline );

   constexpr Real tolerance = 1e-10;
   bool success = true;
   success =
      CheckVectorFinite( "LegacyConfig", y_legacy ) && success;
   success =
      CheckVectorFinite( "DeviceBatch1", y_batch1 ) && success;
   success =
      CheckVectorFinite( "DeviceBatchN", y_batchn ) && success;
   success =
      CheckScaledL2Close(
         "DeviceBatchN vs LegacyConfig",
         y_batchn,
         y_legacy,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;
   success =
      CheckScaledL2Close(
         "DeviceBatch1 vs LegacyConfig",
         y_batch1,
         y_legacy,
         tolerance ) && success;

   if ( actual_num_faces == 0 )
   {
      success =
         CheckScaledL2Close(
            "zero-face DeviceBatchN baseline preservation",
            y_batchn,
            baseline,
            tolerance ) && success;
   }

   return success;
}

template <
   Integer Dim,
   Integer BatchSize,
   template < typename > typename FacePolicyTransform >
bool RunGlobalFaceAdvectionRegisterOnlyCase(
   const char * policy_label,
   const char * case_label,
   const std::array< GlobalIndex, Dim > & extents,
   const GlobalIndex expected_num_faces,
   const bool require_exact_face_count )
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;
   using Batch1Base =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using BatchNBase =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   using DeviceBatch1 = FacePolicyTransform< Batch1Base >;
   using DeviceBatchN = FacePolicyTransform< BatchNBase >;

   const GlobalIndex actual_num_faces =
      GetNumberOfFacesForExtents< Dim >( extents );

   std::cout << "Running policy=" << policy_label
             << ", layout=ThreadBlockLayout<>"
             << ", BatchSize=" << BatchSize
             << ", oracle=DeviceBatch1"
             << ", dim=" << Dim
             << ", mesh_extents=";
   PrintExtents( extents );
   std::cout << ", case=" << case_label
             << ", expected_faces=" << expected_num_faces
             << ", actual_faces=" << actual_num_faces << '\n';

   if ( require_exact_face_count &&
        !CheckExpectedFaceCount(
           case_label,
           actual_num_faces,
           expected_num_faces ) )
   {
      return false;
   }

   if ( !LaunchConfigurationFits< Layout, BatchSize >(
           "ThreadBlockLayout<>" ) )
   {
      return true;
   }

   const GlobalIndex ndofs = GetNumberOfDofsForExtents< Dim >( extents );
   Vector x(
      static_cast< Integer >( ndofs ),
      [] GENDIL_HOST_DEVICE ( Integer i )
      {
         return 0.25 +
            0.03125 * static_cast< Real >( i ) +
            0.09375 * static_cast< Real >( ( i * 7 ) % 13 );
      } );
   Vector baseline =
      MakeBaselineVector( static_cast< Integer >( ndofs ) );

   auto y_batch1 =
      ApplyGlobalFaceAdvectionForExtents< Dim, DeviceBatch1 >(
         extents,
         x,
         baseline );
   auto y_batchn =
      ApplyGlobalFaceAdvectionForExtents< Dim, DeviceBatchN >(
         extents,
         x,
         baseline );

   constexpr Real tolerance = 1e-10;
   bool success = true;
   success =
      CheckVectorFinite( "DeviceBatch1", y_batch1 ) && success;
   success =
      CheckVectorFinite( "DeviceBatchN", y_batchn ) && success;
   success =
      CheckScaledL2Close(
         "Register-only DeviceBatchN vs DeviceBatch1",
         y_batchn,
         y_batch1,
         tolerance ) && success;

   if ( actual_num_faces == 0 )
   {
      success =
         CheckScaledL2Close(
            "zero-face DeviceBatchN baseline preservation",
            y_batchn,
            baseline,
            tolerance ) && success;
   }

   return success;
}

template < Integer BatchSize, typename Lambda >
bool RunNormalizedFaceCountCases( Lambda && run_case )
{
   const std::array< GlobalIndex, 6 > candidates{
      0,
      1,
      BatchSize - 1,
      BatchSize,
      BatchSize + 1,
      10
   };
   std::array< GlobalIndex, 6 > cases{};
   Integer num_cases = 0;

   for ( const GlobalIndex candidate : candidates )
   {
      bool found = false;
      for ( Integer i = 0; i < num_cases; ++i )
      {
         found = found || cases[ i ] == candidate;
      }
      if ( !found )
      {
         cases[ num_cases++ ] = candidate;
      }
   }

   bool success = true;
   for ( Integer i = 0; i < num_cases; ++i )
   {
      const GlobalIndex expected_num_faces = cases[ i ];
      const std::array< GlobalIndex, 1 > extents{ expected_num_faces + 1 };
      success = run_case( expected_num_faces, extents ) && success;
   }
   return success;
}

template <
   Integer BatchSize,
   template < typename > typename FacePolicyTransform >
bool RunRegisterOnlyExactFaceCountCases( const char * policy_label )
{
   return RunNormalizedFaceCountCases< BatchSize >(
      [=] (
         const GlobalIndex expected_num_faces,
         const std::array< GlobalIndex, 1 > & extents )
      {
         return RunGlobalFaceAdvectionRegisterOnlyCase<
            1,
            BatchSize,
            FacePolicyTransform >(
               policy_label,
               "1d-exact-face-count",
               extents,
               expected_num_faces,
               true );
      } );
}

template <
   Integer BatchSize,
   template < typename > typename FacePolicyTransform >
bool RunThreadedExactFaceCountCases( const char * policy_label )
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;

   return RunNormalizedFaceCountCases< BatchSize >(
      [=] (
         const GlobalIndex expected_num_faces,
         const std::array< GlobalIndex, 1 > & extents )
      {
         return RunGlobalFaceAdvectionThreadedCase<
            1,
            Layout,
            MaxSharedDimensions,
            BatchSize,
            FacePolicyTransform >(
               policy_label,
               "ThreadBlockLayout<num_quad_1d>",
               "1d-exact-face-count",
               extents,
               expected_num_faces,
               true );
      } );
}

template <
   Integer BatchSize,
   template < typename > typename FacePolicyTransform >
bool RunRepresentativeMeshCases( const char * policy_label )
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;

   bool success = true;
   success =
      RunGlobalFaceAdvectionRegisterOnlyCase<
         2,
         BatchSize,
         FacePolicyTransform >(
            policy_label,
            "nondegenerate-2d",
            std::array< GlobalIndex, 2 >{ 3, 4 },
            GlobalIndex{ 0 },
            false ) && success;
   success =
      RunGlobalFaceAdvectionThreadedCase<
         2,
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FacePolicyTransform >(
            policy_label,
            "ThreadBlockLayout<num_quad_1d>",
            "nondegenerate-2d",
            std::array< GlobalIndex, 2 >{ 3, 4 },
            GlobalIndex{ 0 },
            false ) && success;
   success =
      RunGlobalFaceAdvectionRegisterOnlyCase<
         3,
         BatchSize,
         FacePolicyTransform >(
            policy_label,
            "compact-3d",
            std::array< GlobalIndex, 3 >{ 3, 2, 2 },
            GlobalIndex{ 0 },
            false ) && success;
   success =
      RunGlobalFaceAdvectionThreadedCase<
         3,
         Layout,
         MaxSharedDimensions,
         BatchSize,
         FacePolicyTransform >(
            policy_label,
            "ThreadBlockLayout<num_quad_1d>",
            "compact-3d",
            std::array< GlobalIndex, 3 >{ 3, 2, 2 },
            GlobalIndex{ 0 },
            false ) && success;
   return success;
}

template <
   Integer BatchSize,
   template < typename > typename FacePolicyTransform >
bool RunBroadBatchCases( const char * policy_label )
{
   bool success = true;
   success =
      RunRegisterOnlyExactFaceCountCases<
         BatchSize,
         FacePolicyTransform >( policy_label ) && success;
   success =
      RunThreadedExactFaceCountCases<
         BatchSize,
         FacePolicyTransform >( policy_label ) && success;
   success =
      RunRepresentativeMeshCases<
         BatchSize,
         FacePolicyTransform >( policy_label ) && success;
   return success;
}

template < template < typename > typename FacePolicyTransform >
bool RunFullSharedRepresentativeCases( const char * policy_label )
{
   bool success = true;
   success =
      RunRegisterOnlyExactFaceCountCases<
         2,
         FacePolicyTransform >( policy_label ) && success;
   success =
      RunThreadedExactFaceCountCases<
         2,
         FacePolicyTransform >( policy_label ) && success;
   success =
      RunRegisterOnlyExactFaceCountCases<
         device_warp_size,
         FacePolicyTransform >( policy_label ) && success;
   success =
      RunThreadedExactFaceCountCases<
         device_warp_size,
         FacePolicyTransform >( policy_label ) && success;
   success =
      RunRepresentativeMeshCases<
         2,
         FacePolicyTransform >( policy_label ) && success;
   return success;
}

bool RunDefaultDirectGlobalCases()
{
   bool success = true;
   success =
      RunBroadBatchCases< 1, DefaultFacePolicies >(
         "DefaultDirectGlobal" ) && success;
   success =
      RunBroadBatchCases< 2, DefaultFacePolicies >(
         "DefaultDirectGlobal" ) && success;
   success =
      RunBroadBatchCases< 4, DefaultFacePolicies >(
         "DefaultDirectGlobal" ) && success;
   success =
      RunBroadBatchCases<
         device_warp_size,
         DefaultFacePolicies >(
            "DefaultDirectGlobal" ) && success;
   return success;
}

void CheckSharedMemoryRequirements()
{
   using Space =
      FiniteElementSpace<
         Cartesian1DMesh,
         GLLFiniteElement< 2 >,
         L2Restriction >;
   using Rule =
      decltype( MakeIntegrationRule( IntegrationRuleNumPoints< 4 >{} ) );
   using DirectGlobalEmpty =
      DeviceKernelConfiguration< ThreadBlockLayout<>, 0, 2 >;
   using FullSharedEmpty =
      FullSharedFacePolicies< DirectGlobalEmpty >;
   using DirectGlobalThreaded =
      DeviceKernelConfiguration< ThreadBlockLayout< 4 >, 1, 2 >;
   using FullSharedThreaded =
      FullSharedFacePolicies< DirectGlobalThreaded >;

   static_assert(
      global_face_advection_required_shared_memory_v<
         DirectGlobalEmpty,
         Rule,
         Space > ==
      required_shared_memory_v< DirectGlobalEmpty, Rule > );
   static_assert(
      global_face_advection_required_shared_memory_v<
         FullSharedEmpty,
         Rule,
         Space > ==
      required_shared_memory_v< FullSharedEmpty, Rule > );
   static_assert(
      global_face_advection_required_shared_memory_v<
         DirectGlobalThreaded,
         Rule,
         Space > ==
      required_shared_memory_v< DirectGlobalThreaded, Rule > );
   static_assert(
      global_face_advection_required_shared_memory_v<
         FullSharedThreaded,
         Rule,
         Space > ==
      Max(
         Space::finite_element_type::GetNumDofs(),
         required_shared_memory_v< FullSharedThreaded, Rule > ) );
}

} // namespace

int main()
{
   CheckSharedMemoryRequirements();

   bool success = true;
   success = RunDefaultDirectGlobalCases() && success;
   success =
      RunFullSharedRepresentativeCases< FullSharedFacePolicies >(
         "FullSharedOverride" ) && success;

   return success ? 0 : 1;
}

#endif
