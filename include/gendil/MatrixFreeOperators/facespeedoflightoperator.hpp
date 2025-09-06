// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

enum class FaceSoLType
{
   ReadCell,
   WriteCell
};

/**
 * @brief Implementation of the "face" speed-of-light operator at the element level.
 * 
 * @tparam KernelContext The type of the kernel context.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @param kernel_conf The kernel configuration.
 * @param fe_space The finite element space.
 * @param element_index The index of the finite element in the finite element space.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelContext,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
void FaceReadSpeedOfLightElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   FaceLoop( fe_space, element_index,
      [&]( auto const & face_info )
      {
         auto neighbor_u = ReadDofs( kernel_conf, fe_space, face_info, dofs_in );

         u += neighbor_u;
      }
   );

   WriteDofs( kernel_conf, fe_space, element_index, u, dofs_out );
}

template <
   typename KernelContext,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
void FaceWriteSpeedOfLightElementOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const GlobalIndex element_index,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   // Read the current element DOFs
   auto u = ReadDofs( kernel_conf, fe_space, element_index, dofs_in );

   FaceLoop( fe_space, element_index,
      [&]( auto const & face_info )
      {
         // Write the result back to the neighbor
         WriteAddDofs( kernel_conf, fe_space, face_info, u, dofs_out );
      }
   );

   // Write own contribution to own dofs_out
   WriteAddDofs( kernel_conf, fe_space, element_index, u, dofs_out );
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename FaceMesh >
GENDIL_HOST_DEVICE
void ReadWriteSpeedOfLightFaceOperator(
   const KernelContext & kernel_conf,
   const FiniteElementSpace & fe_space,
   const FaceMesh & face_mesh,
   const GlobalIndex face_index,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   const auto face_info = face_mesh.GetGlobalFaceInfo( face_index );
   
   // Read the current element DOFs
   auto u_minus = ReadDofs( kernel_conf, fe_space, face_info.minus_side(), dofs_in );
   auto u_plus = ReadDofs( kernel_conf, fe_space, face_info.plus_side(), dofs_in );

   u_minus += u_plus;

   // Write own contribution to own dofs_out
   WriteAddDofs( kernel_conf, fe_space, face_info.minus_side(), u_minus, dofs_out );
   WriteAddDofs( kernel_conf, fe_space, face_info.plus_side(), u_minus, dofs_out );
}
/**
 * @brief Explicit speed-of-light "face" operator.
 * 
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam FaceMesh The type of face mesh associated to the operator.
 * @param fe_space The finite element space.
 * @param face_mesh The face mesh.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename FiniteElementSpace,
   typename FaceMesh >
void FaceSpeedOfLightExplicitFaceOperator(
   const FiniteElementSpace & fe_space,
   const FaceMesh & face_mesh,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::GlobalFaceIterator<KernelConfiguration>(
      face_mesh,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex face_index ) mutable
      {
         constexpr size_t required_shared_mem = FiniteElementSpace::finite_element_type::GetNumDofs();
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         ReadWriteSpeedOfLightFaceOperator(
            kernel_conf,
            fe_space,
            face_mesh,
            face_index,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Explicit speed-of-light "face" operator.
 * 
 * @tparam KernelType The type of the kernel (read, write, or face).
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @param fe_space The finite element space.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   FaceSoLType KernelType,
   typename KernelConfiguration,
   typename FiniteElementSpace >
void FaceSpeedOfLightExplicitOperator(
   const FiniteElementSpace & fe_space,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   if constexpr ( KernelType == FaceSoLType::ReadCell )
      mesh::CellIterator<KernelConfiguration>(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = FiniteElementSpace::finite_element_type::GetNumDofs();
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         FaceReadSpeedOfLightElementOperator(
            kernel_conf,
            fe_space,
            element_index,
            dofs_in,
            dofs_out );
      });
   else if constexpr ( KernelType == FaceSoLType::WriteCell )
      mesh::CellIterator<KernelConfiguration>(
      fe_space,
      [=] GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
      {
         constexpr size_t required_shared_mem = FiniteElementSpace::finite_element_type::GetNumDofs();
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];

         KernelContext< KernelConfiguration, required_shared_mem > kernel_conf( _shared_mem );

         FaceWriteSpeedOfLightElementOperator(
            kernel_conf,
            fe_space,
            element_index,
            dofs_in,
            dofs_out );
      });
}


/**
 * @brief Explicit speed-of-light "face" operator.
 * 
 * @tparam KernelType The type of the kernel (read, write, or face).
 * @tparam KernelConfiguration The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @param fe_space The finite element space.
 * @param dofs_in The input degrees of freedom.
 * @param dofs_out The output degrees of freedom.
 */
template <
   typename KernelConfiguration,
   typename FiniteElementSpace,
   typename FaceMesh >
void GlobalFaceSpeedOfLightExplicitOperator(
   const FiniteElementSpace & fe_space,
   const FaceMesh & face_meshes,
   const StridedView< FiniteElementSpace::Dim + 1, const Real > & dofs_in,
   StridedView< FiniteElementSpace::Dim + 1, Real > & dofs_out )
{
   mesh::ForEachFaceMesh(
      face_meshes,
      [&] ( const auto & face_mesh ) mutable
      {
         FaceSpeedOfLightExplicitFaceOperator< KernelConfiguration >(
            fe_space,
            face_mesh,
            dofs_in,
            dofs_out );
      }
   );
}

/**
 * @brief Represent a speed-of-light face operator.
 * 
 * @tparam Read Whether the operator reads or writes the neighbors degrees of freedom.
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 */
template <
   FaceSoLType KernelType,
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
class FaceSpeedOfLightOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;

   using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

public:
   /**
    * @brief Construct a new FaceSpeedOfLightOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    */
   FaceSpeedOfLightOperator( const FiniteElementSpace & finite_element_space,
                             const IntegrationRule & int_rules ) :
      base( finite_element_space, int_rules )
   { }

   /**
    * @brief Apply the speed-of-light face operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      FaceSpeedOfLightExplicitOperator< KernelType, KernelPolicy >
         ( this->finite_element_space,
           dofs_in,
           dofs_out );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the speed-of-light face operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   void Mult( const mfem::Vector & dofs_vector_in,
              mfem::Vector & dofs_vector_out ) const override
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }
   #endif // GENDIL_USE_MFEM
};

/**
 * @brief Represent a speed-of-light global face operator.
 * 
 * @tparam Read Whether the operator reads or writes the neighbors degrees of freedom.
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The finite element space associated to the operator.
 * @tparam IntegrationRule The integration rule used by the operator.
 * @tparam Adv The type of the function to evaluate the advection vector at physical coordinates.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename FaceMeshes,
   typename IntegrationRule >
class GlobalFaceSpeedOfLightOperator
   : public MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >
{
   using base = MatrixFreeBilinearFiniteElementOperator< FiniteElementSpace, IntegrationRule >;

   using input = StridedView< FiniteElementSpace::Dim + 1, const Real >;
   using output = StridedView< FiniteElementSpace::Dim + 1, Real >;

   const FaceMeshes & face_meshes;

public:
   /**
    * @brief Construct a new FaceSpeedOfLightOperator object.
    * 
    * @param finite_element_space The finite element space associated to the operator.
    * @param int_rules The integration rule used by the operator.
    */
   GlobalFaceSpeedOfLightOperator(
      const FiniteElementSpace & finite_element_space,
      const FaceMeshes & face_meshes,
      const IntegrationRule & int_rules ) :
      base( finite_element_space, int_rules ),
      face_meshes( face_meshes )
   { }

   /**
    * @brief Apply the speed-of-light face operator.
    * 
    * @param dofs_in The input degrees of freedom.
    * @param dofs_out The output degrees of freedom.
    */
   void Apply( const input & dofs_in,
               output & dofs_out ) const
   {
      GlobalFaceSpeedOfLightExplicitOperator< KernelPolicy >
         ( this->finite_element_space,
           this->face_meshes,
           dofs_in,
           dofs_out );
   }

   void operator()( const Vector & dofs_vector_in, Vector & dofs_vector_out ) const
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }

   #ifdef GENDIL_USE_MFEM
   /**
    * @brief Apply the speed-of-light face operator.
    * 
    * @param dofs_vector_in The input degrees of freedom.
    * @param dofs_vector_out The output degrees of freedom.
    */
   void Mult( const mfem::Vector & dofs_vector_in,
              mfem::Vector & dofs_vector_out ) const override
   {
      auto dofs_in = MakeReadOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyEVectorView< KernelPolicy >( this->finite_element_space, dofs_vector_out );

      Apply( dofs_in, dofs_out );
   }
   #endif // GENDIL_USE_MFEM
};

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   return FaceSpeedOfLightOperator<
             FaceSoLType::ReadCell,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return FaceSpeedOfLightOperator<
             FaceSoLType::ReadCell,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeWriteFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   return FaceSpeedOfLightOperator<
             FaceSoLType::WriteCell,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename FiniteElementSpace,
   typename IntegrationRule >
auto MakeWriteFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const IntegrationRule & int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return FaceSpeedOfLightOperator<
             FaceSoLType::WriteCell,
             KernelPolicy,
             FiniteElementSpace,
             IntegrationRule
          >( finite_element_space, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam KernelPolicy The execution policy for the hardware.
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename KernelPolicy,
   typename FiniteElementSpace,
   typename FacesMeshes,
   typename IntegrationRule >
auto MakeGlobalFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const FacesMeshes & face_meshes,
   const IntegrationRule & int_rule )
{
   return GlobalFaceSpeedOfLightOperator<
             KernelPolicy,
             FiniteElementSpace,
             FacesMeshes,
             IntegrationRule
          >( finite_element_space, face_meshes, int_rule );
}

/**
 * @brief Factory to build speed-of-light face operators. Useful to hide
 * the type of the operator.
 * 
 * @tparam FiniteElementSpace The type of the finite element space associated to the operator.
 * @tparam IntegrationRule The type of the integration rule used by the operator.
 * @param finite_element_space The finite element space associated to the operator.
 * @param int_rule The integration rule used by the operator.
 * @return auto The advection operator.
 */
template <
   typename FiniteElementSpace,
   typename FacesMeshes,
   typename IntegrationRule >
auto MakeGlobalFaceSpeedOfLightOperator(
   const FiniteElementSpace & finite_element_space,
   const FacesMeshes & face_meshes,
   const IntegrationRule & int_rule )
{
   using KernelPolicy = SerialKernelConfiguration;

   return GlobalFaceSpeedOfLightOperator<
             KernelPolicy,
             FiniteElementSpace,
             FacesMeshes,
             IntegrationRule
          >( finite_element_space, face_meshes, int_rule );
}

}
