// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"

namespace gendil
{

template <
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   size_t... Dims,
   typename KernelConf,
   typename Input >
GENDIL_HOST_DEVICE
auto SerialAggregateDimensions(
   const KernelConf & kernel_conf,
   const Input & u
)
{
   auto v = MakeQuadraturePointValuesContainer( kernel_conf, TestIntegrationRule{} );
   QuadraturePointLoop< TestIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      WriteQuadratureLocalValues( kernel_conf, quad_index, 0.0, v );
   });
   QuadraturePointLoop< TrialIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      auto u_q = ReadQuadratureLocalValues( kernel_conf, quad_index, u );

      // auto sub_quad_index = qud_index.template Sub< 0, TestDim >();
      auto sub_quad_index = GetSubIndex< Dims ... >( quad_index );

      WriteAddQuadratureLocalValues( kernel_conf, sub_quad_index, u_q, v );
   });

   return v;
}

// TODO Do without IntegrationRule?
template <
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   size_t... Dims,
   typename KernelConf,
   typename Input >
GENDIL_HOST_DEVICE
auto ThreadedAggregateDimensions(
   const KernelConf & kernel_conf,
   const Input & u
)
{
   // TODO Use a shared memory slice.
   auto v_shared = MakeSharedQuadraturePointValuesContainer( kernel_conf, TestIntegrationRule{} );
   // TODO set v_shared to 0?
   QuadraturePointLoop< TrialIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      auto u_q = ReadQuadratureLocalValues( kernel_conf, quad_index, u );

      // auto sub_quad_index = qud_index.template Sub< 0, TestDim >();
      auto sub_quad_index = GetSubIndex< Dims ... >( quad_index );

      WriteAddQuadratureLocalValues( kernel_conf, sub_quad_index, u_q, v_shared );
   });
   kernel_conf.Synchronize();

   auto v = MakeQuadraturePointValuesContainer( kernel_conf, TestIntegrationRule{} );
   QuadraturePointLoop< TestIntegrationRule >( kernel_conf, [&] ( auto const & quad_index )
   {
      const auto v_q = Apply( v_shared, quad_index );

      WriteQuadratureLocalValues( kernel_conf, quad_index, v_q, v );
   });
   kernel_conf.Synchronize();
   kernel_conf.SharedAllocator.reset();

   return v;
}

template <
   typename TrialIntegrationRule,
   typename TestIntegrationRule,
   size_t... Dims,
   typename KernelConf,
   typename Input >
GENDIL_HOST_DEVICE
auto AggregateDimensions(
   const KernelConf & kernel_conf,
   const Input & u,
   std::index_sequence< Dims ... >
)
{
   if constexpr ( is_serial_v< KernelConf > )
   {
      return SerialAggregateDimensions< TrialIntegrationRule, TestIntegrationRule, Dims... >( kernel_conf, u );
   }
   else
   {
      return ThreadedAggregateDimensions< TrialIntegrationRule, TestIntegrationRule, Dims... >( kernel_conf, u );
   }
}

}