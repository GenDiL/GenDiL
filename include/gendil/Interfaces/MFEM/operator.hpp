// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

/**
 * @file
 * Use a GenDiL operator or sparse matrix with MFEM solvers without converting
 * its matrix or vectors:
 *
 * @code
 * auto mfem_operator = gendil::MakeMFEMOperator(gendil_operator);
 * mfem::CGSolver solver;
 * solver.SetOperator(mfem_operator);
 * @endcode
 *
 * Lvalues are borrowed; rvalues are stored in the adapter:
 *
 * @code
 * auto owned = gendil::MakeMFEMOperator(std::move(matrix));
 * @endcode
 *
 * Application uses MFEM vectors and preserves the source's algebraic numbering
 * and execution backend. Choose a solver appropriate for the source: a
 * nonlinear vector map stays nonlinear.
 */

#include "gendil/Algebra/vectormapconcept.hpp"
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"
#include "gendil/Utilities/debug.hpp"

#include <mfem.hpp>

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace gendil {

template<class KernelPolicy, class IntegrationRule, class WeakForm, class WeakFormContext>
class PullbackGenericOperator;

/**
 * Describe where a custom vector map executes when used through MFEM.
 *
 * Generic operators and sparse backends are recognized automatically. For
 * another map, specialize this trait on its unqualified type and provide
 * `static constexpr bool on_device`: false for host execution, true for the
 * configured GenDiL CUDA/HIP device. The trait describes execution; it does
 * not change the map's backend or add vector access capabilities.
 */
template<class Map>
struct MFEMOperatorTraits { };

template<class Map>
   requires requires { typename Map::backend_type; }
struct MFEMOperatorTraits<Map>
{
   static_assert(
      is_host_matvec_backend_v<typename Map::backend_type> !=
         is_device_matvec_backend_v<typename Map::backend_type>,
      "MFEM operator backend must identify exactly one host or device placement; "
      "specialize MFEMOperatorTraits for custom maps.");

   static constexpr bool on_device =
      is_device_matvec_backend_v<typename Map::backend_type>;
};

template<class Policy, class Rule, class Form, class Context>
struct MFEMOperatorTraits<PullbackGenericOperator<Policy, Rule, Form, Context>>
{
   static constexpr bool on_device = is_device_configuration_v<Policy>;
};

namespace mfem_operator_detail {

template<class Map>
concept AdaptableMap =
   VectorMap<std::remove_reference_t<Map>, mfem::Vector, mfem::Vector> &&
   (std::is_lvalue_reference_v<Map&&> ||
    std::is_constructible_v<std::remove_cvref_t<Map>, Map&&>);

template<class Map>
constexpr bool OnDevice()
{
   static_assert(requires { MFEMOperatorTraits<Map>::on_device; },
      "MFEM operator placement is unknown; specialize MFEMOperatorTraits<Map> "
      "with static constexpr bool on_device.");
   if constexpr (requires { MFEMOperatorTraits<Map>::on_device; })
      return MFEMOperatorTraits<Map>::on_device;
   else
      return false;
}

inline int CheckedSize(const std::size_t size, const char* message)
{
   GENDIL_VERIFY(size <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
      message);
   return static_cast<int>(size);
}

template<bool OnDevice>
void ValidateDevice()
{
   if constexpr (OnDevice)
   {
#if defined(GENDIL_USE_CUDA)
      GENDIL_VERIFY(mfem::Device::Allows(mfem::Backend::CUDA_MASK),
         "MFEM operator requires a CUDA-enabled MFEM device.");
#elif defined(GENDIL_USE_HIP)
      GENDIL_VERIFY(mfem::Device::Allows(mfem::Backend::HIP_MASK),
         "MFEM operator requires a HIP-enabled MFEM device.");
#else
      static_assert(!OnDevice,
         "MFEM device operators require GenDiL CUDA or HIP support.");
#endif
#if defined(GENDIL_USE_CUDA) || defined(GENDIL_USE_HIP)
      int current_device = -1;
      GENDIL_DEVICE_CHECK(DeviceGetDevice(&current_device));
      GENDIL_VERIFY(current_device == mfem::Device::GetId(),
         "MFEM operator requires matching GenDiL and MFEM device IDs.");
#endif
   }
}

} // namespace mfem_operator_detail

template<class Map>
   requires mfem_operator_detail::AdaptableMap<Map>
auto MakeMFEMOperator(Map&& map);

/**
 * An MFEM operator adapting a GenDiL vector map; construct with MakeMFEMOperator.
 * The factory selects borrowed storage for lvalues and owned storage for rvalues.
 * Copying or moving an owning adapter follows the stored map's capabilities;
 * an adapter owning a move-only matrix is itself move-only.
 *
 * Only forward application and scaled addition are provided. Transpose,
 * Jacobian, and diagonal operations retain MFEM's unsupported defaults.
 */
template<class Map, bool OwnsMap = false>
class MFEMOperator : public mfem::Operator
{
   static constexpr bool on_device = mfem_operator_detail::OnDevice<Map>();
   std::conditional_t<OwnsMap, Map, const Map&> map;

   template<class Source>
      requires std::is_same_v<std::remove_cvref_t<Source>, Map> &&
         std::is_constructible_v<
            std::conditional_t<OwnsMap, Map, const Map&>, Source&&>
   explicit MFEMOperator(Source&& source)
      : mfem::Operator(
           mfem_operator_detail::CheckedSize(GetOutputSize(std::as_const(source)),
              "MFEM operator output size exceeds INT_MAX."),
           mfem_operator_detail::CheckedSize(GetInputSize(std::as_const(source)),
              "MFEM operator input size exceeds INT_MAX.")),
        map(std::forward<Source>(source))
   {
      mfem_operator_detail::ValidateDevice<on_device>();
   }

   template<class Source>
      requires mfem_operator_detail::AdaptableMap<Source>
   friend auto MakeMFEMOperator(Source&& source);

   void ValidateVectors(const mfem::Vector& x, const mfem::Vector& y) const
   {
      GENDIL_VERIFY(x.Size() == Width(),
         "MFEM operator input vector has the wrong size.");
      GENDIL_VERIFY(y.Size() == Height(),
         "MFEM operator output vector has the wrong size.");
   }

public:
   mfem::MemoryClass GetMemoryClass() const override
   {
      if constexpr (on_device)
         return mfem::Device::GetMemoryClass();
      else
         return mfem::MemoryClass::HOST;
   }

   void Mult(const mfem::Vector& x, mfem::Vector& y) const override
   {
      ValidateVectors(x, y);
      Apply(map, x, y);
   }

   void AddMult(const mfem::Vector& x, mfem::Vector& y,
                const mfem::real_t a = 1.0) const override
   {
      ValidateVectors(x, y);
      if (a == 0.0) { return; }
      if (a == 1.0)
         ApplyAdd(map, x, y);
      else
         mfem::Operator::AddMult(x, y, a);
   }
};

/**
 * Expose an operator or sparse matrix through MFEM's operator interface.
 *
 * Const and mutable lvalues are borrowed without copying. Their source objects
 * must remain alive and unmoved while the adapter is used. Rvalues are owned
 * by value, moving the source when possible; const rvalues require a copyable
 * source. An owning adapter can outlive the original source object.
 *
 * Ownership of the map does not extend the lifetime of any mesh, restrictions,
 * coefficients, or other storage the map itself borrows. Those dependencies
 * must remain valid. Dimensions must stay fixed; value updates follow the
 * source's normal update contract. An MFEM solver retaining the adapter
 * requires that adapter to remain alive and unmoved.
 *
 * Mult and AddMult require exactly sized, non-overlapping input/output vectors.
 * Existing source concurrency restrictions still apply. Configure MFEM's
 * matching CUDA/HIP device before wrapping a device map, and keep its device
 * configuration and the current device stable during use. No DoF renumbering,
 * constraint elimination, or parallel assembly is performed.
 */
template<class Map>
   requires mfem_operator_detail::AdaptableMap<Map>
auto MakeMFEMOperator(Map&& map)
{
   return MFEMOperator<std::remove_cvref_t<Map>,
      !std::is_lvalue_reference_v<Map&&>>(std::forward<Map>(map));
}

} // namespace gendil

#endif // GENDIL_USE_MFEM
