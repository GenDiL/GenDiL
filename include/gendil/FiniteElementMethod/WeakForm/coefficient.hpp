// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/readquadraturelocalvalues.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/applymapping.hpp"

#include <type_traits>
#include <utility>

namespace gendil
{

struct ElementIndex         : CoefficientInputBase {};
struct QuadraturePointIndex : CoefficientInputBase {};
struct PhysicalCoordinate   : CoefficientInputBase {};
struct ReferenceCoordinate  : CoefficientInputBase {};
struct JacobianDeterminant  : CoefficientInputBase {};
struct Jacobian             : CoefficientInputBase {};
struct JacobianInverse      : CoefficientInputBase {};

/**
 * @brief Inverse facet size coefficient input: ||J^{-T} n_ref||
 *
 * Available only with FacetQuadraturePointContext.
 * For Cartesian mesh with spacing h, returns 1/h.
 *
 * Used for SIPDG penalty scaling: tau = kappa * h^{-1}
 */
struct InverseFacetSize : CoefficientInputBase {};

template <StaticString Name>
struct FieldValue : CoefficientInputBase
{
   static constexpr auto name = Name;
};

template <StaticString Name>
struct FieldGradient : CoefficientInputBase
{
   static constexpr auto name = Name;
};

// -----------------------------------------------------------------------------
// Reusable field readers
// -----------------------------------------------------------------------------

template <
   StaticString Name,
   typename KernelContext,
   typename QuadPtContext,
   typename Fields >
GENDIL_HOST_DEVICE
decltype(auto) ReadFieldValueAtQuadraturePoint(
   const KernelContext& kernel_context,
   const QuadPtContext& quad_pt_context,
   const Fields& fields)
{
   return ReadQuadratureLocalValues(
      kernel_context,
      quad_pt_context.quad_index,
      fields.template get<NameTag<Name>>().values);
}

template <
   StaticString Name,
   typename KernelContext,
   typename QuadPtContext,
   typename Fields >
GENDIL_HOST_DEVICE
decltype(auto) ReadFieldGradientAtQuadraturePoint(
   const KernelContext& kernel_context,
   const QuadPtContext& quad_pt_context,
   const Fields& fields)
{
   auto grad_q = ReadQuadratureLocalGradients(
      kernel_context,
      quad_pt_context.quad_index,
      fields.template get<NameTag<Name>>().gradients);
   ApplyMapping(quad_pt_context.inv_J_mesh, grad_q);
   return grad_q;
}

// -----------------------------------------------------------------------------
// Input extraction
// -----------------------------------------------------------------------------

template <typename InputTag>
struct CoefficientInputGetter;

template <>
struct CoefficientInputGetter<ElementIndex>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext& element_context,
      const QuadPtContext&,
      const Fields&)
   {
      return (element_context.element_index);
   }
};

template <>
struct CoefficientInputGetter<QuadraturePointIndex>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&)
   {
      return (quad_pt_context.quad_index);
   }
};

template <>
struct CoefficientInputGetter<PhysicalCoordinate>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&)
   {
      if constexpr (requires { quad_pt_context.X; })
      {
         return (quad_pt_context.X);
      }
      else if constexpr (requires { quad_pt_context.X_minus; })
      {
         return (quad_pt_context.X_minus);
      }
      else
      {
         // QuadPtContext has no physical coordinates field, cannot provide PhysicalCoordinate input
         static_assert(requires { quad_pt_context.X; },
            "PhysicalCoordinate coefficient input requires QuadPtContext with physical coordinates (e.g., GenericQuadraturePointContext or FacetQuadraturePointContext)");

         // Unreachable, but needed for return type deduction
         return Real{};
      }
   }
};

template <>
struct CoefficientInputGetter<ReferenceCoordinate>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&)
   {
      using IntegrationRule = typename OperatorContext::integration_rule_type;
      return IntegrationRule::GetPoint(quad_pt_context.quad_index);
   }
};

template <>
struct CoefficientInputGetter<JacobianDeterminant>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&)
   {
      return (quad_pt_context.det_J);
   }
};

template <>
struct CoefficientInputGetter<Jacobian>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&)
   {
      return (quad_pt_context.J_mesh);
   }
};

template <>
struct CoefficientInputGetter<JacobianInverse>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&)
   {
      if constexpr (requires { quad_pt_context.inv_J_mesh; })
      {
         return (quad_pt_context.inv_J_mesh);
      }
      else
      {
         static_assert(requires { quad_pt_context.inv_J_mesh; },
            "JacobianInverse coefficient input requires QuadPtContext with inv_J_mesh field");
         return Empty{};
      }
   }
};

/**
 * @brief Getter for InverseFacetSize coefficient input
 *
 * Accesses quad_pt_context.inverse_facet_size from FacetQuadraturePointContext.
 * For cell contexts without inverse_facet_size field, produces compile error.
 */
template <>
struct CoefficientInputGetter<InverseFacetSize>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&)
   {
      // Check if quad_pt_context has inverse_facet_size field (FacetQuadraturePointContext)
      if constexpr (requires { quad_pt_context.inverse_facet_size; })
      {
         return quad_pt_context.inverse_facet_size;
      }
      else if constexpr (requires { quad_pt_context.inverse_facet_size_minus; })
      {
         // Interior facets use the current/minus-side facet size for now.
         return quad_pt_context.inverse_facet_size_minus;
      }
      else
      {
         // InverseFacetSize requires FacetQuadraturePointContext
         static_assert(requires { quad_pt_context.inverse_facet_size; },
            "InverseFacetSize coefficient input requires FacetQuadraturePointContext with inverse_facet_size field");

         // Unreachable, but needed for return type deduction
         return Real{};
      }
   }
};

// New: named field value input
template <StaticString Name>
struct CoefficientInputGetter<FieldValue<Name>>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext& kernel_context,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields& fields)
   {
      return ReadFieldValueAtQuadraturePoint<Name>(
         kernel_context,
         quad_pt_context,
         fields);
   }
};

// New: named field gradient input
template <StaticString Name>
struct CoefficientInputGetter<FieldGradient<Name>>
{
   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   static decltype(auto) Get(
      const KernelContext& kernel_context,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields& fields)
   {
      return ReadFieldGradientAtQuadraturePoint<Name>(
         kernel_context,
         quad_pt_context,
         fields);
   }
};

template <
   typename InputTag,
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename QuadPtContext,
   typename Fields >
GENDIL_HOST_DEVICE
decltype(auto) GetCoefficientInput(
   const KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const QuadPtContext& quad_pt_context,
   const Fields& fields)
{
   return CoefficientInputGetter<std::remove_cvref_t<InputTag>>::Get(
      kernel_context,
      weak_form_context,
      operator_context,
      element_context,
      quad_pt_context,
      fields);
}

// -----------------------------------------------------------------------------
// Coefficient traits
// -----------------------------------------------------------------------------

template<typename T>
struct is_coefficient : std::false_type {};

template<StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
struct is_coefficient<Coefficient<Name, Shape, Fn, Inputs...>> : std::true_type {};

template<typename T>
inline constexpr bool is_coefficient_v = is_coefficient<std::remove_cvref_t<T>>::value;

// Shape-specific coefficient traits using partial specializations
template<typename T>
struct is_scalar_coefficient : std::false_type {};

template<StaticString Name, typename Fn, CoefficientInput... Inputs>
struct is_scalar_coefficient<Coefficient<Name, FieldShape::Scalar, Fn, Inputs...>>
   : std::true_type {};

template<typename T>
inline constexpr bool is_scalar_coefficient_v = is_scalar_coefficient<std::remove_cvref_t<T>>::value;

template<typename T>
struct is_vector_coefficient : std::false_type {};

template<StaticString Name, typename Fn, CoefficientInput... Inputs>
struct is_vector_coefficient<Coefficient<Name, FieldShape::Vector, Fn, Inputs...>>
   : std::true_type {};

template<typename T>
inline constexpr bool is_vector_coefficient_v = is_vector_coefficient<std::remove_cvref_t<T>>::value;

template<typename T>
struct is_matrix_coefficient : std::false_type {};

template<StaticString Name, typename Fn, CoefficientInput... Inputs>
struct is_matrix_coefficient<Coefficient<Name, FieldShape::Matrix, Fn, Inputs...>>
   : std::true_type {};

template<typename T>
inline constexpr bool is_matrix_coefficient_v = is_matrix_coefficient<std::remove_cvref_t<T>>::value;

// -----------------------------------------------------------------------------
// Scalar multiplier trait
// -----------------------------------------------------------------------------

/**
 * @brief Trait to identify FieldExprs suitable as scalar multipliers.
 *
 * True for:
 * - Scalar coefficients (FieldShape::Scalar)
 * - ScaleExpr (literal Real scalar)
 *
 * False for:
 * - Vector/matrix coefficients
 * - Normal (geometry vector, not a scalar)
 * - Trial/test spaces (field references, not scalar values)
 * - GradientExpr, JumpExpr, AverageExpr, matrix-vector ProductExpr
 * - MultFieldExpr
 */
template<typename T>
struct is_scalar_multiplier_expr : std::false_type {};

// Scalar coefficients are scalar multipliers
template<StaticString Name, typename Fn, CoefficientInput... Inputs>
struct is_scalar_multiplier_expr<Coefficient<Name, FieldShape::Scalar, Fn, Inputs...>>
   : std::true_type {};

// ScaleExpr is a scalar multiplier (literal Real scalar)
template<>
struct is_scalar_multiplier_expr<ScaleExpr> : std::true_type {};

template<typename T>
inline constexpr bool is_scalar_multiplier_expr_v = is_scalar_multiplier_expr<std::remove_cvref_t<T>>::value;

// -----------------------------------------------------------------------------
// Coefficient leaf
// -----------------------------------------------------------------------------

template <StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
struct Coefficient : FieldBase
{
   static constexpr auto name = Name;
   static constexpr auto field_shape = Shape;
   using fn_type = Fn;

   [[no_unique_address]] Fn fn;

   GENDIL_HOST_DEVICE
   constexpr explicit Coefficient(Fn f)
      : FieldBase{}, fn(std::move(f))
   {}

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   decltype(auto) operator()(
      const KernelContext& kernel_context,
      const WeakFormContext& weak_form_context,
      const OperatorContext& operator_context,
      const ElementContext& element_context,
      const QuadPtContext& quad_pt_context,
      const Fields& fields) const
   {
      static_assert(
         std::is_invocable_v<
            const Fn&,
            decltype(GetCoefficientInput<Inputs>(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               quad_pt_context,
               fields))...>,
         "Coefficient function cannot be called with the requested inputs.");

      return fn(
         GetCoefficientInput<Inputs>(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields)...);
   }
};

template <StaticString Name, CoefficientInput... Inputs, typename Fn>
GENDIL_HOST_DEVICE
constexpr auto MakeCoefficient(Fn&& fn)
{
   using StoredFn = std::remove_cvref_t<Fn>;
   return Coefficient<Name, FieldShape::Scalar, StoredFn, Inputs...>{ std::forward<Fn>(fn) };
}

template <StaticString Name, CoefficientInput... Inputs, typename Fn>
GENDIL_HOST_DEVICE
constexpr auto MakeVectorCoefficient(Fn&& fn)
{
   using StoredFn = std::remove_cvref_t<Fn>;
   return Coefficient<Name, FieldShape::Vector, StoredFn, Inputs...>{ std::forward<Fn>(fn) };
}

template <StaticString Name, CoefficientInput... Inputs, typename Fn>
GENDIL_HOST_DEVICE
constexpr auto MakeMatrixCoefficient(Fn&& fn)
{
   using StoredFn = std::remove_cvref_t<Fn>;
   return Coefficient<Name, FieldShape::Matrix, StoredFn, Inputs...>{ std::forward<Fn>(fn) };
}

// -----------------------------------------------------------------------------
// Printing helpers
// -----------------------------------------------------------------------------

template <typename... Ts, std::size_t... I>
inline void PrintCoefficientInputs(std::ostream& os, std::index_sequence<I...>)
{
   ((os << (I == 0 ? "" : ", ") << Ts{}), ...);
}

template <StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
inline std::ostream& operator<<(std::ostream& os,
                                const Coefficient<Name, Shape, Fn, Inputs...>&)
{
   os << "c_" << Name.view() << "(";
   PrintCoefficientInputs<Inputs...>(os, std::index_sequence_for<Inputs...>{});
   os << ")";
   return os;
}

inline std::ostream& operator<<(std::ostream& os, const ElementIndex&)
{
   return os << "e_index";
}

inline std::ostream& operator<<(std::ostream& os, const QuadraturePointIndex&)
{
   return os << "q_index";
}

inline std::ostream& operator<<(std::ostream& os, const JacobianDeterminant&)
{
   return os << "detJ";
}

inline std::ostream& operator<<(std::ostream& os, const Jacobian&)
{
   return os << "J";
}

inline std::ostream& operator<<(std::ostream& os, const JacobianInverse&)
{
   return os << "invJ";
}

/**
 * @brief Print helper for InverseFacetSize coefficient input
 */
inline std::ostream& operator<<(std::ostream& os, const InverseFacetSize&)
{
   return os << "h_inv";
}

inline std::ostream& operator<<(std::ostream& os, const PhysicalCoordinate&)
{
   return os << "x_phys";
}

inline std::ostream& operator<<(std::ostream& os, const ReferenceCoordinate&)
{
   return os << "x_ref";
}

template <StaticString Name>
inline std::ostream& operator<<(std::ostream& os, const FieldValue<Name>&)
{
   return os << "value(" << Name.view() << ")";
}

template <StaticString Name>
inline std::ostream& operator<<(std::ostream& os, const FieldGradient<Name>&)
{
   return os << "grad(" << Name.view() << ")";
}

} // namespace gendil
