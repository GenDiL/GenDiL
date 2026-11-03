// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/accessors.hpp"
#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"

namespace gendil
{

template<FieldExpr Expr>
struct TransposeExpr : FieldBase
{
   Expr expr;

   static constexpr FieldShape operand_shape =
      field_shape_v<std::remove_cvref_t<Expr>>;

   GENDIL_HOST_DEVICE
   constexpr explicit TransposeExpr(const Expr& expr_)
      : expr(expr_)
   {}

   template<typename... Args>
   GENDIL_HOST_DEVICE
   auto operator()(Args&&... args) const
   {
      const auto value = expr(std::forward<Args>(args)...);
      using ValueType = std::remove_cvref_t<decltype(value)>;

      if constexpr (operand_shape == FieldShape::Scalar)
      {
         return value;
      }
      else if constexpr (operand_shape == FieldShape::Vector)
      {
         constexpr Integer size = static_extent_v<ValueType>;
         static_assert(size != std::dynamic_extent,
            "transpose(vector) requires a statically sized vector value.");

         using EntryType = std::conditional_t<StaticVector<ValueType>,
            vector_value_t<ValueType>,
            typename vector_element_type<
               std::remove_cvref_t<decltype(GetValue(value, 0))>>::type>;

         SerialRecursiveArray<EntryType, 1, size> result;
         for (Integer i = 0; i < size; ++i)
         {
            result(0, i) = static_cast<EntryType>(GetValue(value, i));
         }
         return result;
      }
      else
      {
         constexpr Integer rows = static_num_rows_v<ValueType>;
         constexpr Integer cols = static_num_cols_v<ValueType>;
         static_assert(rows != std::dynamic_extent &&
                       cols != std::dynamic_extent,
            "transpose(matrix) requires a statically sized matrix value.");

         // Readable proxies describe their scalar through matrix_value_t;
         // the owning transpose stores those scalars, not accessor proxies.
         using EntryType = std::conditional_t<StaticMatrix<ValueType>,
            matrix_value_t<ValueType>,
            typename vector_element_type<
               std::remove_cvref_t<decltype(GetValue(value, 0, 0))>>::type>;

         SerialRecursiveArray<EntryType, cols, rows> result;
         for (Integer i = 0; i < rows; ++i)
         {
            for (Integer j = 0; j < cols; ++j)
            {
               result(j, i) = static_cast<EntryType>(GetValue(value, i, j));
            }
         }
         return result;
      }
   }
};

template<FieldExpr Expr>
struct field_shape_impl<TransposeExpr<Expr>>
{
   static constexpr FieldShape operand_shape =
      field_shape_v<std::remove_cvref_t<Expr>>;

   static constexpr FieldShape value =
      operand_shape == FieldShape::Scalar
         ? FieldShape::Scalar
         : FieldShape::Matrix;
};

template<FieldExpr Expr>
std::ostream& operator<<(std::ostream& os, const TransposeExpr<Expr>& expr)
{
   return os << "transpose(" << expr.expr << ")";
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
constexpr auto transpose(const Expr& expr)
{
   return TransposeExpr<std::remove_cvref_t<Expr>>{expr};
}

} // namespace gendil
