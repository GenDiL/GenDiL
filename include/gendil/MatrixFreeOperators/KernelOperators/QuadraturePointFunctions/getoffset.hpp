// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

template < typename MatrixTuple, size_t I >
struct GetOffset;

template < >
struct GetOffset< std::tuple< >, 0 >
{
    static constexpr size_t value = 0;
};

template < typename... RestMatrices >
struct GetOffset< std::tuple< Real, RestMatrices... >, 0 >
{
    static constexpr size_t value = 0;
};

template < typename... RestMatrices, size_t I >
struct GetOffset< std::tuple< Real, RestMatrices... >, I >
{
    static constexpr size_t value = 1 + GetOffset< std::tuple< RestMatrices... >, I-1 >::value;
};

template < size_t N, typename... RestMatrices >
struct GetOffset< std::tuple< Real[N][N], RestMatrices... >, 0 >
{
    static constexpr size_t value = 0;
};

template < size_t N, typename... RestMatrices, size_t I >
struct GetOffset< std::tuple< Real[N][N], RestMatrices... >, I >
{
    static constexpr size_t value = N + GetOffset< std::tuple< RestMatrices... >, I-1 >::value;
};

template < size_t N, typename... RestMatrices >
struct GetOffset< std::tuple< std::array< std::array< Real, N >, N >, RestMatrices... >, 0 >
{
    static constexpr size_t value = 0;
};

template < size_t N, typename... RestMatrices, size_t I >
struct GetOffset< std::tuple< std::array< std::array< Real, N >, N >, RestMatrices... >, I >
{
    static constexpr size_t value = N + GetOffset< std::tuple< RestMatrices... >, I-1 >::value;
};

template < typename... SubMatrices, typename... RestMatrices >
struct GetOffset< std::tuple< std::tuple< SubMatrices... >, RestMatrices... >, 0 >
{
    static constexpr size_t value = 0;
};

template < typename... SubMatrices, typename... RestMatrices, size_t I >
struct GetOffset< std::tuple< std::tuple< SubMatrices... >, RestMatrices... >, I >
{
    static constexpr size_t value = GetOffset< std::tuple< SubMatrices... >, sizeof...( SubMatrices ) >::value + GetOffset< std::tuple< RestMatrices... >, I-1 >::value;
};

}
