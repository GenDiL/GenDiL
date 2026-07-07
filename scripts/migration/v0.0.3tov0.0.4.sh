#!/usr/bin/env bash
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MIGRATION_NAME="GenDiL v0.0.3 to v0.0.4 migration"
MIGRATION_DESCRIPTION="Migrate mechanical include-path changes from GenDiL v0.0.3 to v0.0.4."

. "${SCRIPT_DIR}/common.sh"

migration_parse_args "$@"

migration_replace_literal \
   "gendil/MatrixFreeOperators/KernelOperators/vector.hpp" \
   "gendil/Algebra/vector.hpp"

migration_replace_regex \
   'gendil/MatrixFreeOperators/(?!KernelOperators/vector\.hpp)' \
   "gendil/FiniteElementMethod/MatrixFreeOperators/" \
   "gendil/MatrixFreeOperators/"

migration_finish
