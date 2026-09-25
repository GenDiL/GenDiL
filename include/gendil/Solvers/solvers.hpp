
// A set of possible implementations of a conjugate gradient solver targetting different situations.
#include "gendil/Solvers/conjugategradient.hpp"

// A diagnostic version of conjugate gradient with detailed iteration tracking
#include "gendil/Solvers/conjugate_gradient_diagnostic.hpp"

// A simple GMRES implementation
#include "gendil/Solvers/gmres.hpp"

// A simple BiCGSTAB implementation
#include "gendil/Solvers/bicgstab.hpp"
