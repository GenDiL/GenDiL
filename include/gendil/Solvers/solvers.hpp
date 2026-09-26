
// A set of possible implementations of a conjugate gradient solver targetting different situations.
#include "gendil/Solvers/conjugategradient.hpp"

// Restarted GMRES and flexible GMRES with reusable or caller-provided scratch.
#include "gendil/Solvers/GMRES/gmres.hpp"

// A simple BiCGSTAB implementation
#include "gendil/Solvers/bicgstab.hpp"
