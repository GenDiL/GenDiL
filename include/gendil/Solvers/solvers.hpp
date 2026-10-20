
// A set of possible implementations of a conjugate gradient solver targetting different situations.
#include "gendil/Solvers/conjugategradient.hpp"

// Restarted GMRES and flexible GMRES with reusable or caller-provided scratch.
#include "gendil/Solvers/GMRES/gmres.hpp"

// Ordinary BiCGSTAB and explicitly bounded restarting.
#include "gendil/Solvers/bicgstab.hpp"

// Repeat a bound solve with an explicit recovery decision.
#include "gendil/Solvers/restartedsolver.hpp"
