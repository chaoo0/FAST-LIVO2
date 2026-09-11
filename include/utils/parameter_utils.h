#ifndef PARAMETER_UTILS_H
#define PARAMETER_UTILS_H

#include <stdexcept>

namespace fast_livo
{

inline int resolveMaximumIterations(
  const bool canonical_configured,
  const int canonical_value,
  const bool legacy_min_configured,
  const int legacy_min_value)
{
  const int selected = (!canonical_configured && legacy_min_configured) ?
    legacy_min_value : canonical_value;
  if (selected <= 0)
  {
    throw std::invalid_argument("lio.max_iterations must be greater than zero");
  }
  return selected;
}

}  // namespace fast_livo

#endif  // PARAMETER_UTILS_H
