#ifndef TIMESTAMP_UTILS_H
#define TIMESTAMP_UTILS_H

#include <cmath>

namespace fast_livo
{

enum class TimestampOrder
{
  kFirst,
  kInOrder,
  kForwardGap,
  kBackward,
  kInvalid,
};

inline TimestampOrder classifyTimestamp(
  const double last_timestamp,
  const double current_timestamp,
  const double expected_max_gap)
{
  if (!std::isfinite(last_timestamp) || !std::isfinite(current_timestamp) ||
    !std::isfinite(expected_max_gap) || expected_max_gap < 0.0)
  {
    return TimestampOrder::kInvalid;
  }
  if (last_timestamp < 0.0) return TimestampOrder::kFirst;
  if (current_timestamp < last_timestamp) return TimestampOrder::kBackward;
  if (current_timestamp > last_timestamp + expected_max_gap) return TimestampOrder::kForwardGap;
  return TimestampOrder::kInOrder;
}

}  // namespace fast_livo

#endif  // TIMESTAMP_UTILS_H
