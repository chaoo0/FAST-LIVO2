#include "utils/timestamp_utils.h"

#include <gtest/gtest.h>

#include <limits>

namespace
{

using fast_livo::TimestampOrder;

TEST(ClassifyTimestamp, AcceptsFirstAndInOrderSamples)
{
  EXPECT_EQ(fast_livo::classifyTimestamp(-1.0, 100.0, 0.2), TimestampOrder::kFirst);
  EXPECT_EQ(fast_livo::classifyTimestamp(100.0, 100.2, 0.2), TimestampOrder::kInOrder);
}

TEST(ClassifyTimestamp, DistinguishesForwardGapFromBackwardTime)
{
  EXPECT_EQ(fast_livo::classifyTimestamp(100.0, 100.200001, 0.2), TimestampOrder::kForwardGap);
  EXPECT_EQ(fast_livo::classifyTimestamp(100.0, 99.999999, 0.2), TimestampOrder::kBackward);
}

TEST(ClassifyTimestamp, RejectsNonFiniteOrInvalidInputs)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  EXPECT_EQ(fast_livo::classifyTimestamp(100.0, nan, 0.2), TimestampOrder::kInvalid);
  EXPECT_EQ(fast_livo::classifyTimestamp(100.0, infinity, 0.2), TimestampOrder::kInvalid);
  EXPECT_EQ(fast_livo::classifyTimestamp(nan, 101.0, 0.2), TimestampOrder::kInvalid);
  EXPECT_EQ(fast_livo::classifyTimestamp(-infinity, 101.0, 0.2), TimestampOrder::kInvalid);
  EXPECT_EQ(fast_livo::classifyTimestamp(100.0, 101.0, -0.1), TimestampOrder::kInvalid);
}

}  // namespace
