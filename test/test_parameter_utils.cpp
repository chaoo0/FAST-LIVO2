#include "utils/parameter_utils.h"

#include <gtest/gtest.h>

namespace
{

TEST(ResolveMaximumIterations, UsesCanonicalValue)
{
  EXPECT_EQ(fast_livo::resolveMaximumIterations(true, 7, false, 5), 7);
}

TEST(ResolveMaximumIterations, SupportsLegacyOnlyConfiguration)
{
  EXPECT_EQ(fast_livo::resolveMaximumIterations(false, 5, true, 8), 8);
}

TEST(ResolveMaximumIterations, CanonicalValueWinsWhenBothArePresent)
{
  EXPECT_EQ(fast_livo::resolveMaximumIterations(true, 7, true, 8), 7);
}

TEST(ResolveMaximumIterations, RejectsNonPositiveSelection)
{
  EXPECT_THROW(fast_livo::resolveMaximumIterations(true, 0, false, 5), std::invalid_argument);
  EXPECT_THROW(fast_livo::resolveMaximumIterations(false, 5, true, -1), std::invalid_argument);
}

}  // namespace
