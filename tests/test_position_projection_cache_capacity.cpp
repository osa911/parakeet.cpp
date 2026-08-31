#include <gtest/gtest.h>

#include "parakeet/models/encoder.hpp"

#include <cstdlib>
#include <optional>
#include <string>

namespace {

class ScopedEnvironmentVariable {
  public:
    explicit ScopedEnvironmentVariable(const std::optional<std::string> &value) {
        if (const char *current = std::getenv(
                "PARAKEET_POSITION_PROJECTION_CACHE_ENTRIES")) {
            previous_ = current;
        }
        if (value.has_value()) {
            setenv("PARAKEET_POSITION_PROJECTION_CACHE_ENTRIES",
                   value->c_str(), /*overwrite=*/1);
        } else {
            unsetenv("PARAKEET_POSITION_PROJECTION_CACHE_ENTRIES");
        }
    }

    ~ScopedEnvironmentVariable() {
        if (previous_.has_value()) {
            setenv("PARAKEET_POSITION_PROJECTION_CACHE_ENTRIES",
                   previous_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv("PARAKEET_POSITION_PROJECTION_CACHE_ENTRIES");
        }
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(const ScopedEnvironmentVariable &) =
        delete;

  private:
    std::optional<std::string> previous_;
};

TEST(PositionProjectionCacheCapacity,
     DefaultsToNineAndOnlyAcceptsPositiveBoundedOverrides) {
    {
        const ScopedEnvironmentVariable unset(std::nullopt);
        EXPECT_EQ(parakeet::models::detail::position_projection_cache_capacity(),
                  9U);
    }
    {
        const ScopedEnvironmentVariable one(std::string("1"));
        EXPECT_EQ(parakeet::models::detail::position_projection_cache_capacity(),
                  1U);
    }
    {
        const ScopedEnvironmentVariable nine(std::string("9"));
        EXPECT_EQ(parakeet::models::detail::position_projection_cache_capacity(),
                  9U);
    }
    {
        const ScopedEnvironmentVariable zero(std::string("0"));
        EXPECT_EQ(parakeet::models::detail::position_projection_cache_capacity(),
                  9U);
    }
    {
        const ScopedEnvironmentVariable over_limit(std::string("10"));
        EXPECT_EQ(parakeet::models::detail::position_projection_cache_capacity(),
                  9U);
    }
    {
        const ScopedEnvironmentVariable malformed(std::string("nine"));
        EXPECT_EQ(parakeet::models::detail::position_projection_cache_capacity(),
                  9U);
    }
}

} // namespace
