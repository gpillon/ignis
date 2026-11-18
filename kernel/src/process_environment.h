// ignis kernel leaf: the process environment as the operating system holds it -- OURS.
//
// The leaf's measurement switches (IGNIS_GDN_PREFILL, IGNIS_HQ_DECODE_ATTENTION) are read at each
// model load from here, not with std::getenv: on Windows the C runtime's getenv holds a copy taken
// at start, which a later SetEnvironmentVariable (Rust's std::env::set_var, as a test that loads
// under both routes does) does not reach.
#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace ignis {

// `name` in the operating system's environment block, empty when unset.
std::string process_environment(const char *name);

// What the measurement switch `name` names in the operating system's environment
// (`process_environment`), read at each call: the one parse of every switch of the leaf. `unset`
// when it is unset or empty, else the value paired with the spelling it equals. Throws
// std::invalid_argument, naming the variable and the spellings, on any other value.
template <class Value>
Value named_switch(const char *name, Value unset,
                   std::initializer_list<std::pair<const char *, Value>> spellings) {
  const std::string named = process_environment(name);
  if (named.empty()) return unset;
  for (const auto &spelling : spellings) {
    if (named == spelling.first) return spelling.second;
  }
  std::string expected;
  std::size_t i = 0;
  for (const auto &spelling : spellings) {
    expected += i == 0 ? "" : i + 1 == spellings.size() ? " or " : ", ";
    expected += std::string("`") + spelling.first + "`";
    ++i;
  }
  throw std::invalid_argument(std::string(name) + " is " + expected + ", not `" + named + "`");
}

// What a measurement switch of the leaf names at a load: unset (or empty), `ours` or `vendored`.
enum class RouteSwitch { Unset, Ours, Vendored };

// The switch `name` (`named_switch`): `ours` or `vendored`. Throws std::invalid_argument, naming
// the variable, on any other value.
RouteSwitch route_switch(const char *name);

} // namespace ignis
