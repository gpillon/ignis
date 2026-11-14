// ignis kernel leaf: the process environment as the operating system holds it -- OURS.
//
// The leaf's measurement switches (IGNIS_GDN_PREFILL, IGNIS_HQ_DECODE_ATTENTION) are read at each
// model load from here, not with std::getenv: on Windows the C runtime's getenv holds a copy taken
// at start, which a later SetEnvironmentVariable (Rust's std::env::set_var, as a test that loads
// under both routes does) does not reach.
#pragma once

#include <string>

namespace ignis {

// `name` in the operating system's environment block, empty when unset.
std::string process_environment(const char *name);

// What a measurement switch of the leaf names at a load: unset (or empty), `ours` or `vendored`.
enum class RouteSwitch { Unset, Ours, Vendored };

// The switch `name` in the operating system's environment (`process_environment`), read at each
// call: the one parse of `ours` and `vendored`. Throws std::invalid_argument, naming the
// variable, on any other value.
RouteSwitch route_switch(const char *name);

} // namespace ignis
