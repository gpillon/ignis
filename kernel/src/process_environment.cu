// ignis kernel leaf: the process environment as the operating system holds it -- OURS
// (process_environment.h).
#include "process_environment.h"

#include <cstdlib>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ignis {

std::string process_environment(const char *name) {
#ifdef _WIN32
  std::string value(64, '\0');
  DWORD n = GetEnvironmentVariableA(name, value.data(), static_cast<DWORD>(value.size()));
  if (n >= value.size()) {
    value.assign(n, '\0');
    n = GetEnvironmentVariableA(name, value.data(), static_cast<DWORD>(value.size()));
  }
  value.resize(n < value.size() ? n : 0);
  return value;
#else
  const char *v = std::getenv(name);
  return v == nullptr ? std::string() : std::string(v);
#endif
}

} // namespace ignis
