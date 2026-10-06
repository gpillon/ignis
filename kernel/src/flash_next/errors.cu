// ignis kernel leaf -- the Flash-Next program's error reporting (spec
// flash-next/04, GitHub #302; OURS). Every entry point of
// flash_next_internal.h reports a failure through fn_set_error; the program's
// C ABI reads it back with fn_last_error.

#include "flash_next_internal.h"

#include <string>
#include <utility>

namespace ignis::flash_next {

namespace {

// The last message on this thread, like every other leaf file's.
thread_local std::string g_last_error;

}  // namespace

void fn_set_error(std::string message) {
  g_last_error = std::move(message);
}

const char *fn_last_error() {
  return g_last_error.c_str();
}

}  // namespace ignis::flash_next
