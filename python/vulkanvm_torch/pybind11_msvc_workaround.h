// pybind11_msvc_workaround.h
// Workaround for MSVC 19.44 bug with pybind11's ssize_t in nested namespaces.
// Must be included BEFORE any pybind11 headers.

#pragma once

// On Windows, MSVC doesn't define ssize_t (POSIX type).
// Python defines Py_ssize_t. pybind11 uses `using ssize_t = Py_ssize_t;`
// inside its namespace, but MSVC 19.44 fails to find it in nested namespaces.

#include <Python.h>  // for Py_ssize_t

// Define ssize_t in global namespace so unqualified lookup works
#ifndef ssize_t
using ssize_t = Py_ssize_t;
#endif

// Also define it in pybind11::detail namespace explicitly
// This needs to be included AFTER pybind11's common.h but we can't control that.
// Instead, we can use a macro to patch it.

#endif // PYBIND11_MSVC_WORKAROUND_H