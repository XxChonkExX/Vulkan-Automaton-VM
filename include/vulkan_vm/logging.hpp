#pragma once

#ifdef VVM_PLATFORM_LINUX
#include <unistd.h>
#endif

#ifdef VVM_PLATFORM_ANDROID
#include <android/hardware_buffer.h>
#endif

// Export macros for shared library (canonical definition in core.hpp)
#ifndef VVM_API
#if defined(VVM_BUILD_SHARED) && defined(VVM_EXPORT)
#  if defined(_MSC_VER)
#    define VVM_API __declspec(dllexport)
#  else
#    define VVM_API __attribute__((visibility("default")))
#  endif
#elif defined(VVM_BUILD_SHARED) && defined(_MSC_VER)
#  define VVM_API __declspec(dllimport)
#else
#  define VVM_API
#endif
#endif

#include <vector>
#include <string>
#include <optional>
#include <mutex>
#include <chrono>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <sstream>
#include <string_view>
#include <tuple>
#include <cstring>
#include <type_traits>

namespace vvm {

// ============================================================================
// Logging
// ============================================================================

enum class LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4
};

// Default level from VVM_LOG_LEVEL (trace/debug/info/warn[ing]/error;
// unset or unrecognized => Info). Read once per process.
inline LogLevel levelFromEnv() {
    if (const char* e = std::getenv("VVM_LOG_LEVEL")) {
        if (std::strcmp(e, "trace") == 0) return LogLevel::Trace;
        if (std::strcmp(e, "debug") == 0) return LogLevel::Debug;
        if (std::strcmp(e, "info") == 0) return LogLevel::Info;
        if (std::strcmp(e, "warn") == 0 || std::strcmp(e, "warning") == 0)
            return LogLevel::Warning;
        if (std::strcmp(e, "error") == 0) return LogLevel::Error;
    }
    return LogLevel::Info;
}

namespace detail {

// Safe argument to string conversion - no variadic snprintf
template<typename T>
std::string logArgToString(const T& v) {
    if constexpr (std::is_same_v<T, std::string>) {
        return v;
    } else if constexpr (std::is_same_v<T, const char*>) {
        return v ? v : "(null)";
    } else if constexpr (std::is_same_v<T, char*>) {
        return v ? v : "(null)";
    } else if constexpr (std::is_same_v<T, bool>) {
        return v ? "true" : "false";
    } else if constexpr (std::is_same_v<T, char>) {
        return std::string(1, v);
    } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(v);
    } else if constexpr (std::is_floating_point_v<T>) {
        std::ostringstream os;
        os << v;
        return os.str();
    } else if constexpr (std::is_pointer_v<T>) {
        std::ostringstream os;
        os << static_cast<const void*>(v);
        return os.str();
    } else {
        std::ostringstream os;
        os << v;
        return os.str();
    }
}

// Substitute {} placeholders with rendered arguments
// Safe: no variadic snprintf, no format-security issues
template<typename... Args>
std::string logFormat(const char* fmt, Args&&... args) {
    if (fmt == nullptr) return "";
    std::string_view fv(fmt);
    std::vector<std::string> rendered;
    rendered.reserve(sizeof...(Args));
    (rendered.push_back(logArgToString(std::forward<Args>(args))), ...);

    std::string out;
    out.reserve(fv.size() + rendered.size() * 8);
    size_t argIndex = 0;
    for (size_t i = 0; i < fv.size(); ++i) {
        if (i + 1 < fv.size() && fv[i] == '{' && fv[i + 1] == '}') {
            if (argIndex < rendered.size()) {
                out += rendered[argIndex++];
            } else {
                out += "{}";
            }
            ++i;
        } else {
            out += fv[i];
        }
    }
    return out;
}

} // namespace detail

class VVM_API Logger {
public:
    static Logger& instance();
    
    void setLevel(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }
    void setCallback(std::function<void(LogLevel, const std::string&)> cb) { callback_ = std::move(cb); }

    // Type-safe logging with {} placeholder support
    template<typename... Args>
    void log(LogLevel lvl, const char* fmt, Args&&... args) {
        if (lvl < level_) return;

        std::string msg = detail::logFormat(fmt, std::forward<Args>(args)...);

        if (callback_) {
            callback_(lvl, msg);
        } else {
            static const char* levelStr[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
            std::fprintf(stderr, "[%s] %s\n", levelStr[static_cast<int>(lvl)], msg.c_str());
        }
    }

private:
    Logger() = default;
    LogLevel level_ = levelFromEnv();
    std::function<void(LogLevel, const std::string&)> callback_;
};

// Level check first: filtered-out calls neither format arguments nor
// allocate, so hot-path INFO/DEBUG logging is free under a higher level.
#define VVM_LOG_TRACE(...)  do { if (::vvm::Logger::instance().level() <= ::vvm::LogLevel::Trace) ::vvm::Logger::instance().log(::vvm::LogLevel::Trace, __VA_ARGS__); } while (0)
#define VVM_LOG_DEBUG(...)  do { if (::vvm::Logger::instance().level() <= ::vvm::LogLevel::Debug) ::vvm::Logger::instance().log(::vvm::LogLevel::Debug, __VA_ARGS__); } while (0)
#define VVM_LOG_INFO(...)   do { if (::vvm::Logger::instance().level() <= ::vvm::LogLevel::Info) ::vvm::Logger::instance().log(::vvm::LogLevel::Info, __VA_ARGS__); } while (0)
#define VVM_LOG_WARN(...)   do { if (::vvm::Logger::instance().level() <= ::vvm::LogLevel::Warning) ::vvm::Logger::instance().log(::vvm::LogLevel::Warning, __VA_ARGS__); } while (0)
#define VVM_LOG_ERROR(...)  do { if (::vvm::Logger::instance().level() <= ::vvm::LogLevel::Error) ::vvm::Logger::instance().log(::vvm::LogLevel::Error, __VA_ARGS__); } while (0)

} // namespace vvm