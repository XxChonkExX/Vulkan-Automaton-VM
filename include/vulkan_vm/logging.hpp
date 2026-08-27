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

namespace detail {

template<typename... Args>
std::string logFormat(const char* fmt, Args&&... args) {
    size_t len = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
    if (len <= 0) return "";
    std::string result(len + 1, '\0');
    std::snprintf(&result[0], result.size(), fmt, std::forward<Args>(args)...);
    result.pop_back();
    return result;
}

} // namespace detail

class VVM_API Logger {
public:
    static Logger& instance();
    
    void setLevel(LogLevel level) { level_ = level; }
    void setCallback(std::function<void(LogLevel, const std::string&)> cb) { callback_ = std::move(cb); }

    // Supports both printf-style (%d, %s, ...) and {}-style formats.
    template<typename... Args>
    void log(LogLevel lvl, const char* fmt, Args&&... args) {
        if (lvl < level_) return;

        std::string msg;
        if (fmt != nullptr && std::strstr(fmt, "{}") != nullptr) {
            msg = detail::logFormat(fmt, std::forward<Args>(args)...);
        } else {
            char buffer[1024];
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wnon-pod-varargs"
#endif
            int len = std::snprintf(buffer, sizeof(buffer), fmt ? fmt : "", std::forward<Args>(args)...);
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            if (len < 0) return;
            msg.assign(buffer, static_cast<size_t>(len));
        }

        if (callback_) {
            callback_(lvl, msg);
        } else {
            static const char* levelStr[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
            std::fprintf(stderr, "[%s] %s\n", levelStr[static_cast<int>(lvl)], msg.c_str());
        }
    }

private:
    Logger() = default;
    LogLevel level_ = LogLevel::Info;
    std::function<void(LogLevel, const std::string&)> callback_;
};

#define VVM_LOG_TRACE(...)  ::vvm::Logger::instance().log(::vvm::LogLevel::Trace, __VA_ARGS__)
#define VVM_LOG_DEBUG(...)  ::vvm::Logger::instance().log(::vvm::LogLevel::Debug, __VA_ARGS__)
#define VVM_LOG_INFO(...)   ::vvm::Logger::instance().log(::vvm::LogLevel::Info, __VA_ARGS__)
#define VVM_LOG_WARN(...)   ::vvm::Logger::instance().log(::vvm::LogLevel::Warning, __VA_ARGS__)
#define VVM_LOG_ERROR(...)  ::vvm::Logger::instance().log(::vvm::LogLevel::Error, __VA_ARGS__)

} // namespace vvm