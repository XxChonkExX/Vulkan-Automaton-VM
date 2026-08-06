#pragma once

#include "vulkan_vm/network/model_types.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace vvm {
namespace network {

// ============================================================================
// ModelHub — publish/fetch model weights over TCP
// ============================================================================
//
// Server side:
//   ModelHub hub("0.0.0.0", 51010, "/tmp/model-store");
//   hub.start();
//   hub.publish("chonk/llama-3b", "/path/to/model-files");
//   hub.stop();
//
// Client side (static helpers):
//   auto manifest = ModelHub::fetchManifest("127.0.0.1:51010", "chonk/llama-3b");
//   ModelHub::fetchModel("127.0.0.1:51010", "chonk/llama-3b", manifest,
//                         "./my-models");
//   // or combined:
//   ModelHub::fetch("127.0.0.1:51010", "chonk/llama-3b", "./my-models");
//
// Cache layout (HF-style):
//   <cacheDir>/<model_id>/<version>/
//     weights.safetensors
//     config.json
//     .vvm_complete

class ModelHub {
public:
    // Server construction.  cacheDir is the root for model file storage.
    ModelHub(const std::string& cacheDir);

    // Non-copyable / movable
    ModelHub(const ModelHub&) = delete;
    ModelHub& operator=(const ModelHub&) = delete;
    ModelHub(ModelHub&&) noexcept;
    ModelHub& operator=(ModelHub&&) noexcept;
    ~ModelHub();

    // Start listening for model requests on the given address + port.
    bool start(const std::string& listenAddr, uint16_t port);

    // Stop the hub.
    void stop();

    bool isRunning() const { return running_; }
    uint16_t getBoundPort() const { return boundPort_; }
    const std::string& getCacheDir() const { return cacheDir_; }

    // Register a model from a local directory (chunk + hash + import into
    // the hub's cache).  Returns true if at least one file was imported.
    bool publish(const std::string& modelId, const std::string& sourceDir,
                 const std::string& version = "1");

    // List models known to this hub.
    std::vector<ModelIndexEntry> listModels() const;

    // Get the manifest for a published model.
    std::optional<ModelManifest> getManifest(const std::string& modelId,
                                             const std::string& version = "1") const;

    // ========================================================================
    // Client-side static helpers (connect to a remote hub)
    // ========================================================================

    struct DownloadProgress {
        uint64_t bytesDone = 0;
        uint64_t totalBytes = 0;
        uint32_t chunksDone = 0;
        uint32_t totalChunks = 0;
        bool done = false;
        std::string error;
    };

    using ProgressFn = std::function<void(const DownloadProgress&)>;

    // Fetch a manifest from a remote hub.
    static std::optional<ModelManifest> fetchManifest(
        const std::string& connection,   // "host:port"
        const std::string& modelId,
        const std::string& version = "1",
        int32_t timeoutMs = 60000);

    // Fetch all files for a model from a remote hub into a destination
    // directory.  Resumes: skips chunks already present (by hash).
    // Returns total bytes downloaded (0 on complete failure).
    static uint64_t fetchModel(
        const std::string& connection,
        const std::string& modelId,
        const ModelManifest& manifest,
        const std::string& destDir,
        ProgressFn progress = nullptr,
        int32_t timeoutMs = 60000);

    // Combined fetch: manifest + all files.
    static bool fetch(const std::string& connection,
                      const std::string& modelId,
                      const std::string& destDir,
                      const std::string& version = "1",
                      ProgressFn progress = nullptr,
                      int32_t timeoutMs = 60000);

private:
    // ========================================================================
    // Server internals
    // ========================================================================

    void onRequest(TcpMessage& req, TcpMessage& resp);

    std::string modelPath(const std::string& modelId, const std::string& version) const;
    std::string completeMarker(const std::string& modelId, const std::string& version) const;

    bool readChunkFromDisk(const std::string& filePath, uint32_t chunkIndex,
                           uint32_t chunkSize, std::vector<uint8_t>& outChunk) const;

    std::string cacheDir_;
    std::unique_ptr<TcpTransport> transport_;
    uint16_t boundPort_ = 0;
    bool running_ = false;

    // Published model manifests (in memory, survives until stop).
    mutable std::mutex manifestMutex_;
    // key = "model_id:version"
    std::unordered_map<std::string, ModelManifest> manifests_;
};

} // namespace network
} // namespace vvm