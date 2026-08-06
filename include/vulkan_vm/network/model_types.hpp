#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vvm {
namespace network {

// ============================================================================
// Model file metadata
// ============================================================================

struct ModelFileEntry {
    std::string path;       // relative path within the model directory
    uint64_t size = 0;
    uint8_t sha256[32] = {};
};

// ============================================================================
// Model manifest — the content-addressable "index" for a model version
// ============================================================================

struct ModelManifest {
    std::string modelId;    // e.g. "chonk/llama-3b-q4"
    std::string version;    // e.g. "1" or a git hash
    uint32_t chunkSize = 4u * 1024u * 1024u;  // chunk size (default 4 MiB)
    std::vector<ModelFileEntry> files;
    uint64_t totalSize = 0; // auto-computed, not serialized

    // Serialize to a binary blob (compact protocol-buffer-style).
    std::vector<uint8_t> serialize() const;

    // Deserialize from a buffer; returns false on corruption.
    bool deserialize(const uint8_t* data, size_t len);
};

// ============================================================================
// Chunk request / response helpers
// ============================================================================

struct ChunkRequest {
    std::string model_id;
    std::string version;
    std::string filePath;
    uint32_t chunkIndex = 0;  // 0-based chunk index within the file
};

struct ChunkResponse {
    uint32_t status = 0;  // 0 = success, 1 = not found
    uint8_t sha256[32] = {};
};

// ============================================================================
// Model list entry (returned by MsgModelList)
// ============================================================================

struct ModelIndexEntry {
    std::string model_id;
    std::string version;
    uint64_t totalSize = 0;
    uint32_t fileCount = 0;
    uint64_t timestamp = 0;
};

} // namespace network
} // namespace vvm