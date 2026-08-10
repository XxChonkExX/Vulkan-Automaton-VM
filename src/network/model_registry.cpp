#include "vulkan_vm/network/model_registry.hpp"
#include "vulkan_vm/network/sha256.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/network/wire_format.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <filesystem>
#include <iostream>
#include <sstream>

namespace vvm {
namespace network {
namespace fs = std::filesystem;

using namespace std::chrono_literals;

// ============================================================================
// Helpers: binary put/get (canonical implementation in wire_format.hpp)
// ============================================================================

namespace detail = ::vvm::network::wire;

// ============================================================================
// ModelManifest serialization
// ============================================================================

std::vector<uint8_t> ModelManifest::serialize() const {
    std::vector<uint8_t> out;
    detail::putStr(out, modelId);
    detail::putStr(out, version);
    detail::putU32(out, chunkSize);
    detail::putU32(out, static_cast<uint32_t>(files.size()));
    for (const auto& f : files) {
        detail::putStr(out, f.path);
        detail::putU64(out, f.size);
        out.insert(out.end(), f.sha256, f.sha256 + 32);
    }
    return out;
}

bool ModelManifest::deserialize(const uint8_t* data, size_t len) {
    const uint8_t* p = data;
    const uint8_t* end = data + len;
    if (!detail::getStr(p, end, modelId)) return false;
    if (!detail::getStr(p, end, version)) return false;
    if (!detail::getU32(p, end, chunkSize)) return false;
    uint32_t fileCount = 0;
    if (!detail::getU32(p, end, fileCount)) return false;
    files.clear();
    totalSize = 0;
    for (uint32_t i = 0; i < fileCount; ++i) {
        ModelFileEntry f;
        if (!detail::getStr(p, end, f.path)) return false;
        if (!detail::getU64(p, end, f.size)) return false;
        if (p + 32 > end) return false;
        std::copy(p, p + 32, f.sha256);
        p += 32;
        files.push_back(std::move(f));
        totalSize += f.size;
    }
    return true;
}

// ============================================================================
// ModelHub Implementation
// ============================================================================

ModelHub::ModelHub(const std::string& cacheDir) : cacheDir_(cacheDir) {
    fs::create_directories(cacheDir_);
}

ModelHub::ModelHub(ModelHub&& other) noexcept
    : cacheDir_(std::move(other.cacheDir_))
    , transport_(std::move(other.transport_))
    , boundPort_(other.boundPort_)
    , running_(other.running_)
    , manifests_(std::move(other.manifests_)) {
    other.running_ = false;
    other.boundPort_ = 0;
}

ModelHub& ModelHub::operator=(ModelHub&& other) noexcept {
    if (this != &other) {
        stop();
        cacheDir_ = std::move(other.cacheDir_);
        transport_ = std::move(other.transport_);
        boundPort_ = other.boundPort_;
        running_ = other.running_;
        manifests_ = std::move(other.manifests_);
        other.running_ = false;
        other.boundPort_ = 0;
    }
    return *this;
}

ModelHub::~ModelHub() { stop(); }

bool ModelHub::start(const std::string& listenAddr, uint16_t port) {
    if (running_) return false;
    transport_ = std::make_unique<TcpTransport>();
    auto handler = [this](TcpMessage& req, TcpMessage& resp) { onRequest(req, resp); };
    // NetworkConfig with defaults for ModelHub
    NetworkConfig netConfig;
    netConfig.listenAddress = listenAddr + ":" + std::to_string(port);
    if (!transport_->start(listenAddr, port, std::move(handler), netConfig)) return false;
    boundPort_ = transport_->getBoundPort();
    running_ = true;
    VVM_LOG_INFO("ModelHub listening on {}:{} (cache: {})", listenAddr, boundPort_, cacheDir_);
    return true;
}

void ModelHub::stop() {
    if (!running_) return;
    running_ = false;
    if (transport_) transport_->stop();
    transport_.reset();
    VVM_LOG_INFO("ModelHub stopped");
}

std::string ModelHub::modelPath(const std::string& modelId, const std::string& version) const {
    return (fs::path(cacheDir_) / modelId / version).string();
}

std::string ModelHub::completeMarker(const std::string& modelId, const std::string& version) const {
    return (fs::path(modelPath(modelId, version)) / ".vvm_complete").string();
}

bool ModelHub::publish(const std::string& modelId, const std::string& sourceDir,
                       const std::string& version) {
    fs::path src = sourceDir;
    if (!fs::exists(src) || !fs::is_directory(src)) {
        VVM_LOG_ERROR("publish: source dir does not exist: {}", sourceDir);
        return false;
    }

    std::string destDir = modelPath(modelId, version);
    fs::create_directories(destDir);

    ModelManifest manifest;
    manifest.modelId = modelId;
    manifest.version = version;
    manifest.chunkSize = 4u * 1024u * 1024u;

    bool any = false;
    for (const auto& entry : fs::recursive_directory_iterator(src)) {
        if (!entry.is_regular_file()) continue;
        fs::path rel = fs::relative(entry.path(), src);
        std::string relPath = rel.generic_string();
        std::replace(relPath.begin(), relPath.end(), '\\', '/');

        // Hash the whole file
        Sha256 h;
        std::ifstream in(entry.path(), std::ios::binary);
        if (!in) {
            VVM_LOG_WARN("publish: failed to open {}", entry.path().string());
            continue;
        }
        constexpr size_t kBuf = 1u << 20;  // 1 MiB
        std::vector<uint8_t> buf(1u << 20);
        while (in) {
            in.read(reinterpret_cast<char*>(buf.data()), buf.size());
            h.update(buf.data(), static_cast<size_t>(in.gcount()));
        }
        uint8_t digest[32];
        h.finalize(digest);

        // Copy to cache
        fs::path dst = fs::path(destDir) / relPath;
        fs::create_directories(dst.parent_path());
        fs::copy_file(entry.path(), dst, fs::copy_options::overwrite_existing);

        ModelFileEntry f;
        f.path = relPath;
        f.size = fs::file_size(entry);
        std::copy(digest, digest + 32, f.sha256);
        manifest.files.push_back(std::move(f));
        manifest.totalSize += f.size;
        any = true;

        VVM_LOG_INFO("publish: {} ({} bytes, sha256={})",
                     relPath, f.size, Sha256::hex(digest));
    }

    if (!any) {
        VVM_LOG_WARN("publish: no regular files found in {}", sourceDir);
        return false;
    }

    // Persist manifest and register
    {
        std::lock_guard<std::mutex> lock(manifestMutex_);
        manifests_[modelId + ":" + version] = manifest;
    }

    // Write .vvm_complete marker
    std::ofstream marker(completeMarker(modelId, version));
    if (marker) marker << "complete\n";
    marker.close();

    VVM_LOG_INFO("publish: model {}@{} registered ({} files, {} MB total)",
                 modelId, version, manifest.files.size(), manifest.totalSize / (1024 * 1024));
    return true;
}

std::vector<ModelIndexEntry> ModelHub::listModels() const {
    std::vector<ModelIndexEntry> out;
    std::lock_guard<std::mutex> lock(manifestMutex_);
    out.reserve(manifests_.size());
    for (const auto& [k, m] : manifests_) {
        ModelIndexEntry e;
        e.model_id = m.modelId;
        e.version = m.version;
        e.totalSize = m.totalSize;
        e.fileCount = static_cast<uint32_t>(m.files.size());
        out.push_back(std::move(e));
    }
    return out;
}

std::optional<ModelManifest> ModelHub::getManifest(const std::string& modelId,
                                                   const std::string& version) const {
    std::lock_guard<std::mutex> lock(manifestMutex_);
    auto it = manifests_.find(modelId + ":" + version);
    if (it != manifests_.end()) return it->second;
    return std::nullopt;
}

void ModelHub::onRequest(TcpMessage& req, TcpMessage& resp) {
    resp.type = req.type;
    resp.flags = TcpFlagsResponse;
    resp.seq = req.seq;

    if (req.type == MsgModelList) {
        auto models = listModels();
        // Serialize ModelIndexEntry list directly
        std::vector<uint8_t> out;
        detail::putU32(out, static_cast<uint32_t>(models.size()));
        for (const auto& m : models) {
            detail::putStr(out, m.model_id);
            detail::putStr(out, m.version);
            detail::putU64(out, m.totalSize);
            detail::putU32(out, m.fileCount);
            detail::putU64(out, m.timestamp);
        }
        resp.body = std::move(out);
        return;
    }

    if (req.type == MsgModelManifest) {
        // body: model_id \0 version
        const uint8_t* p = req.body.data();
        const uint8_t* end = p + req.body.size();
        std::string modelId, version = "1";
        if (p == end) { resp.flags = TcpFlagsError; return; }
        const uint8_t* term = std::find(p, end, 0);
        if (term == end) { resp.flags = TcpFlagsError; return; }  // unterminated
        modelId.assign(reinterpret_cast<const char*>(p), static_cast<size_t>(term - p));
        p = term + 1;
        if (p < end) {
            term = std::find(p, end, 0);
            version.assign(reinterpret_cast<const char*>(p), static_cast<size_t>(term - p));
        }
        auto man = getManifest(modelId, version);
        if (!man) {
            resp.flags = TcpFlagsError;
            return;
        }
        resp.body = man->serialize();
        return;
    }

    if (req.type == MsgModelChunk) {
        // body: model_id \0 version \0 filePath \0 chunkIndex (u32)
        const uint8_t* p = req.body.data();
        const uint8_t* end = p + req.body.size();
        auto readCStr = [&](std::string& s) -> bool {
            if (p >= end) return false;
            const uint8_t* term = std::find(p, end, 0);
            if (term == end) return false;  // unterminated: refuse (no overread)
            s.assign(reinterpret_cast<const char*>(p), static_cast<size_t>(term - p));
            p = term + 1;
            return true;
        };
        std::string modelId, version, filePath;
        if (!readCStr(modelId) || !readCStr(version) || !readCStr(filePath)) {
            resp.flags = TcpFlagsError;
            return;
        }
        if (p + 4 > end) { resp.flags = TcpFlagsError; return; }
        uint32_t chunkIdx = (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
        p += 4;

        auto manOpt = getManifest(modelId, version);
        if (!manOpt) {
            resp.flags = TcpFlagsError;
            return;
        }
        const auto& man = *manOpt;
        const auto fIt = std::find_if(man.files.begin(), man.files.end(),
            [&](const ModelFileEntry& f) { return f.path == filePath; });
        if (fIt == man.files.end()) {
            resp.flags = TcpFlagsError;
            return;
        }
        if (man.chunkSize == 0) {
            resp.flags = TcpFlagsError;
            return;
        }

        // Defense in depth: even though filePath must match a published
        // manifest entry, refuse anything that would escape the model root.
        fs::path fileRel = fs::path(filePath).lexically_normal();
        const std::string fileRelStr = fileRel.string();
        const bool escapes = fileRel.is_absolute() ||
                             fileRelStr == ".." ||
                             fileRelStr.rfind("..\\", 0) == 0 ||
                             fileRelStr.rfind("../", 0) == 0;
        if (escapes) {
            resp.flags = TcpFlagsError;
            return;
        }

        uint64_t fileSize = fIt->size;
        uint32_t chunkSize = man.chunkSize;
        uint64_t offset = static_cast<uint64_t>(chunkIdx) * chunkSize;
        if (offset >= fileSize) {
            resp.flags = TcpFlagsError;
            return;
        }
        uint64_t toRead = std::min<uint64_t>(chunkSize, fileSize - offset);

        std::string fullPath = (fs::path(modelPath(modelId, version)) / fileRel).string();
        std::vector<uint8_t> chunk;
        chunk.resize(static_cast<size_t>(toRead));
        if (!readChunkFromDisk(fullPath, chunkIdx, chunkSize, chunk)) {
            resp.flags = TcpFlagsError;
            return;
        }

        // Compute chunk hash for response
        Sha256 h;
        h.update(chunk.data(), chunk.size());
        uint8_t digest[32];
        h.finalize(digest);

        // Response body: status (u32=0) + chunkSha256[32]
        resp.body.resize(4 + 32);
        resp.body[0] = resp.body[1] = resp.body[2] = 0;
        resp.body[3] = 0;  // status = 0
        std::copy(digest, digest + 32, resp.body.begin() + 4);

        // Stream the chunk data
        resp.streamSource = chunk.data();
        resp.streamLen = chunk.size();
        resp.streamCleanup = [buf = std::move(chunk)]() mutable {};
        return;
    }

    resp.flags = TcpFlagsError;
}

bool ModelHub::readChunkFromDisk(const std::string& filePath, uint32_t chunkIdx,
                                 uint32_t chunkSize, std::vector<uint8_t>& out) const {
    std::ifstream in(filePath, std::ios::binary);
    if (!in) return false;
    uint64_t offset = static_cast<uint64_t>(chunkIdx) * chunkSize;
    in.seekg(static_cast<std::streamoff>(offset));
    if (!in) return false;
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    return in.good() || in.eof();
}

// ============================================================================
// Client-side static helpers
// ============================================================================

std::optional<ModelManifest> ModelHub::fetchManifest(
    const std::string& connection, const std::string& modelId,
    const std::string& version, int32_t timeoutMs) {
    size_t colon = connection.rfind(':');
    if (colon == std::string::npos) return std::nullopt;
    std::string host = connection.substr(0, colon);
    uint16_t port = 0;
    try {
        port = static_cast<uint16_t>(std::stoi(connection.substr(colon + 1)));
    } catch (const std::exception&) {
        return std::nullopt;
    }

    TcpTransport transport;
    auto conn = transport.connect(host, port, timeoutMs);
    if (!conn) return std::nullopt;

    TcpMessage req;
    req.type = MsgModelManifest;
    req.flags = TcpFlagsRequest;
    req.body.insert(req.body.end(), modelId.begin(), modelId.end());
    req.body.push_back(0);
    req.body.insert(req.body.end(), version.begin(), version.end());
    req.body.push_back(0);

    auto resp = transport.request(*conn, req);
    if (!resp || resp->flags == TcpFlagsError) return std::nullopt;

    ModelManifest m;
    if (!m.deserialize(resp->body.data(), resp->body.size())) return std::nullopt;
    return m;
}

static size_t countFileChunks(uint64_t fileSize, uint32_t chunkSize) {
    return (fileSize + chunkSize - 1) / chunkSize;
}

uint64_t ModelHub::fetchModel(
    const std::string& connection,
    const std::string& modelId,
    const ModelManifest& manifest,
    const std::string& destDir,
    ProgressFn progress,
    int32_t timeoutMs) {
    if (manifest.chunkSize == 0) {
        VVM_LOG_ERROR("fetchModel: manifest has chunkSize 0");
        return 0;
    }
    size_t colon = connection.rfind(':');
    if (colon == std::string::npos) return 0;
    uint16_t port = 0;
    try {
        port = static_cast<uint16_t>(std::stoi(connection.substr(colon + 1)));
    } catch (const std::exception&) {
        return 0;
    }
    std::string host = connection.substr(0, colon);

    fs::create_directories(destDir);

    TcpTransport transport;
    auto conn = transport.connect(host, port, timeoutMs);
    if (!conn) return 0;

    uint64_t totalDownloaded = 0;
    uint64_t totalChunks = 0;
    for (const auto& f : manifest.files) totalChunks += countFileChunks(f.size, manifest.chunkSize);
    uint32_t chunksDone = 0;

    DownloadProgress dl;
    dl.totalBytes = manifest.totalSize;
    dl.totalChunks = totalChunks;

    for (const auto& file : manifest.files) {
        fs::path outPath = fs::path(destDir) / file.path;
        fs::create_directories(outPath.parent_path());

        const uint64_t fileChunks = countFileChunks(file.size, manifest.chunkSize);

        // Resume granularity is the whole file: a partial file cannot be
        // validated chunk-by-chunk against the manifest (which only carries
        // the whole-file sha256), so only an exact whole-file hash match lets
        // us skip re-downloading.
        bool fileComplete = false;
        if (fs::exists(outPath)) {
            std::ifstream check(outPath, std::ios::binary);
            if (check) {
                std::vector<uint8_t> whole(file.size > 0
                    ? static_cast<size_t>(file.size) : 1);
                check.read(reinterpret_cast<char*>(whole.data()),
                           static_cast<std::streamsize>(whole.size()));
                if (check.gcount() == static_cast<std::streamsize>(whole.size())) {
                    Sha256 h;
                    h.update(whole.data(), whole.size());
                    uint8_t digest[32];
                    h.finalize(digest);
                    if (std::equal(digest, digest + 32, file.sha256)) {
                        fileComplete = true;
                        chunksDone += static_cast<uint32_t>(fileChunks);
                        totalDownloaded += file.size;
                        if (progress) {
                            dl.bytesDone = totalDownloaded;
                            dl.chunksDone = chunksDone;
                            progress(dl);
                        }
                        continue;
                    }
                }
            }
        }

        std::fstream out(outPath, std::ios::binary | std::ios::in | std::ios::out);
        if (!out.is_open()) {
            out.open(outPath, std::ios::binary | std::ios::out);
            if (!out) {
                VVM_LOG_ERROR("fetchModel: cannot create {}", outPath.string());
                return totalDownloaded;
            }
        }

        for (uint32_t ci = 0; ci < fileChunks; ++ci) {
            const uint64_t offset = static_cast<uint64_t>(ci) * manifest.chunkSize;
            TcpMessage req;
            req.type = MsgModelChunk;
            req.flags = TcpFlagsRequest;
            req.body.insert(req.body.end(), modelId.begin(), modelId.end());
            req.body.push_back(0);
            req.body.insert(req.body.end(), manifest.version.begin(), manifest.version.end());
            req.body.push_back(0);
            req.body.insert(req.body.end(), file.path.begin(), file.path.end());
            req.body.push_back(0);
            uint32_t idx = ci;
            req.body.push_back((idx >> 24) & 0xff);
            req.body.push_back((idx >> 16) & 0xff);
            req.body.push_back((idx >> 8) & 0xff);
            req.body.push_back(idx & 0xff);

            auto resp = transport.request(*conn, req, nullptr);
            if (!resp || resp->flags == TcpFlagsError) {
                VVM_LOG_ERROR("fetchModel: chunk request failed for {} chunk {}", file.path, ci);
                return totalDownloaded;
            }
            // Parse response body: status u32 + chunkSha256[32]
            if (resp->body.size() < 4 + 32) return totalDownloaded;
            uint32_t status = (resp->body[0] << 24) | (resp->body[1] << 16) |
                              (resp->body[2] << 8) | resp->body[3];
            if (status != 0) {
                VVM_LOG_ERROR("fetchModel: server error for {} chunk {}", file.path, ci);
                return totalDownloaded;
            }
            // Verify chunk hash
            uint8_t recvDigest[32];
            std::copy(resp->body.begin() + 4, resp->body.begin() + 4 + 32, recvDigest);
            Sha256 h;
            h.update(resp->stream.data(), resp->stream.size());
            uint8_t localDigest[32];
            h.finalize(localDigest);
            if (!std::equal(localDigest, localDigest + 32, recvDigest)) {
                VVM_LOG_ERROR("fetchModel: hash mismatch for {} chunk {}", file.path, ci);
                return totalDownloaded;
            }

            // Write chunk
            out.seekp(static_cast<std::streamoff>(offset));
            out.write(reinterpret_cast<const char*>(resp->stream.data()), resp->stream.size());
            if (!out) {
                VVM_LOG_ERROR("fetchModel: write failed for {} chunk {}", file.path, ci);
                return totalDownloaded;
            }
            out.flush();

            totalDownloaded += resp->stream.size();
            chunksDone++;
            if (progress) {
                dl.bytesDone = totalDownloaded;
                dl.chunksDone = chunksDone;
                progress(dl);
            }
        }
    }

    return totalDownloaded;
}

bool ModelHub::fetch(const std::string& connection,
                     const std::string& modelId,
                     const std::string& destDir,
                     const std::string& version,
                     ProgressFn progress,
                     int32_t timeoutMs) {
    auto manOpt = fetchManifest(connection, modelId, version, timeoutMs);
    if (!manOpt) return false;
    uint64_t got = fetchModel(connection, modelId, *manOpt, destDir, progress, timeoutMs);
    return got > 0 || manOpt->totalSize == 0;
}

} // namespace network
} // namespace vvm