#include "vulkan_vm/network/model_registry.hpp"
#include "vulkan_vm/network/sha256.hpp"
#include "vulkan_vm/utils.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

using vvm::network::ModelHub;
using vvm::network::ModelManifest;
using vvm::network::Sha256;

static int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

static void createTestModel(const std::string& dir) {
    fs::create_directories(dir + "/weights");
    fs::create_directories(dir + "/tokenizer");

    { // weights.safetensors
        std::ofstream f(dir + "/weights/model.safetensors", std::ios::binary);
        std::string data(1024 * 1024, 'A');
        f.write(data.data(), data.size());
    }
    { // extra.bin
        std::ofstream f(dir + "/weights/extra.bin", std::ios::binary);
        std::string data(2 * 1024 * 1024, 'B');
        f.write(data.data(), data.size());
    }
    { // tokenizer.json
        std::ofstream f(dir + "/tokenizer/tokenizer.json");
        f << R"({"model": "test", "version": "1.0"})";
    }
    { // config.json
        std::ofstream f(dir + "/config.json");
        f << R"({"model_type": "llama", "hidden_size": 4096})";
    }
}

static void cleanupDir(const std::string& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
}

static void hashFile(const std::string& path, uint8_t out[32]) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::fill(out, out + 32, 0); return; }
    Sha256 h;
    std::vector<uint8_t> buf(1u << 20);
    while (in) {
        in.read(reinterpret_cast<char*>(buf.data()), buf.size());
        h.update(buf.data(), static_cast<size_t>(in.gcount()));
    }
    h.finalize(out);
}

static void checkFile(const std::string& relPath,
                      const std::string& srcDir,
                      const std::string& dstDir) {
    fs::path src = fs::path(srcDir) / relPath;
    fs::path dst = fs::path(dstDir) / relPath;
    CHECK(fs::exists(dst));
    CHECK(fs::file_size(src) == fs::file_size(dst));
    uint8_t d1[32], d2[32];
    {
        std::ifstream in1(src, std::ios::binary);
        Sha256 h1;
        std::vector<uint8_t> buf(1u << 20);
        while (in1) {
            in1.read(reinterpret_cast<char*>(buf.data()), buf.size());
            h1.update(buf.data(), static_cast<size_t>(in1.gcount()));
        }
        h1.finalize(d1);
    }
    {
        std::ifstream in2(dst, std::ios::binary);
        Sha256 h2;
        std::vector<uint8_t> buf(1u << 20);
        while (in2) {
            in2.read(reinterpret_cast<char*>(buf.data()), buf.size());
            h2.update(buf.data(), static_cast<size_t>(in2.gcount()));
        }
        h2.finalize(d2);
    }
    CHECK(std::equal(d1, d1 + 32, d2));
}

int main() {
    const std::string cacheDir = "./test_model_store";
    const std::string sourceDir = "./test_model_source";
    const std::string fetchDir = "./test_model_fetch";
    const std::string fetchDir2 = "./test_model_fetch2";

    cleanupDir(cacheDir);
    cleanupDir(sourceDir);
    cleanupDir(fetchDir);
    cleanupDir(fetchDir2);

    createTestModel(sourceDir);

    // --- Start hub ---
    ModelHub hub(cacheDir);
    CHECK(hub.start("127.0.0.1", 51020));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // --- Publish model ---
    CHECK(hub.publish("chonk/test-model", sourceDir, "v1"));

    // Verify manifest exists locally
    auto manifestOpt = hub.getManifest("chonk/test-model", "v1");
    CHECK(manifestOpt.has_value());
    const ModelManifest& man = *manifestOpt;
    CHECK(man.modelId == "chonk/test-model");
    CHECK(man.version == "v1");
    CHECK(man.files.size() == 4);
    CHECK(man.totalSize > 3 * 1024 * 1024);

    // --- Fetch manifest from hub (client) ---
    auto fetchedManOpt = ModelHub::fetchManifest("127.0.0.1:51020", "chonk/test-model", "v1");
    CHECK(fetchedManOpt.has_value());
    const ModelManifest& fm = *fetchedManOpt;
    CHECK(fm.modelId == "chonk/test-model");
    CHECK(fm.version == "v1");
    CHECK(fm.files.size() == man.files.size());

    // --- Fetch full model to destination ---
    bool fetchOk = ModelHub::fetch("127.0.0.1:51020", "chonk/test-model", fetchDir, "v1");
    CHECK(fetchOk);

    // Verify fetched files match source
    checkFile("config.json", sourceDir, fetchDir);
    checkFile("weights/model.safetensors", sourceDir, fetchDir);
    checkFile("weights/extra.bin", sourceDir, fetchDir);
    checkFile("tokenizer/tokenizer.json", sourceDir, fetchDir);

    // --- Test cache hit (second fetch should skip already-present chunks) ---
    cleanupDir("./test_model_fetch2");
    bool fetch2 = ModelHub::fetch("127.0.0.1:51020", "chonk/test-model", "./test_model_fetch2", "v1");
    CHECK(fetch2);

    checkFile("config.json", sourceDir, "./test_model_fetch2");
    checkFile("weights/model.safetensors", sourceDir, "./test_model_fetch2");

    // --- List models ---
    auto models = hub.listModels();
    CHECK(models.size() == 1);
    CHECK(models[0].model_id == "chonk/test-model");
    CHECK(models[0].version == "v1");

    // --- Cleanup ---
    hub.stop();

    cleanupDir(cacheDir);
    cleanupDir(sourceDir);
    cleanupDir(fetchDir);
    cleanupDir("./test_model_fetch2");

    if (failures == 0) {
        std::printf("=== ALL MODEL REGISTRY TESTS PASSED (0 failures) ===\n");
        return 0;
    }
    std::printf("=== SOME TESTS FAILED (%d failures) ===\n", failures);
    return 1;
}