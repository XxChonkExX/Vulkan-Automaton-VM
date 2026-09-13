// moe_pack: pack/list/verify/slice tool for MoE expert .vmex files.
// CPU-only (no GPU needed). Pack layout v1, see storage_stream.hpp.
//
//   moe_pack pack   out.vmex id:path [id:path ...]  (each path = one expert blob)
//   moe_pack list   pack.vmex
//   moe_pack verify pack.vmex                        (re-read + checksum)
//   moe_pack slice  in.bin out.blob <offset> <size>  (carve GGUF/safetensors shard)
//   moe_pack demo   out.vmex [sizeMB]                (synthetic experts 0,1,2)

#include "vulkan_vm/storage_stream.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using vvm::storage::ExpertSpec;

static uint64_t fnv1a(const std::vector<uint8_t>& b) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t c : b) { h ^= c; h *= 1099511628211ull; }
    return h;
}

static bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize(n > 0 ? static_cast<size_t>(n) : 0);
    if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
    return static_cast<bool>(f);
}

static int cmdPack(const std::vector<std::string>& args) {
    // args: out.vmex id:path ...
    if (args.size() < 3) {
        std::cerr << "usage: moe_pack pack out.vmex id:path [id:path ...]\n";
        return 1;
    }
    const std::string out = args[1];
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> experts;
    for (size_t i = 2; i < args.size(); ++i) {
        const auto& tok = args[i];
        auto colon = tok.find(':');
        if (colon == std::string::npos) {
            std::cerr << "bad expert '" << tok << "' (want id:path)\n";
            return 1;
        }
        uint32_t id = static_cast<uint32_t>(std::stoul(tok.substr(0, colon)));
        std::string path = tok.substr(colon + 1);
        std::vector<uint8_t> blob;
        if (!readFile(path, blob)) {
            std::cerr << "cannot read " << path << "\n";
            return 1;
        }
        std::cout << "  expert " << id << ": " << blob.size() << " bytes from " << path << "\n";
        experts.emplace_back(id, std::move(blob));
    }
    std::vector<ExpertSpec> specs;
    if (!vvm::storage::writePackFile(out, experts, &specs)) {
        std::cerr << "write failed: " << out << "\n";
        return 1;
    }
    std::cout << "wrote " << out << " (" << specs.size() << " experts)\n";
    for (auto& s : specs)
        std::cout << "  id=" << s.expertId << " off=" << s.fileOffset << " bytes=" << s.bytes << "\n";
    return 0;
}

static int cmdList(const std::string& path) {
    auto table = vvm::storage::readPackTable(path);
    if (!table) {
        std::cerr << "bad pack: " << path << "\n";
        return 1;
    }
    std::cout << path << ": " << table->size() << " experts\n";
    for (auto& s : *table)
        std::cout << "  id=" << s.expertId << " off=" << s.fileOffset << " bytes=" << s.bytes << "\n";
    return 0;
}

static int cmdVerify(const std::string& path) {
    auto table = vvm::storage::readPackTable(path);
    if (!table) {
        std::cerr << "bad pack: " << path << "\n";
        return 1;
    }
    bool ok = true;
    for (auto& s : *table) {
        auto blob = vvm::storage::readPackBlob(path, s);
        if (!blob || blob->size() != s.bytes) {
            std::cout << "  id=" << s.expertId << " SHORT READ\n";
            ok = false;
            continue;
        }
        std::cout << "  id=" << s.expertId << " bytes=" << s.bytes
                  << " fnv=0x" << std::hex << fnv1a(*blob) << std::dec << " ok\n";
    }
    std::cout << (ok ? "verify OK\n" : "verify FAILED\n");
    return ok ? 0 : 1;
}

static int cmdSlice(const std::vector<std::string>& args) {
    // slice in.bin out.blob offset size  (offset/size accept k/m/g suffix)
    if (args.size() != 5) {
        std::cerr << "usage: moe_pack slice in.bin out.blob <offset> <size>  (k/m/g suffix ok)\n";
        return 1;
    }
    auto parse = [](const std::string& s) -> uint64_t {
        uint64_t mult = 1;
        std::string n = s;
        if (!n.empty()) {
            char c = n.back();
            if (c == 'k' || c == 'K') { mult = 1024; n.pop_back(); }
            else if (c == 'm' || c == 'M') { mult = 1024 * 1024; n.pop_back(); }
            else if (c == 'g' || c == 'G') { mult = 1024ull * 1024 * 1024; n.pop_back(); }
        }
        return std::stoull(n) * mult;
    };
    uint64_t off = parse(args[3]), size = parse(args[4]);
    std::ifstream in(args[1], std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << args[1] << "\n";
        return 1;
    }
    in.seekg(static_cast<std::streamoff>(off));
    if (!in) {
        std::cerr << "seek failed\n";
        return 1;
    }
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    if (!in) {
        std::cerr << "short read\n";
        return 1;
    }
    std::ofstream out(args[2], std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(size));
    std::cout << "sliced " << size << " bytes @ " << off << " -> " << args[2] << "\n";
    return out ? 0 : 1;
}

static int cmdDemo(const std::vector<std::string>& args) {
    // demo out.vmex [sizeMB]
    if (args.size() < 2) {
        std::cerr << "usage: moe_pack demo out.vmex [sizeMB]\n";
        return 1;
    }
    size_t sizeMB = args.size() > 2 ? static_cast<size_t>(std::stoul(args[2])) : 4;
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> experts;
    for (uint32_t id = 0; id < 3; ++id) {
        std::vector<uint8_t> blob(sizeMB * 1024 * 1024, static_cast<uint8_t>(0xA0 + id));
        // stamp first/last 8 bytes with id so GPU round-trip can verify
        for (int i = 0; i < 8 && i < static_cast<int>(blob.size()); ++i) blob[i] = static_cast<uint8_t>(id);
        for (int i = 0; i < 8 && i < static_cast<int>(blob.size()); ++i)
            blob[blob.size() - 1 - i] = static_cast<uint8_t>(id);
        experts.emplace_back(id, std::move(blob));
    }
    if (!vvm::storage::writePackFile(args[1], experts, nullptr)) {
        std::cerr << "write failed\n";
        return 1;
    }
    std::cout << "demo pack: " << args[1] << " (3 experts x " << sizeMB << "MB)\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: moe_pack <pack|list|verify|slice|demo> ...\n";
        return 1;
    }
    std::string cmd = argv[1];
    std::vector<std::string> args(argv + 1, argv + argc);
    if (cmd == "pack") return cmdPack(args);
    if (cmd == "list") {
        if (args.size() != 2) {
            std::cerr << "usage: moe_pack list pack.vmex\n";
            return 1;
        }
        return cmdList(args[1]);
    }
    if (cmd == "verify") {
        if (args.size() != 2) {
            std::cerr << "usage: moe_pack verify pack.vmex\n";
            return 1;
        }
        return cmdVerify(args[1]);
    }
    if (cmd == "slice") return cmdSlice(args);
    if (cmd == "demo") return cmdDemo(args);
    std::cerr << "unknown cmd: " << cmd << "\n";
    return 1;
}
