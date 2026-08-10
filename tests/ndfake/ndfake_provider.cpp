// ndfake_provider.cpp - In-process Network Direct SPI provider (test double)
//
// Implements IND2Provider/Adapter/CompletionQueue/MemoryRegion/QueuePair/Connector/Listener
// on host RAM. Loaded via VVM_ND_PROVIDER_DLL env override in ndk_transport.cpp:openNdkAdapter.
// All operations are synchronous and complete immediately (no real async/overlapped).

#define INITGUID
#include <initguid.h>
#include <winsock2.h>
#include <ws2spi.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <ndspi.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ndfake_provider.h"

namespace {

// ============================================================================
// Helpers
// ============================================================================

inline void setOverlappedSuccess(OVERLAPPED* ovl) {
    if (ovl) ovl->Internal = 0; // STATUS_SUCCESS
}

inline HRESULT hrFromWin32(DWORD err) {
    return HRESULT_FROM_WIN32(err);
}

// Global token generator
static std::atomic<uint32_t> g_tokenGen{1};
inline uint32_t nextToken() {
    uint32_t t = g_tokenGen.fetch_add(1);
    return t == 0 ? 1 : t;
}

// Fake HANDLE generator (non-INVALID)
static std::atomic<uintptr_t> g_handleGen{0x1000};
inline HANDLE nextFakeHandle() {
    return reinterpret_cast<HANDLE>(g_handleGen.fetch_add(1));
}

// ============================================================================
// Fake memory region registry (shared across provider instance)
// ============================================================================
struct FakeMemoryInfo {
    void* base = nullptr;
    size_t length = 0;
};

static std::mutex g_mrMutex;
static std::unordered_map<uint32_t, FakeMemoryInfo> g_memoryRegions; // key = token (lkey == rkey)

uint32_t registerFakeRegion(void* base, size_t length) {
    std::lock_guard<std::mutex> lock(g_mrMutex);
    uint32_t token = nextToken();
    g_memoryRegions[token] = {base, length};
    return token;
}

bool unregisterFakeRegion(uint32_t token) {
    std::lock_guard<std::mutex> lock(g_mrMutex);
    return g_memoryRegions.erase(token) > 0;
}

bool validateRegion(uint32_t token, uint64_t remoteAddr, size_t size) {
    std::lock_guard<std::mutex> lock(g_mrMutex);
    auto it = g_memoryRegions.find(token);
    if (it == g_memoryRegions.end()) return false;
    uintptr_t base = reinterpret_cast<uintptr_t>(it->second.base);
    uintptr_t end = base + it->second.length;
    uintptr_t start = remoteAddr;
    return start >= base && start + size <= end;
}

void copyToRemote(uint32_t remoteToken, uint64_t remoteAddr, const void* src, size_t size) {
    std::lock_guard<std::mutex> lock(g_mrMutex);
    auto it = g_memoryRegions.find(remoteToken);
    if (it == g_memoryRegions.end()) return;
    // ND semantics: remoteAddress is the absolute virtual address of the
    // destination inside the remote region (region base + offset).
    void* dst = reinterpret_cast<void*>(remoteAddr);
    std::memcpy(dst, src, size);
}

void copyFromRemote(uint32_t remoteToken, uint64_t remoteAddr, void* dst, size_t size) {
    std::lock_guard<std::mutex> lock(g_mrMutex);
    auto it = g_memoryRegions.find(remoteToken);
    if (it == g_memoryRegions.end()) return;
    const void* src = reinterpret_cast<void*>(remoteAddr);
    std::memcpy(dst, src, size);
}

// ============================================================================
// Fake completion queue
// ============================================================================
struct FakeCompletionEntry {
    ND2_RESULT result;
};

class FakeCompletionQueue : public IND2CompletionQueue {
    std::atomic<ULONG> refCount_{1};
    std::mutex mutex_;
    std::vector<FakeCompletionEntry> completions_;
    bool cancelled_ = false;

public:
    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IND2CompletionQueue || riid == IID_IND2Overlapped) {
            *ppv = static_cast<IND2CompletionQueue*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++refCount_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG c = --refCount_;
        if (c == 0) delete this;
        return c;
    }

    // IND2Overlapped
    STDMETHODIMP CancelOverlappedRequests() override {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        return S_OK;
    }
    STDMETHODIMP GetOverlappedResult(OVERLAPPED* ovl, BOOL wait) override {
        (void)wait;
        setOverlappedSuccess(ovl);
        return S_OK;
    }

    // IND2CompletionQueue
    STDMETHODIMP GetNotifyAffinity(USHORT* pGroup, KAFFINITY* pAffinity) override {
        if (pGroup) *pGroup = 0;
        if (pAffinity) *pAffinity = 1;
        return S_OK;
    }
    STDMETHODIMP Resize(ULONG queueDepth) override { (void)queueDepth; return S_OK; }
    STDMETHODIMP Notify(ULONG type, OVERLAPPED* pOverlapped) override {
        (void)type;
        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }
    STDMETHODIMP_(ULONG) GetResults(ND2_RESULT results[], ULONG nResults) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ULONG count = 0;
        while (count < nResults && !completions_.empty()) {
            results[count++] = completions_.front().result;
            completions_.erase(completions_.begin());
        }
        return count;
    }

    void postCompletion(const ND2_RESULT& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        completions_.push_back({result});
    }
};

// ============================================================================
// Fake memory region
// ============================================================================
class FakeMemoryRegion : public IND2MemoryRegion {
    std::atomic<ULONG> refCount_{1};
    uint32_t token_ = 0;

public:
    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IND2MemoryRegion || riid == IID_IND2Overlapped) {
            *ppv = static_cast<IND2MemoryRegion*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++refCount_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG c = --refCount_;
        if (c == 0) delete this;
        return c;
    }

    // IND2Overlapped
    STDMETHODIMP CancelOverlappedRequests() override { return S_OK; }
    STDMETHODIMP GetOverlappedResult(OVERLAPPED* ovl, BOOL wait) override {
        (void)wait;
        setOverlappedSuccess(ovl);
        return S_OK;
    }

    // IND2MemoryRegion
    STDMETHODIMP Register(const void* pBuffer, SIZE_T cbBuffer, ULONG flags, OVERLAPPED* pOverlapped) override {
        (void)flags;
        token_ = registerFakeRegion(const_cast<void*>(pBuffer), cbBuffer);
        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }
    STDMETHODIMP Deregister(OVERLAPPED* pOverlapped) override {
        if (token_) unregisterFakeRegion(token_);
        token_ = 0;
        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }
    STDMETHODIMP_(UINT32) GetLocalToken() override { return token_; }
    STDMETHODIMP_(UINT32) GetRemoteToken() override { return token_; }
};

// ============================================================================
// Fake queue pair
// ============================================================================
class FakeQueuePair : public IND2QueuePair {
    std::atomic<ULONG> refCount_{1};
    FakeCompletionQueue* cq_ = nullptr;
    void* qpContext_ = nullptr;

public:
    FakeQueuePair(FakeCompletionQueue* cq, void* context) : cq_(cq), qpContext_(context) {
        if (cq_) cq_->AddRef();
    }
    ~FakeQueuePair() { if (cq_) cq_->Release(); }

    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IND2QueuePair) {
            *ppv = static_cast<IND2QueuePair*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++refCount_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG c = --refCount_;
        if (c == 0) delete this;
        return c;
    }

    // IND2QueuePair
    STDMETHODIMP Flush() override { return S_OK; }
    STDMETHODIMP Send(void* requestContext, const ND2_SGE sge[], ULONG nSge, ULONG flags) override {
        (void)requestContext; (void)sge; (void)nSge; (void)flags; return S_OK;
    }
    STDMETHODIMP Receive(void* requestContext, const ND2_SGE sge[], ULONG nSge) override {
        (void)requestContext; (void)sge; (void)nSge; return S_OK;
    }
    STDMETHODIMP Bind(void* requestContext, IUnknown* pMemoryRegion, IUnknown* pMemoryWindow,
                       const void* pBuffer, SIZE_T cbBuffer, ULONG flags) override {
        (void)requestContext; (void)pMemoryRegion; (void)pMemoryWindow; (void)pBuffer; (void)cbBuffer; (void)flags;
        return S_OK;
    }
    STDMETHODIMP Invalidate(void* requestContext, IUnknown* pMemoryWindow, ULONG flags) override {
        (void)requestContext; (void)pMemoryWindow; (void)flags; return S_OK;
    }
    STDMETHODIMP Read(void* requestContext, const ND2_SGE sge[], ULONG nSge,
                       UINT64 remoteAddress, UINT32 remoteToken, ULONG flags) override {
        (void)flags;
        if (nSge == 0 || !sge) return S_OK;
        size_t total = 0;
        for (ULONG i = 0; i < nSge; ++i) total += sge[i].BufferLength;
        if (total == 0) return S_OK;

        copyFromRemote(remoteToken, remoteAddress, const_cast<void*>(sge[0].Buffer), total);

        ND2_RESULT result{};
        result.Status = S_OK;
        result.BytesTransferred = static_cast<ULONG>(total);
        result.QueuePairContext = qpContext_;
        result.RequestContext = requestContext;
        cq_->postCompletion(result);
        return S_OK;
    }
    STDMETHODIMP Write(void* requestContext, const ND2_SGE sge[], ULONG nSge,
                        UINT64 remoteAddress, UINT32 remoteToken, ULONG flags) override {
        (void)flags;
        if (nSge == 0 || !sge) return S_OK;
        size_t total = 0;
        for (ULONG i = 0; i < nSge; ++i) total += sge[i].BufferLength;
        if (total == 0) return S_OK;

        copyToRemote(remoteToken, remoteAddress, sge[0].Buffer, total);

        ND2_RESULT result{};
        result.Status = S_OK;
        result.BytesTransferred = static_cast<ULONG>(total);
        result.QueuePairContext = qpContext_;
        result.RequestContext = requestContext;
        cq_->postCompletion(result);
        return S_OK;
    }
};

// ============================================================================
// Fake connector
// ============================================================================
struct PendingConnect {
    IND2Connector* pending = nullptr;
    sockaddr_in peerAddr{};
};

static std::mutex g_pendingMutex;
static std::condition_variable g_pendingCv;
static std::vector<PendingConnect> g_pendingConnects;
static bool g_pendingCancelled = false;

class FakeConnector : public IND2Connector {
    std::atomic<ULONG> refCount_{1};
    sockaddr_in localAddr_{};
    sockaddr_in peerAddr_{};
    bool connected_ = false;

public:
    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IND2Connector || riid == IID_IND2Overlapped) {
            *ppv = static_cast<IND2Connector*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++refCount_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG c = --refCount_;
        if (c == 0) delete this;
        return c;
    }

    // IND2Overlapped
    STDMETHODIMP CancelOverlappedRequests() override { return S_OK; }
    STDMETHODIMP GetOverlappedResult(OVERLAPPED* ovl, BOOL wait) override {
        (void)wait;
        setOverlappedSuccess(ovl);
        return S_OK;
    }

    // IND2Connector
    STDMETHODIMP Bind(const sockaddr* pAddress, ULONG cbAddress) override {
        if (cbAddress >= sizeof(sockaddr_in)) {
            std::memcpy(&localAddr_, pAddress, sizeof(sockaddr_in));
        }
        return S_OK;
    }
    STDMETHODIMP Connect(IUnknown* pQueuePair, const sockaddr* pDestAddress, ULONG cbDestAddress,
                          ULONG inboundReadLimit, ULONG outboundReadLimit,
                          const void* pPrivateData, ULONG cbPrivateData,
                          OVERLAPPED* pOverlapped) override {
        (void)pQueuePair; (void)inboundReadLimit; (void)outboundReadLimit;
        (void)pPrivateData; (void)cbPrivateData;
        if (cbDestAddress >= sizeof(sockaddr_in)) {
            std::memcpy(&peerAddr_, pDestAddress, sizeof(sockaddr_in));
        }
        connected_ = true;

        // Deliver to listener's pending queue
        {
            std::lock_guard<std::mutex> lock(g_pendingMutex);
            g_pendingConnects.push_back({nullptr, peerAddr_}); // pending filled by listener
        }
        g_pendingCv.notify_one();

        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }
    STDMETHODIMP CompleteConnect(OVERLAPPED* pOverlapped) override {
        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }
    STDMETHODIMP Accept(IUnknown* pQueuePair, ULONG inboundReadLimit, ULONG outboundReadLimit,
                         const void* pPrivateData, ULONG cbPrivateData, OVERLAPPED* pOverlapped) override {
        (void)pQueuePair; (void)inboundReadLimit; (void)outboundReadLimit;
        (void)pPrivateData; (void)cbPrivateData;
        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }
    STDMETHODIMP Reject(const void* pPrivateData, ULONG cbPrivateData) override {
        (void)pPrivateData; (void)cbPrivateData; return S_OK;
    }
    STDMETHODIMP GetReadLimits(ULONG* pInboundReadLimit, ULONG* pOutboundReadLimit) override {
        if (pInboundReadLimit) *pInboundReadLimit = 16;
        if (pOutboundReadLimit) *pOutboundReadLimit = 16;
        return S_OK;
    }
    STDMETHODIMP GetPrivateData(void* pPrivateData, ULONG* pcbPrivateData) override {
        (void)pPrivateData; (void)pcbPrivateData; return S_OK;
    }
    STDMETHODIMP GetLocalAddress(sockaddr* pAddress, ULONG* pAddressLength) override {
        if (*pAddressLength >= sizeof(sockaddr_in)) {
            std::memcpy(pAddress, &localAddr_, sizeof(sockaddr_in));
            *pAddressLength = sizeof(sockaddr_in);
        } else {
            *pAddressLength = sizeof(sockaddr_in);
            return ND_BUFFER_OVERFLOW;
        }
        return S_OK;
    }
    STDMETHODIMP GetPeerAddress(sockaddr* pAddress, ULONG* pAddressLength) override {
        if (*pAddressLength >= sizeof(sockaddr_in)) {
            std::memcpy(pAddress, &peerAddr_, sizeof(sockaddr_in));
            *pAddressLength = sizeof(sockaddr_in);
        } else {
            *pAddressLength = sizeof(sockaddr_in);
            return ND_BUFFER_OVERFLOW;
        }
        return S_OK;
    }
    STDMETHODIMP NotifyDisconnect(OVERLAPPED* pOverlapped) override {
        (void)pOverlapped; return S_OK;
    }
    STDMETHODIMP Disconnect(OVERLAPPED* pOverlapped) override {
        connected_ = false;
        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }

    void setPendingConnector(IND2Connector* pending) {
        // Called by listener when it takes this from pending queue
        (void)pending;
    }
};

// ============================================================================
// Fake listener
// ============================================================================
class FakeListener : public IND2Listener {
    std::atomic<ULONG> refCount_{1};
    uint16_t boundPort_ = 0;
    std::atomic<bool> cancelled_{false};

public:
    FakeListener() = default;
    ~FakeListener() { cancelled_ = true; g_pendingCv.notify_all(); }

    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IND2Listener || riid == IID_IND2Overlapped) {
            *ppv = static_cast<IND2Listener*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++refCount_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG c = --refCount_;
        if (c == 0) delete this;
        return c;
    }

    // IND2Overlapped
    STDMETHODIMP CancelOverlappedRequests() override {
        cancelled_ = true;
        {
            std::lock_guard<std::mutex> lock(g_pendingMutex);
            g_pendingCancelled = true;
        }
        g_pendingCv.notify_all();
        return S_OK;
    }
    STDMETHODIMP GetOverlappedResult(OVERLAPPED* ovl, BOOL wait) override {
        (void)wait;
        setOverlappedSuccess(ovl);
        return S_OK;
    }

    // IND2Listener
    STDMETHODIMP Bind(const sockaddr* pAddress, ULONG cbAddress) override {
        if (cbAddress >= sizeof(sockaddr_in)) {
            sockaddr_in* a = const_cast<sockaddr_in*>(reinterpret_cast<const sockaddr_in*>(pAddress));
            boundPort_ = ntohs(a->sin_port);
        }
        return S_OK;
    }
    STDMETHODIMP Listen(ULONG backlog) override { (void)backlog; return S_OK; }
    STDMETHODIMP GetLocalAddress(sockaddr* pAddress, ULONG* pcbAddress) override {
        if (*pcbAddress >= sizeof(sockaddr_in)) {
            sockaddr_in a{};
            a.sin_family = AF_INET;
            a.sin_addr.s_addr = INADDR_ANY;
            a.sin_port = htons(boundPort_);
            std::memcpy(pAddress, &a, sizeof(sockaddr_in));
            *pcbAddress = sizeof(sockaddr_in);
        } else {
            *pcbAddress = sizeof(sockaddr_in);
            return ND_BUFFER_OVERFLOW;
        }
        return S_OK;
    }
    STDMETHODIMP GetConnectionRequest(IUnknown* pConnector, OVERLAPPED* pOverlapped) override {
        // Block until a pending connect arrives or cancelled
        std::unique_lock<std::mutex> lock(g_pendingMutex);
        g_pendingCv.wait(lock, []{ return g_pendingCancelled || !g_pendingConnects.empty(); });

        if (g_pendingCancelled || g_pendingConnects.empty()) {
            return ND_CANCELED;
        }

        // Pop one pending
        PendingConnect pc = g_pendingConnects.front();
        g_pendingConnects.erase(g_pendingConnects.begin());
        lock.unlock();

        // The pConnector passed is the "pending" connector created by acceptLoop
        // Cast to FakeConnector and store peer address
        FakeConnector* fc = dynamic_cast<FakeConnector*>(static_cast<IND2Connector*>(pConnector));
        if (fc) {
            fc->setPendingConnector(static_cast<IND2Connector*>(pConnector));
        }

        setOverlappedSuccess(pOverlapped);
        return S_OK;
    }
};

// ============================================================================
// Fake adapter
// ============================================================================
class FakeAdapter : public IND2Adapter {
    std::atomic<ULONG> refCount_{1};
    FakeListener* listener_ = nullptr;
    FakeCompletionQueue* cq_ = nullptr;

public:
    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IND2Adapter) {
            *ppv = static_cast<IND2Adapter*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++refCount_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG c = --refCount_;
        if (c == 0) delete this;
        return c;
    }

    // IND2Adapter
    STDMETHODIMP CreateOverlappedFile(HANDLE* phOverlappedFile) override {
        *phOverlappedFile = nextFakeHandle();
        return S_OK;
    }
    STDMETHODIMP Query(ND2_ADAPTER_INFO* pInfo, ULONG* pcbInfo) override {
        if (*pcbInfo < sizeof(ND2_ADAPTER_INFO)) {
            *pcbInfo = sizeof(ND2_ADAPTER_INFO);
            return ND_BUFFER_OVERFLOW;
        }
        std::memset(pInfo, 0, sizeof(ND2_ADAPTER_INFO));
        pInfo->InfoVersion = ND_VERSION_2;
        pInfo->VendorId = 0xFACA;
        pInfo->DeviceId = 0x0011;
        pInfo->AdapterId = 0xDEADBEEFCAFEBABE;
        pInfo->MaxRegistrationSize = 0x100000000ULL; // 4GB
        pInfo->MaxWindowSize = 0x100000; // 1MB
        pInfo->MaxInitiatorSge = 4;
        pInfo->MaxReceiveSge = 4;
        pInfo->MaxReadSge = 4;
        pInfo->MaxTransferLength = 0x40000000; // 1GB
        pInfo->MaxInlineDataSize = 256;
        pInfo->MaxInboundReadLimit = 16;
        pInfo->MaxOutboundReadLimit = 16;
        pInfo->MaxReceiveQueueDepth = 2048;
        pInfo->MaxInitiatorQueueDepth = 2048;
        pInfo->MaxSharedReceiveQueueDepth = 1024;
        pInfo->MaxCompletionQueueDepth = 4096;
        pInfo->InlineRequestThreshold = 256;
        pInfo->LargeRequestThreshold = 0x100000;
        pInfo->MaxCallerData = 64;
        pInfo->MaxCalleeData = 64;
        *pcbInfo = sizeof(ND2_ADAPTER_INFO);
        return S_OK;
    }
    STDMETHODIMP QueryAddressList(SOCKET_ADDRESS_LIST* pAddressList, ULONG* pcbAddressList) override {
        size_t need = sizeof(SOCKET_ADDRESS_LIST) + sizeof(SOCKET_ADDRESS);
        if (!pAddressList) {
            *pcbAddressList = static_cast<ULONG>(need);
            return ND_BUFFER_OVERFLOW;
        }
        if (*pcbAddressList < need) {
            *pcbAddressList = static_cast<ULONG>(need);
            return ND_BUFFER_OVERFLOW;
        }
        pAddressList->iAddressCount = 1;
        pAddressList->Address[0].iSockaddrLength = sizeof(sockaddr_in);
        sockaddr_in* a = reinterpret_cast<sockaddr_in*>(pAddressList->Address[0].lpSockaddr);
        std::memset(a, 0, sizeof(*a));
        a->sin_family = AF_INET;
        a->sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a->sin_port = 0;
        *pcbAddressList = static_cast<ULONG>(need);
        return S_OK;
    }
    STDMETHODIMP CreateCompletionQueue(REFIID iid, HANDLE hOverlappedFile, ULONG queueDepth,
                                        USHORT group, KAFFINITY affinity, void** ppCompletionQueue) override {
        (void)hOverlappedFile; (void)queueDepth; (void)group; (void)affinity;
        if (iid != IID_IND2CompletionQueue) return E_NOINTERFACE;
        FakeCompletionQueue* cq = new FakeCompletionQueue();
        cq_ = cq;
        *ppCompletionQueue = cq;
        return S_OK;
    }
    STDMETHODIMP CreateMemoryRegion(REFIID iid, HANDLE hOverlappedFile, void** ppMemoryRegion) override {
        (void)hOverlappedFile;
        if (iid != IID_IND2MemoryRegion) return E_NOINTERFACE;
        *ppMemoryRegion = new FakeMemoryRegion();
        return S_OK;
    }
STDMETHODIMP CreateMemoryWindow(REFIID iid, void** ppMemoryWindow) override {
        (void)iid; (void)ppMemoryWindow;
        return E_NOTIMPL;
    }
    STDMETHODIMP CreateSharedReceiveQueue(REFIID iid, HANDLE hOverlappedFile, ULONG queueDepth,
                                          ULONG maxRequestSge, ULONG notifyThreshold,
                                          USHORT group, KAFFINITY affinity,
                                          void** ppSharedReceiveQueue) override {
        (void)hOverlappedFile; (void)queueDepth; (void)maxRequestSge;
        (void)notifyThreshold; (void)group; (void)affinity; (void)iid; (void)ppSharedReceiveQueue;
        return E_NOTIMPL;
    }
    STDMETHODIMP CreateQueuePair(REFIID iid, IUnknown* pReceiveCompletionQueue, IUnknown* pInitiatorCompletionQueue,
                                  void* context, ULONG receiveQueueDepth, ULONG initiatorQueueDepth,
                                  ULONG maxReceiveRequestSge, ULONG maxInitiatorRequestSge,
                                  ULONG inlineDataSize, void** ppQueuePair) override {
        (void)receiveQueueDepth; (void)initiatorQueueDepth;
        (void)maxReceiveRequestSge; (void)maxInitiatorRequestSge; (void)inlineDataSize;
        if (iid != IID_IND2QueuePair) return E_NOINTERFACE;
        FakeCompletionQueue* cq = dynamic_cast<FakeCompletionQueue*>(pReceiveCompletionQueue);
        if (!cq) cq = dynamic_cast<FakeCompletionQueue*>(pInitiatorCompletionQueue);
        *ppQueuePair = new FakeQueuePair(cq, context);
        return S_OK;
    }
    STDMETHODIMP CreateQueuePairWithSrq(REFIID iid, IUnknown* pReceiveCompletionQueue,
                                         IUnknown* pInitiatorCompletionQueue,
                                         IUnknown* pSharedReceiveQueue, void* context,
                                         ULONG initiatorQueueDepth, ULONG maxInitiatorRequestSge,
                                         ULONG inlineDataSize, void** ppQueuePair) override {
        return CreateQueuePair(iid, pReceiveCompletionQueue, pInitiatorCompletionQueue,
                                context, 0, initiatorQueueDepth, 0, maxInitiatorRequestSge,
                                inlineDataSize, ppQueuePair);
    }
    STDMETHODIMP CreateConnector(REFIID iid, HANDLE hOverlappedFile, void** ppConnector) override {
        (void)hOverlappedFile;
        if (iid != IID_IND2Connector) return E_NOINTERFACE;
        *ppConnector = new FakeConnector();
        return S_OK;
    }
    STDMETHODIMP CreateListener(REFIID iid, HANDLE hOverlappedFile, void** ppListener) override {
        (void)hOverlappedFile;
        if (iid != IID_IND2Listener) return E_NOINTERFACE;
        FakeListener* l = new FakeListener();
        listener_ = l;
        *ppListener = l;
        return S_OK;
    }
};

// ============================================================================
// Fake provider
// ============================================================================
class FakeProvider : public IND2Provider {
    std::atomic<ULONG> refCount_{1};

public:
    // IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (riid == IID_IUnknown || riid == IID_IND2Provider) {
            *ppv = static_cast<IND2Provider*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return ++refCount_; }
    STDMETHODIMP_(ULONG) Release() override {
        ULONG c = --refCount_;
        if (c == 0) delete this;
        return c;
    }

    // IND2Provider
    STDMETHODIMP QueryAddressList(SOCKET_ADDRESS_LIST* pAddressList, ULONG* pcbAddressList) override {
        size_t need = sizeof(SOCKET_ADDRESS_LIST) + sizeof(SOCKET_ADDRESS);
        if (!pAddressList) {
            *pcbAddressList = static_cast<ULONG>(need);
            return ND_BUFFER_OVERFLOW;
        }
        if (*pcbAddressList < need) {
            *pcbAddressList = static_cast<ULONG>(need);
            return ND_BUFFER_OVERFLOW;
        }
        pAddressList->iAddressCount = 1;
        pAddressList->Address[0].iSockaddrLength = sizeof(sockaddr_in);
        sockaddr_in* a = reinterpret_cast<sockaddr_in*>(pAddressList->Address[0].lpSockaddr);
        std::memset(a, 0, sizeof(*a));
        a->sin_family = AF_INET;
        a->sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a->sin_port = 0;
        *pcbAddressList = static_cast<ULONG>(need);
        return S_OK;
    }
    STDMETHODIMP ResolveAddress(const sockaddr* pAddress, ULONG cbAddress, UINT64* pAdapterId) override {
        (void)pAddress; (void)cbAddress;
        *pAdapterId = 0xCAFEBABEDEADBEEF;
        return S_OK;
    }
    STDMETHODIMP OpenAdapter(REFIID iid, UINT64 adapterId, void** ppAdapter) override {
        (void)adapterId;
        if (iid != IID_IND2Adapter) return E_NOINTERFACE;
        *ppAdapter = new FakeAdapter();
        return S_OK;
    }
};

} // anonymous namespace

// ============================================================================
// DLL exports
// ============================================================================

extern "C" NDFAKE_API HRESULT WINAPI NDFakeGetProviderClsid(CLSID* pOut) {
    if (!pOut) return E_POINTER;
    *pOut = NDFAKE_PROVIDER_CLSID;
    return S_OK;
}

extern "C" HRESULT WINAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    if (rclsid != NDFAKE_PROVIDER_CLSID) return CLASS_E_CLASSNOTAVAILABLE;
    FakeProvider* provider = new FakeProvider();
    return provider->QueryInterface(riid, ppv);
}

// DllMain for process attach/detach cleanup
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    (void)hModule; (void)lpReserved;
    if (ul_reason_for_call == DLL_PROCESS_DETACH) {
        // Clear global state
        std::lock_guard<std::mutex> lock(g_mrMutex);
        g_memoryRegions.clear();
        std::lock_guard<std::mutex> lock2(g_pendingMutex);
        g_pendingConnects.clear();
        g_pendingCancelled = true;
        g_pendingCv.notify_all();
    }
    return TRUE;
}