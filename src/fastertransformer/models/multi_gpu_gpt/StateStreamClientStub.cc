// Stub implementation of DejaVuClient for builds without gRPC dependencies
// This provides the minimum interface needed by ParallelGptContextDecoder

#include <cstdio>

class DejaVuClient {
public:
    DejaVuClient() {}

    int GetSlot() {
        // Stub - return 0 as a default slot
        return 0;
    }

    void MarkComplete(int slot_id) {
        // Stub - just log and return
        printf("[DejaVuClient Stub] MarkComplete called with slot_id=%d (no-op)\n", slot_id);
    }
};
