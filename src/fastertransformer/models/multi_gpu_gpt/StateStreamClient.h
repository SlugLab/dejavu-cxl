#pragma once

#ifdef USE_GRPC_DEJAVU
// Full gRPC-based implementation
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "src/fastertransformer/models/multi_gpu_gpt/state_stream.pb.h"
#include "src/fastertransformer/models/multi_gpu_gpt/state_stream.grpc.pb.h"

using namespace grpc;
using namespace dejavu;

class DejaVuClient {

	private:
		std::unique_ptr<DejaVuManager::Stub> stub_;
	public:
		DejaVuClient(std::shared_ptr<Channel> channel);

		int GetSlot();
		void MarkComplete(int slot_id);
		int GetSlotImpl(PushRequest &req, PushResponse *resp);
		void MarkCompleteImpl(CompleteRequest &req, CompleteResponse *resp);
};

#else
// Stub implementation without gRPC dependencies
#include <cstdio>

class DejaVuClient {
public:
    DejaVuClient() {}

    // Stub constructor that matches gRPC API for compatibility
    template<typename T>
    DejaVuClient(T) {}

    int GetSlot() {
        return 0;
    }

    void MarkComplete(int slot_id) {
        // No-op stub
    }
};

#endif  // USE_GRPC_DEJAVU
