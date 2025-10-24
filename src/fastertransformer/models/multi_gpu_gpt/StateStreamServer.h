// Ensure GNU/POSIX extensions for Boost.Asio and socket/termios symbols
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE 1
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <vector>
#include <queue>
#include <mutex>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <linux/if.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

#include <grpc/grpc.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpc/support/log.h>

#include "src/fastertransformer/models/multi_gpu_gpt/StateStreamClient.h"

class DejaVuManagerImpl: public DejaVuManager::Service {

public:
	std::queue<int> avail_queue_;
	std::mutex      avail_mtx_;

	std::queue<int> written_queue_;
	std::mutex      written_mtx_;

	int prompt_buffer_size_;
	std::unique_ptr<Server> dv_server;

	Status Push(ServerContext* context, const PushRequest* request, PushResponse* response);
	Status Complete(ServerContext* context, const CompleteRequest* request, CompleteResponse* response);
};

void RunServer(DejaVuManagerImpl &service, int port, int num_microbatches, int prompt_buffer_size);
void Shutdown(DejaVuManagerImpl& service);
