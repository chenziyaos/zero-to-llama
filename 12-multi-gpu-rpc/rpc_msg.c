// rpc_msg.c：照抄 ggml-rpc.cpp 里 #pragma pack(1) 的消息结构，打印每条消息在网线上占多少字节。
// 不连任何服务器，不开端口，只用编译器算 sizeof。
//
// 对照 llama.cpp 0ec4027：
//   ggml/src/ggml-rpc/ggml-rpc.cpp  34-272 行  rpc_tensor 和 rpc_msg_*（pack(1)）
//   ggml/src/ggml-rpc/ggml-rpc.cpp  375-410 行 请求 = cmd 1 B + size 8 B + data，回应 = size 8 B + data
//   ggml/src/ggml-rpc/ggml-rpc.cpp  1218 行    serialize_graph 的大小公式
//   ggml/src/ggml-rpc/transport.h   11 行      RPC_CONN_CAPS_SIZE = 24
//
// 用法：./rpc_msg [n_nodes n_tensors ...]   每对数字算一次 GRAPH_COMPUTE 的字节数
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GGML_MAX_DIMS 4
#define GGML_MAX_SRC 10
#define GGML_MAX_OP_PARAMS 64
#define GGML_MAX_NAME 64
#define RPC_CONN_CAPS_SIZE 24

#pragma pack(push, 1)
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];
    int32_t  use_count;
};
struct hello_req { uint8_t conn_caps[RPC_CONN_CAPS_SIZE]; };
struct hello_rsp { uint8_t major, minor, patch, padding; uint8_t conn_caps[RPC_CONN_CAPS_SIZE]; };
struct alloc_buffer_req { uint32_t device; uint64_t size; };
struct alloc_buffer_rsp { uint64_t remote_ptr, remote_size; };
struct get_alloc_size_req { uint32_t device; struct rpc_tensor tensor; struct rpc_tensor srcs[GGML_MAX_SRC]; };
struct init_tensor_req { struct rpc_tensor tensor; };
struct memset_tensor_req { struct rpc_tensor tensor; uint64_t offset, size; uint8_t value; };
struct set_tensor_hash_req { struct rpc_tensor tensor; uint64_t offset, hash; };
struct get_tensor_req { struct rpc_tensor tensor; uint64_t offset, size; };
struct get_tensor_2d_req { struct rpc_tensor tensor; uint64_t offset, size, n_copies, stride; };
struct copy_tensor_req { struct rpc_tensor src, dst; };
struct get_device_memory_rsp { uint64_t free_mem, total_mem; };
struct graph_recompute_req { uint32_t device; uint64_t uid; };
struct comm_init_req { uint32_t device, rank, world, port; char host[64]; };
struct comm_allreduce_req { uint32_t device; struct rpc_tensor tensor; };
#pragma pack(pop)

_Static_assert(sizeof(struct rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

#define HDR_REQ 9  // cmd 1 B + size 8 B
#define HDR_RSP 8  // size 8 B

// serialize_graph：device 4 + uid 8 + n_nodes 4 + nodes 8*n + n_tensors 4 + tensors 296*m
static uint64_t graph_bytes(uint64_t n_nodes, uint64_t n_tensors) {
    return 2 * 4 + 8 + n_nodes * 8 + 4 + n_tensors * sizeof(struct rpc_tensor);
}

// SET_TENSOR：rpc_tensor + cache_flag 1 + offset 8 + 数据
static uint64_t set_tensor_bytes(uint64_t size) { return sizeof(struct rpc_tensor) + 1 + 8 + size; }

int main(int argc, char ** argv) {
    printf("rpc_tensor            %4zu B\n", sizeof(struct rpc_tensor));
#define P(name, s) printf("%-21s %4zu B  （上线 %zu B）\n", name, sizeof(s), sizeof(s) + HDR_REQ)
    P("HELLO 请求", struct hello_req);
    printf("%-21s %4zu B\n", "HELLO 回应", sizeof(struct hello_rsp));
    P("ALLOC_BUFFER", struct alloc_buffer_req);
    P("GET_ALLOC_SIZE", struct get_alloc_size_req);
    P("INIT_TENSOR", struct init_tensor_req);
    P("MEMSET_TENSOR", struct memset_tensor_req);
    P("SET_TENSOR_HASH", struct set_tensor_hash_req);
    P("GET_TENSOR", struct get_tensor_req);
    P("GET_TENSOR_2D", struct get_tensor_2d_req);
    P("COPY_TENSOR", struct copy_tensor_req);
    printf("%-21s %4zu B\n", "GET_DEVICE_MEMORY 回应", sizeof(struct get_device_memory_rsp));
    P("GRAPH_RECOMPUTE", struct graph_recompute_req);
    P("COMM_INIT", struct comm_init_req);
    P("COMM_ALLREDUCE", struct comm_allreduce_req);
#undef P
    printf("SET_TENSOR 头部        %4llu B + 数据\n", (unsigned long long) set_tensor_bytes(0));
    // 10 MiB 以上的权重先发 SET_TENSOR_HASH：命中只花 312 B，未命中再整块发
    const uint64_t th = 10ull << 20;
    printf("10 MiB 权重：哈希命中 %zu B，未命中 %llu B\n", sizeof(struct set_tensor_hash_req) + HDR_REQ,
           (unsigned long long) (sizeof(struct set_tensor_hash_req) + HDR_REQ + set_tensor_bytes(th) + HDR_REQ));
    for (int i = 1; i + 1 < argc; i += 2) {
        uint64_t n = strtoull(argv[i], NULL, 10), m = strtoull(argv[i + 1], NULL, 10);
        uint64_t g = graph_bytes(n, m) + HDR_REQ;
        printf("GRAPH_COMPUTE n_nodes=%llu n_tensors=%llu：%llu B（%.1f KiB），重放 GRAPH_RECOMPUTE %zu B，%.0f 倍\n",
               (unsigned long long) n, (unsigned long long) m, (unsigned long long) g, g / 1024.0,
               sizeof(struct graph_recompute_req) + HDR_REQ, (double) g / (sizeof(struct graph_recompute_req) + HDR_REQ));
    }
    return 0;
}
