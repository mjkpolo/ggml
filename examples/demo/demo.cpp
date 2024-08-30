#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const int dim1 = 4;
  const int dim2 = 4;
  const int dim3 = 4;

  ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
  fprintf(stderr, "%s: using CUDA backend\n", __func__);
  backend = ggml_backend_cuda_init(0);
  if (!backend) {
    fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
  }
#endif

  if (!backend) {
    backend = ggml_backend_cpu_init();
  }

  size_t ctx_size = 2 * ggml_tensor_overhead();

  struct ggml_init_params params {
    /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
  };
  struct ggml_context *ctx = ggml_init(params);

  struct ggml_tensor *tensor0 =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, 1, 1);
  struct ggml_tensor *tensor1 =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim2, dim3, 1, 1);

  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

  struct ggml_cgraph *gf = NULL;
  struct ggml_context *ctx_cgraph = NULL;
  {
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
            ggml_graph_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    ctx_cgraph = ggml_init(params0);
    gf = ggml_new_graph(ctx_cgraph);

    struct ggml_tensor *result0 = ggml_mul_mat(ctx_cgraph, tensor0, tensor1);
    ggml_build_forward_expand(gf, result0);
  }

  ggml_gallocr_t allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_alloc_graph(allocr, gf);

  int n_threads = 1;
  if (ggml_backend_is_cpu(backend)) {
    ggml_backend_cpu_set_n_threads(backend, n_threads);
  }
  // int64_t start_time = ggml_time_us();
  ggml_backend_graph_compute(backend, gf);
  // int64_t end_time = ggml_time_us();
  // double time_us = end_time - start_time;
  // printf("time us: %8.2f\n", time_us);

  ggml_free(ctx_cgraph);
  ggml_gallocr_free(allocr);
  ggml_free(ctx);
  ggml_backend_buffer_free(buffer);
  ggml_backend_free(backend);

  return 0;
}
