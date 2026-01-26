import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import torch
#from mineru.backend.vlm.vlm_analyze import ModelSingleton
#mineru_model = ModelSingleton().get_model("vllm-engine", None, None)
#llm = mineru_model.client.vllm_llm

from vllm.utils.mybench_utils import initialize_fake_worker
from vllm.compilation.backends import ToyVllmBackend

# 解析命令行参数
parser = argparse.ArgumentParser(description='Benchmark Qwen2 Vision Model')
parser.add_argument('--enable_compilation', action='store_true', default=False,
                    help='是否开启编译 (默认: False)')
parser.add_argument('--disable_compilation_cudagraph', action='store_true', default=False,
                    help='如果开启编译，禁用CUDA Graph (默认启用CUDA Graph)')
parser.add_argument('--profile_mode', type=str, default=None, choices=['ncu'],
                    help='是否开启profile（以及profile模式）。可选值: ncu (默认: None)')

args = parser.parse_args()

# 从命令行参数获取配置
enable_compilation = args.enable_compilation
enable_compilation_with_cudagraph = not args.disable_compilation_cudagraph
profile_mode = args.profile_mode

# 是否开启cuda graph
use_cudagraph = False

print(f"enable_compilation = {enable_compilation}")

if enable_compilation:
    print(f"enable_compilation_with_cudagraph = {enable_compilation_with_cudagraph}")

print(f"profile_mode = {profile_mode}")

driver_worker, vllm_config = initialize_fake_worker()

# 最顶层模型被vllm.compilation.cuda_graph.CUDAGraphWrapper盖住了
llm_model = driver_worker.model_runner.model

# vllm.model_executor.models.qwen2_vl.Qwen2VLForConditionalGeneration
if use_cudagraph:
    llm_model_inner = llm_model.runnable
else:
    llm_model_inner = llm_model

type(llm_model_inner)

# 可见没有被CUDA Graph盖住
# <class 'vllm.model_executor.models.qwen2_vl.Qwen2VisionTransformer'>
vit_model = llm_model_inner.visual
type(vit_model)
assert(vit_model.training == False)

# 编译vit_model.forward方法
if enable_compilation:
    # vllm.compilation.backends.VllmBackend
    backend = ToyVllmBackend(vllm_config)
    vit_model.forward_compiled = torch.compile(vit_model.forward_compiled, fullgraph=False, backend=backend, options=None)
    # original_code_object = vit_model.__class__.forward.__code__

# 编译通过@support_torch_compile触发
pixel_values_shape = (5476, 1176)
grid_thw_list = [[1, 74, 74]]

print("首次执行，可能引入编译...")

# 执行一次inference
with torch.inference_mode():
    pixel_values = torch.randn(5476, 1176, device='cuda')
    image_embeds = vit_model(pixel_values, grid_thw=grid_thw_list)

print("首次执行完成")

from vllm.compilation.cuda_graph import enable_toy_cuda_graph

if enable_compilation_with_cudagraph:
    enable_toy_cuda_graph()

# 执行一个微观测试程序

import torch
import time
import numpy as np

#vit_model.eval()
assert(vit_model.training == False)

# vit_model.to("cuda")
assert(all(p.device.type == "cuda" for p in vit_model.parameters()))

pixel_values = torch.randn(5476, 1176, device="cuda")
grid_thw_list = [[1, 74, 74]]

print("预热中...")

# ---------- warmup ----------
with torch.inference_mode():
    for _ in range(20):
        _ = vit_model(pixel_values, grid_thw=grid_thw_list)       
torch.cuda.synchronize()

print("预热完成")

if profile_mode == "ncu":
    # 对于NCU，采集过程中只需把模型跑一遍即可

    print("执行中...")
    torch.cuda.profiler.start()
    with torch.inference_mode():
        # 执行1遍已经比较长时间了
        for i in range(1):
            print(f"执行第{i}遍...")
            _ = vit_model(pixel_values, grid_thw=grid_thw_list)       
            torch.cuda.synchronize()
    torch.cuda.profiler.stop()

else:
    torch.cuda.profiler.start()

    # ---------- benchmark ----------
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for _ in range(100):
            torch.cuda.synchronize()
            start.record()
            _ = vit_model(pixel_values, grid_thw=grid_thw_list)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms

    torch.cuda.profiler.stop()

    times = np.array(times)

    print(f"mean   : {times.mean():.3f} ms")
    print(f"p50    : {np.percentile(times, 50):.3f} ms")
    print(f"p90    : {np.percentile(times, 90):.3f} ms")
    print(f"p99    : {np.percentile(times, 99):.3f} ms")
