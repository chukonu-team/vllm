import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from mineru.backend.vlm.vlm_analyze import ModelSingleton
mineru_model = ModelSingleton().get_model("vllm-engine", None, None)
llm = mineru_model.client.vllm_llm

# 最顶层模型被vllm.compilation.cuda_graph.CUDAGraphWrapper盖住了
llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

# vllm.model_executor.models.qwen2_vl.Qwen2VLForConditionalGeneration
llm_model_inner = llm_model.runnable
type(llm_model_inner)

# 可见没有被CUDA Graph盖住
# <class 'vllm.model_executor.models.qwen2_vl.Qwen2VisionTransformer'>
vit_model = llm_model_inner.visual
type(vit_model)
assert(vit_model.training == False)

# 编译通过@support_torch_compile触发
pixel_values_shape = (5476, 1176)
grid_thw_list = [[1, 74, 74]]

# 执行一次inference
with torch.inference_mode():
    pixel_values = torch.randn(5476, 1176, device='cuda')
    image_embeds = vit_model(pixel_values, grid_thw=grid_thw_list)
    
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

# ---------- warmup ----------
with torch.inference_mode():
    for _ in range(20):
        _ = vit_model(pixel_values, grid_thw=grid_thw_list)       
torch.cuda.synchronize()

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

times = np.array(times)

print(f"mean   : {times.mean():.3f} ms")
print(f"p50    : {np.percentile(times, 50):.3f} ms")
print(f"p90    : {np.percentile(times, 90):.3f} ms")
print(f"p99    : {np.percentile(times, 99):.3f} ms")
