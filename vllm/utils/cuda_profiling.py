import torch

from dataclasses import dataclass
from typing import Optional

class CudaProfilingContext:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.before_mm_encode = torch.cuda.Event(enable_timing=True)
        self.after_mm_encode = torch.cuda.Event(enable_timing=True)
        self.before_model_forward = torch.cuda.Event(enable_timing=True)
        self.after_model_forward = torch.cuda.Event(enable_timing=True)
        self.after_postprocess = torch.cuda.Event(enable_timing=True)
        self.gpu_model_runner_statistics: Optional[GpuModelRunnerStatistics] = None

        # 多模态相关
        self.mm_encoder_statistics: Optional[MMEncoderStatistics] = None

@dataclass
class MMEncoderStatistics:
    request_ids: list[str]
    per_request_num_images: list[int]
    input_shapes: list[list[int]]
    time_ms: float

@dataclass
class CudaGraphKey:
    len: int
    is_uniform_decode: bool
    has_lora: bool

@dataclass
class GpuModelRunnerStatistics:
    total_num_scheduled_tokens: int
    num_active_requests: int
    cudagraph_runtime_mode: str
    cudagraph_key: CudaGraphKey | None
    scheduled_cached_reqs: int
    scheduled_new_reqs: int
    running_request_ids: list[str]
    num_scheduled_tokens: list[int]
    num_computed_tokens: list[int]
    active_num_pages: int

@dataclass
class StepStatistics:
    runner_stats: GpuModelRunnerStatistics
    mm_encoder_statistics: Optional[MMEncoderStatistics]
    step_time_ms: float
    schedule_time_ms: float
    model_preprocess_time_ms: float
    model_forward_time_ms: float
    model_postprocess_time_ms: float
