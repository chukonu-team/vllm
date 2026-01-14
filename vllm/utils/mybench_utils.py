import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

import cloudpickle
import torch.nn as nn
from pydantic import ValidationError
from tqdm.auto import tqdm
from typing_extensions import TypeVar

from vllm.beam_search import (BeamSearchInstance, BeamSearchOutput,
                              BeamSearchSequence,
                              create_sort_beams_key_function)
from vllm.config import (CompilationConfig, ModelDType,
                         StructuredOutputsConfig, TokenizerMode, is_init_field)
from vllm.config.compilation import CUDAGraphMode
from vllm.engine.arg_utils import (ConvertOption, EngineArgs, HfOverrides,
                                   PoolerConfig, RunnerOption)
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         ChatTemplateContentFormatOption,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages,
                                         resolve_chat_template_content_format)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.score_utils import (ScoreContentPartParam,
                                          ScoreMultiModalParam,
                                          _cosine_similarity,
                                          _validate_score_input_lens,
                                          compress_token_type_ids,
                                          get_score_prompt)
# yapf: enable
from vllm.entrypoints.utils import (_validate_truncation_size,
                                    log_non_default_args)
from vllm.inputs import (DataPrompt, PromptType, SingletonPrompt, TextPrompt,
                         TokensPrompt)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.outputs import (ClassificationRequestOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput, RequestOutput,
                          ScoringRequestOutput)
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import (BeamSearchParams, RequestOutputKind,
                                  SamplingParams)
from vllm.tasks import PoolingTask
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, Device, as_iter, is_list_of
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.sample.logits_processor import LogitsProcessor
import time

# 创建一个和传入LLM等同的engine_args
def create_engine_args(
    model: str,
    *,
    runner: RunnerOption = "auto",
    convert: ConvertOption = "auto",
    tokenizer: Optional[str] = None,
    tokenizer_mode: TokenizerMode = "auto",
    skip_tokenizer_init: bool = False,
    trust_remote_code: bool = False,
    allowed_local_media_path: str = "",
    allowed_media_domains: Optional[list[str]] = None,
    tensor_parallel_size: int = 1,
    dtype: ModelDType = "auto",
    quantization: Optional[QuantizationMethods] = None,
    revision: Optional[str] = None,
    tokenizer_revision: Optional[str] = None,
    seed: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    swap_space: float = 4,
    cpu_offload_gb: float = 0,
    enforce_eager: bool = False,
    disable_custom_all_reduce: bool = False,
    hf_token: Optional[Union[bool, str]] = None,
    hf_overrides: Optional[HfOverrides] = None,
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
    pooler_config: Optional[PoolerConfig] = None,
    override_pooler_config: Optional[PoolerConfig] = None,
    structured_outputs_config: Optional[Union[dict[
        str, Any], StructuredOutputsConfig]] = None,
    kv_cache_memory_bytes: Optional[int] = None,
    compilation_config: Optional[Union[int, dict[str, Any],
                                        CompilationConfig]] = None,
    logits_processors: Optional[list[Union[str,
                                            type[LogitsProcessor]]]] = None,
    **kwargs: Any,
) -> None:
    """LLM constructor."""

    if "disable_log_stats" not in kwargs:
        kwargs["disable_log_stats"] = True

    if "worker_cls" in kwargs:
        worker_cls = kwargs["worker_cls"]
        # if the worker_cls is not qualified string name,
        # we serialize it using cloudpickle to avoid pickling issues
        if isinstance(worker_cls, type):
            kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

    if "kv_transfer_config" in kwargs and isinstance(
            kwargs["kv_transfer_config"], dict):
        from vllm.config.kv_transfer import KVTransferConfig
        raw_config_dict = kwargs["kv_transfer_config"]
        try:
            kwargs["kv_transfer_config"] = KVTransferConfig(
                **raw_config_dict)
        except ValidationError as e:
            raise ValueError(
                f"Invalid 'kv_transfer_config' provided: {e}") from e

    if hf_overrides is None:
        hf_overrides = {}

    if compilation_config is not None:
        if isinstance(compilation_config, int):
            compilation_config_instance = CompilationConfig(
                level=compilation_config)
        elif isinstance(compilation_config, dict):
            compilation_config_instance = CompilationConfig(
                **{
                    k: v
                    for k, v in compilation_config.items()
                    if is_init_field(CompilationConfig, k)
                })
        else:
            compilation_config_instance = compilation_config
    else:
        compilation_config_instance = CompilationConfig()

    if structured_outputs_config is not None:
        if isinstance(structured_outputs_config, dict):
            structured_outputs_instance = StructuredOutputsConfig(
                **{
                    k: v
                    for k, v in structured_outputs_config.items()
                    if is_init_field(StructuredOutputsConfig, k)
                })
        else:
            structured_outputs_instance = structured_outputs_config
    else:
        structured_outputs_instance = StructuredOutputsConfig()

    engine_args = EngineArgs(
        model=model,
        runner=runner,
        convert=convert,
        tokenizer=tokenizer,
        tokenizer_mode=tokenizer_mode,
        skip_tokenizer_init=skip_tokenizer_init,
        trust_remote_code=trust_remote_code,
        allowed_local_media_path=allowed_local_media_path,
        allowed_media_domains=allowed_media_domains,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        tokenizer_revision=tokenizer_revision,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        swap_space=swap_space,
        cpu_offload_gb=cpu_offload_gb,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=disable_custom_all_reduce,
        hf_token=hf_token,
        hf_overrides=hf_overrides,
        mm_processor_kwargs=mm_processor_kwargs,
        pooler_config=pooler_config,
        override_pooler_config=override_pooler_config,
        structured_outputs_config=structured_outputs_instance,
        compilation_config=compilation_config_instance,
        logits_processors=logits_processors,
        **kwargs,
    )

    log_non_default_args(engine_args)

    return engine_args


from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru_vl_utils import MinerULogitsProcessor
from vllm.utils.mybench_utils import create_engine_args
from vllm.usage.usage_lib import UsageContext
from vllm.model_executor.models.qwen2_vl import Qwen2VisionTransformer
from vllm.v1.worker.gpu_worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port, run_method
from vllm.config import set_current_vllm_config, VllmConfig

def initialize_fake_worker():
    # /home/ubuntu/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/snapshots/879e58bdd9566632b27a8a81f0e2961873311f67
    model_path = auto_download_and_get_model_root_path("/","vlm")
    device = "cuda"
    kwargs = {'gpu_memory_utilization': 0.5, 'model': model_path}
    kwargs["logits_processors"] = [MinerULogitsProcessor]
    kwargs["compilation_config"] = {"cudagraph_mode": CUDAGraphMode.NONE}
    engine_args = create_engine_args(**kwargs)
    vllm_config: VllmConfig = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    compilation_config: CompilationConfig = vllm_config.compilation_config
    compilation_config.splitting_ops.append("xformers_flash3.flash_fwd")
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    worker_kwargs = dict(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=True,
    )
    with set_current_vllm_config(vllm_config):
        worker = Worker(**worker_kwargs)
        worker.init_device()    
        worker.load_model()
    return worker, vllm_config
