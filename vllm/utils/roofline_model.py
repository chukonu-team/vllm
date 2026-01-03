from typing import Tuple
from dataclasses import dataclass

def qwen2_vit_flops(
    pixel_values_shape: Tuple[int, int],
    grid_thw_list: list[list[int]],
    hidden_dim=1280,
    mlp_dim=5120,
    num_layers=32,
):
    """
    Compute forward FLOPs for Qwen2VisionTransformer (image/video).
    """

    N_total, patch_dim = pixel_values_shape
    D = hidden_dim
    Dm = mlp_dim

    # Sanity check
    assert sum(t*h*w for t,h,w in grid_thw_list) == N_total

    # ---------------- Patch Embedding ----------------
    flops_patch = 2 * N_total * patch_dim * D

    # ---------------- Linear parts (sum over all tokens) ----------------
    # QKV + Proj + MLP = 24 * N * D^2 per layer
    # QKV 投影：D → 3D； 2*N*D*3*D = 6ND^2 FLOPS
    # Attention 输出投影：D → D
    # MLP 第一层：D → Dm；MLP 第二层：Dm → D
    flops_linear = num_layers * (8 * N_total * D * D + 
                                 4 * N_total * D * Dm)

    # ---------------- Attention (per image) ----------------
    flops_attn = 0
    for (T, H, W) in grid_thw_list:
        Ni = T * H * W
        flops_attn += 4 * Ni * Ni * D
    flops_attn *= num_layers

    # ---------------- Patch Merger (per image) ----------------
    flops_merger = 0
    for (T, H, W) in grid_thw_list:
        Ni = T * H * W
        Nm = Ni // 4
        flops_merger += (
            2 * Nm * Dm * Dm +
            2 * Nm * Dm * 896
        )

    return flops_patch + flops_linear + flops_attn + flops_merger


def qwen2_vit_bytes(
    pixel_values_shape: Tuple[int, int],
    grid_thw_list,
    hidden_dim=1280,
    mlp_dim=5120,
    num_layers=32,
    bytes_per_elem=2,
    check_consistency=True,
):
    """
    Return total memory bytes (HBM traffic) used in Roofline AI calculation.

    We use a simple GEMM traffic model:
      Bytes ≈ (A_elems + B_elems + C_elems) * bytes_per_elem
    for each major matmul.

    Parameters
    ----------
    pixel_values_shape : tuple
        (num_tokens_total, patch_dim)
    grid_thw_list : list of (T, H, W)
        Each item corresponds to one image/video grid. Token count Ni = T*H*W.
    """

    N_total, patch_dim = pixel_values_shape
    D = hidden_dim
    Dm = mlp_dim
    s = bytes_per_elem

    # Sum tokens from grid_thw_list
    N_from_grid = sum(T * H * W for (T, H, W) in grid_thw_list)

    if check_consistency and N_from_grid != N_total:
        raise ValueError(
            f"Inconsistent token count: pixel_values has N_total={N_total}, "
            f"but grid_thw_list sums to {N_from_grid}."
        )

    total_bytes = 0

    # -------- Patch Embedding (Linear) --------
    # (N_total x patch_dim) @ (patch_dim x D) -> (N_total x D)
    total_bytes += (N_total * patch_dim + patch_dim * D + N_total * D) * s

    # -------- Transformer Blocks --------
    # Linear parts depend on total token count (all tokens participate)
    # QKV: (N_total x D) @ (D x 3D) -> (N_total x 3D)
    bytes_qkv = (N_total * D + D * (3 * D) + N_total * (3 * D)) * s

    # Proj: (N_total x D) @ (D x D) -> (N_total x D)
    bytes_proj = (N_total * D + D * D + N_total * D) * s

    # FC1: (N_total x D) @ (D x Dm) -> (N_total x Dm)
    bytes_fc1 = (N_total * D + D * Dm + N_total * Dm) * s

    # FC2: (N_total x Dm) @ (Dm x D) -> (N_total x D)
    bytes_fc2 = (N_total * Dm + Dm * D + N_total * D) * s

    # TODO: FlashAttention将QK^T与PV流水起来了，不需要过一遍显存
    # Attention parts must be computed per grid (per image/video segment),
    # because of the N^2 terms.
    bytes_attn_per_layer = 0
    for (T, H, W) in grid_thw_list:
        Ni = T * H * W

        # QK^T: (Ni x D) @ (D x Ni) -> (Ni x Ni)
        # A: Ni*D, B: D*Ni, C: Ni*Ni
        bytes_attn_per_layer += (Ni * D + D * Ni + Ni * Ni) * s

        # PV: (Ni x Ni) @ (Ni x D) -> (Ni x D)
        # A: Ni*Ni, B: Ni*D, C: Ni*D
        bytes_attn_per_layer += (Ni * Ni + Ni * D + Ni * D) * s

    bytes_block = bytes_qkv + bytes_proj + bytes_fc1 + bytes_fc2 + bytes_attn_per_layer
    total_bytes += num_layers * bytes_block

    # -------- Patch Merger (once) --------
    # Typically per grid: tokens reduced ~ /4 by 2x2 merge.
    bytes_merger = 0
    for (T, H, W) in grid_thw_list:
        Ni = T * H * W
        Nm = Ni // 4  # if your impl pads/ceils, adjust this accordingly

        # 5120 -> 5120 : (Nm x Dm) @ (Dm x Dm) -> (Nm x Dm)
        bytes_merger += (Nm * Dm + Dm * Dm + Nm * Dm) * s

        # 5120 -> 896 : (Nm x Dm) @ (Dm x 896) -> (Nm x 896)
        bytes_merger += (Nm * Dm + Dm * 896 + Nm * 896) * s

    total_bytes += bytes_merger

    return total_bytes

@dataclass
class RooflineAnalysis:
    pixel_values_shape: Tuple[int, int]
    grid_thw_list: list[list[int]]
    flops_per_call: int
    bytes_per_call: int
    arithmetic_intensity: float
    roofline_bound_flops: float
    regime: str # "memory-bound" or "compute-bound"

    def achieved_flops(self, call_duration_sec: float) -> float:
        return self.flops_per_call / call_duration_sec


def qwen2_vit_roofline(
    pixel_values_shape: Tuple[int, int],
    grid_thw_list: list[list[int]],
    peak_flops: float,
    peak_bw: float,
) -> RooflineAnalysis:
    flops = qwen2_vit_flops(pixel_values_shape, grid_thw_list)
    bytes_ = qwen2_vit_bytes(pixel_values_shape, grid_thw_list)
    AI = flops / bytes_
    roofline_bound = min(peak_flops, AI * peak_bw)
    if AI * peak_bw < peak_flops:
        regime = "memory-bound"
    else:
        regime = "compute-bound"

    return RooflineAnalysis(
        pixel_values_shape = pixel_values_shape,
        grid_thw_list = grid_thw_list,
        flops_per_call = flops,
        bytes_per_call = bytes_,
        arithmetic_intensity = AI,
        roofline_bound_flops = roofline_bound,
        regime = regime,
    )


if __name__ == "__main__":
    pixel_values_shape = (5476, 1176)
    grid_thw_list = [[1, 74, 74]]
    duration_ms = 285.3724060058594

    # A10G specs (bf16)
    peak_flops = 70e12      # 70 TFLOPs
    peak_bw = 600e9         # 600 GB/s

    result = qwen2_vit_roofline(
        pixel_values_shape=pixel_values_shape,
        grid_thw_list=grid_thw_list,
        peak_flops=peak_flops,
        peak_bw=peak_bw,
    )

    achieved_flops = result.achieved_flops(duration_ms * 1e-3)

    print("===== Roofline Analysis =====")
    print(f"Tokens           : {pixel_values_shape[0]}")
    print(f"FLOPs            : {result.flops_per_call/1e12:.2f} TF")
    print(f"Bytes (HBM)      : {result.bytes_per_call/1e9:.2f} GB")
    print(f"AI               : {result.arithmetic_intensity:.1f} FLOPs/byte")
    print()
    print(f"Peak Compute     : {peak_flops/1e12:.1f} TF/s")
    print(f"Peak Bandwidth   : {peak_bw/1e9:.1f} GB/s")
    print(f"Roofline Bound   : {result.roofline_bound_flops/1e12:.1f} TF/s")
    print(f"Achieved         : {achieved_flops/1e12:.1f} TF/s")
    print()
    print(f"Regime           : {result.regime}")