#pragma once
#include "para.cuh"
#include <iostream>

namespace {

template <typename ThreadConfig_, typename TileCount_>
struct Global2SharedMem {
    using TileCount = TileCount_;
    using ThreadConfig = ThreadConfig_;
    const int tid = threadIdx.y * ThreadConfig::thread_x + threadIdx.x;

    float* smem;
    const float* g_ptr;
    int stride;
    int start_h, start_w, bound_h, bound_w;
    

    __device__ __forceinline__ Global2SharedMem(
            float* smem_, int stride_, int s_h, int s_w, int b_h, int b_w);
    __device__ __forceinline__ void copy();
    __device__ __forceinline__ float* sh_ptr_as_copy_t(int y, int x) {
        return smem + y * TileCount::smem_w_bank + x;
    }
};

template <typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__
Global2SharedMem<ThreadConfig_, TileCount_>::Global2SharedMem(
        float* smem_, int stride_, int s_h, int s_w, int b_h, int b_w)
        : smem(smem_),
          stride(stride_),
          start_h(s_h),
          start_w(s_w),
          bound_h(b_h),
          bound_w(b_w) {}

template <typename ThreadConfig_, typename TileCount_>
__device__ __forceinline__ void Global2SharedMem<ThreadConfig_, TileCount_>::copy() {
    static int const load_w = TileCount::smem_w > 32 ? 32 : TileCount::smem_w;
    static int const load_h = ThreadConfig::nr_threads / load_w;
    static int const num_free_thread = ThreadConfig::nr_threads - load_w*load_h;
    static int const h_per_thread = DIVUP(TileCount::smem_h, load_h);
    static int const w_per_thread = DIVUP(TileCount::smem_w, load_w);
    static bool constexpr check_bounds_h = (TileCount::smem_h % load_h != 0)||(num_free_thread != 0);
    static bool constexpr check_bounds_w = TileCount::smem_w % load_w != 0;
    const int y_base_idx = tid / load_w;
    const int x_base_idx = tid - y_base_idx * load_w;
#pragma unroll
    for (int i = 0; i < h_per_thread; ++i) {
        int smem_h_idx = y_base_idx + i * load_h;
        int src_h_idx;
        src_h_idx = start_h + smem_h_idx;
        if (check_bounds_h && smem_h_idx >= TileCount::smem_h)
            break;
#pragma unroll
        for (int j = 0; j < w_per_thread; ++j) {
            int smem_w_idx = x_base_idx + j * load_w;
            int src_w_idx;
            src_w_idx = start_w + smem_w_idx;
            if (check_bounds_w && smem_w_idx >= TileCount::smem_w)
                break;
            float val = 0.0f;
            if (src_h_idx >= 0 && src_h_idx < bound_h && src_w_idx >= 0 &&
                src_w_idx < bound_w) {
                val = g_ptr[src_h_idx * stride + src_w_idx];
            }
            *(sh_ptr_as_copy_t(smem_h_idx, smem_w_idx)) = val;
        }
    }
}


template <typename ThreadConfig_, typename FilterTileConfig_, typename OgradTileConfig_, typename SrcTileConfig_, typename SrcTileCount_, typename OgradTileCount_, typename FilterTileCount_>
__global__ void dwconv2dkernelfp32wgrad(
        const Param param, const float* input, const float* ograd, float* fgrad) {

    using ThreadConfig = ThreadConfig_;
    using SrcTileConfig = SrcTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    using OgradTileConfig = OgradTileConfig_;
    using SrcTileCount = SrcTileCount_;
    using OgradTileCount = OgradTileCount_;
    using FilterTileCount = FilterTileCount_;
    

    using SrcGlobal2ShareVisitor = Global2SharedMem<ThreadConfig, SrcTileCount>;
    using OgradGlobal2ShareVisitor = Global2SharedMem<ThreadConfig, OgradTileCount>;

    int off_batch_channel = blockIdx.x, off_obw = blockIdx.y, off_obh = blockIdx.z,
        off_oh = threadIdx.y, off_ow = threadIdx.x;
    
    int tid = threadIdx.y*ThreadConfig::thread_x + threadIdx.x;
    int warp_id = tid / 32;
    int id_in_warp = tid % 32;

    extern __shared__ __align__(8) unsigned char smem[];
    static_assert(sizeof(float) <= 8, "Insufficient alignment detected");
    float* smem_src = reinterpret_cast<float*>(smem);
    float* smem_ograd = reinterpret_cast<float*>(&smem_src[SrcTileCount::smem_size]);
    float* smem_fgrad = reinterpret_cast<float*>(&smem_src[SrcTileCount::smem_size + OgradTileCount::smem_size]);
    // smem_ograd = reinterpret_cast<T*>(smem);

    int off_channel = off_batch_channel % param.src_chl,
        ograd_start_h = off_obh * OgradTileConfig::block_h,
        ograd_start_w = off_obw * OgradTileConfig::block_w,
        src_start_h = ograd_start_h - param.pad_h,
        src_start_w = ograd_start_w - param.pad_w,
        out_base_h_idx = ograd_start_h + off_oh * OgradTileConfig::unroll_h;

    

    SrcGlobal2ShareVisitor gl2sh_src = {
            smem_src,
            static_cast<int>(param.src_w),
            static_cast<int>(src_start_h),
            static_cast<int>(src_start_w),
            static_cast<int>(param.src_h),
            static_cast<int>(param.src_w)};

    OgradGlobal2ShareVisitor gl2sh_ograd = {
            smem_ograd,
            static_cast<int>(param.out_w),
            static_cast<int>(ograd_start_h),
            static_cast<int>(ograd_start_w),
            static_cast<int>(param.out_h),
            static_cast<int>(param.out_w)};

    gl2sh_src.g_ptr = input + off_batch_channel * param.src_h * param.src_w;
    gl2sh_ograd.g_ptr = ograd + off_batch_channel * param.out_h * param.out_w;
    float* f_grad_idx = fgrad + off_channel*param.flt_h*param.flt_w;

    gl2sh_src.copy();
    gl2sh_ograd.copy();

    __syncthreads();
    // if (out_base_h_idx >= param.out_h) {
    //     return ;
    // }

    float* smem_src_ptr = smem_src + off_ow * OgradTileConfig::unroll_w;
    float* smem_ograd_ptr = smem_ograd + off_oh * OgradTileCount::smem_w_bank + off_ow * OgradTileConfig::unroll_w;

    float reg_src[SrcTileConfig::unroll_h * SrcTileConfig::unroll_w];
    float reg_ograd[OgradTileConfig::unroll_h * OgradTileConfig::unroll_w];
    float sum[FilterTileConfig::unroll_size] = {0.0};

    for (int s_w = 0; s_w < OgradTileConfig::unroll_w; ++s_w) {
            reg_ograd[s_w] = smem_ograd_ptr[s_w];
        }

    for (int fh = 0; fh < param.flt_h; fh += 1) {

        #pragma unroll
        for (int s_w = 0; s_w < SrcTileConfig::unroll_w; ++s_w) { //加载输入
            reg_src[s_w] = smem_src_ptr[(off_oh + fh) * SrcTileCount::smem_w_bank + s_w];
        }

        #pragma unroll
        for (int fw = 0; fw < FilterTileConfig::block_w; ++fw) {  //卷积
            #pragma unroll
            for (int ow = 0; ow < OgradTileConfig::unroll_w; ++ow) {
                sum[fw] += reg_ograd[ow] * reg_src[ow + fw];
            }

            #pragma unroll
            for (int i = 1; i < warp_size; i = i << 1) {
            sum[fw] += __shfl_xor_sync(0xffffffff, sum[fw], i, warp_size);
            }
            if (id_in_warp==0)
            {
                smem_fgrad[warp_id*FilterTileCount::smem_w_bank + fw] = sum[fw];
            }
            sum[fw] = 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int activeWarps = ThreadConfig::warp_num / 2; activeWarps > 0; activeWarps /= 2) //warp间线程归并
        {
            if ((warp_id < activeWarps)&&(id_in_warp<FilterTileConfig::unroll_size))
            {
                smem_fgrad[warp_id*FilterTileCount::smem_w_bank + id_in_warp] += smem_fgrad[(warp_id+activeWarps)*FilterTileCount::smem_w_bank + id_in_warp];
            }
            __syncthreads();
        }

        int f_grad_idx_h_off = fh*param.flt_w;
        if ((warp_id==0)&&(id_in_warp<param.flt_w))
        {
            atomicAdd(&(f_grad_idx[f_grad_idx_h_off + id_in_warp]), smem_fgrad[id_in_warp]);
        }
    }
}

template <int fh, int fw, int thread_x>
void Dwconv2dKernelFP32Wgrad(const Param& param, const float* input, const float* ograd, float* fgrad) {

    
    using ThreadConfig = ThreadConfig<thread_x, 32>;
    using FilterTileConfig = FilterTileConfig<fh, fw>;
    using OgradTileConfig = OgradTileConfig<ThreadConfig, 16>;
    using SrcTileConfig = SrcTileConfig<OgradTileConfig, FilterTileConfig>;
    using SrcTileCount = SrcTileCount<SrcTileConfig>;
    using OgradTileCount = OgradTileCount<OgradTileConfig>;
    using FilterTileCount = FilterTileCount<FilterTileConfig, ThreadConfig>;

    dim3 block(ThreadConfig::thread_x, ThreadConfig::thread_y);
    dim3 grid;

    // std::cout<<SrcTileCount::smem_w<<std::endl;
    // std::cout<<SrcTileCount::smem_w_bank<<std::endl;
    // std::cout<<OgradTileConfig::unroll_w<<std::endl;
    
    grid.x = param.batch * param.src_chl;
    grid.y = DIVUP(param.out_w, OgradTileConfig::block_w);
    grid.z = DIVUP(param.out_h, OgradTileConfig::block_h);
    const int shared_storage = (SrcTileCount::smem_size + OgradTileCount::smem_size + FilterTileCount::smem_size) * sizeof(float);

    void (*kernel)(const Param, const float*, const float*, float*);
    kernel = dwconv2dkernelfp32wgrad<ThreadConfig, FilterTileConfig, OgradTileConfig, SrcTileConfig, SrcTileCount, OgradTileCount, FilterTileCount>;
    kernel<<<grid, block, shared_storage>>>(param, input, ograd, fgrad);
}


#define INSTANCE_ABC(a, b, c)\
    if (param.out_w <= 16||c==2) {\
        Dwconv2dKernelFP32Wgrad<a, b, c>(param, src_, ograd_, dst_);\
    }


#define INSTANCE_AB(a, b)\
if (param.flt_w > b-2) {\
    INSTANCE_ABC(a, b, 1)\
    else INSTANCE_ABC(a, b, 2)\
}

#define INSTANCE_A(a)\
if (param.flt_h > a-2) {\
    INSTANCE_AB(a, 31) \
    else INSTANCE_AB(a, 29) \
    else INSTANCE_AB(a, 27) \
    else INSTANCE_AB(a, 25) \
    else INSTANCE_AB(a, 23) \
    else INSTANCE_AB(a, 21) \
    else INSTANCE_AB(a, 19) \
    else INSTANCE_AB(a, 17) \
    else INSTANCE_AB(a, 15) \
    else INSTANCE_AB(a, 13) \
    else INSTANCE_AB(a, 11) \
    else INSTANCE_AB(a, 9) \
    else INSTANCE_AB(a, 7) \
    else INSTANCE_AB(a, 5) \
    else INSTANCE_AB(a, 3) \
}


// #define INSTANCE_A(a)\
// if (param.flt_w > a-2) {\
//     INSTANCE_ABC(a, a, 1)\
//     else INSTANCE_ABC(a, a, 2)\
// }

#define INSTANCE \
    INSTANCE_A(31) \
    else INSTANCE_A(29) \
    else INSTANCE_A(27) \
    else INSTANCE_A(25) \
    else INSTANCE_A(23) \
    else INSTANCE_A(21) \
    else INSTANCE_A(19) \
    else INSTANCE_A(17) \
    else INSTANCE_A(15) \
    else INSTANCE_A(13) \
    else INSTANCE_A(11) \
    else INSTANCE_A(9) \
    else INSTANCE_A(7) \
    else INSTANCE_A(5) \
    else INSTANCE_A(3)



}  // anonymous namespace