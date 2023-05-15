#pragma once
namespace {
#define DIVUP(x, y) (((x) + (y)-1) / (y))
#define warp_size 32

//
struct Param {
    int batch, src_chl, src_h, src_w, flt_h, flt_w, out_h, out_w, pad_h, pad_w;};

template <int x_, int y_>
struct ThreadConfig {
    static int const thread_x = x_;
    static_assert((thread_x & (thread_x - 1)) == 0, "thread_x must be pow of 2!");
    static int const thread_y = y_;
    static int const nr_threads = x_ * y_;
    static int const warp_num = DIVUP(nr_threads, warp_size);
};

template <typename ThreadConfig_, int unroll_w_>
struct OgradTileConfig {
    using ThreadConfig = ThreadConfig_;
    static int const unroll_h = 1;
    static int const unroll_w = unroll_w_;
    static int const block_h = unroll_h * ThreadConfig::thread_y;
    static int const block_w = ThreadConfig::thread_x * unroll_w_;
    static int const row_size = unroll_h * block_w;
    
};

template <int fh_, int fw_>
struct FilterTileConfig {
    static int const unroll_h = 1;
    static int const unroll_w = fw_;
    static int const block_w = fw_;  //为了编译器优化 权宜之计 宽不小于9
    static int const block_h = fh_;  //为了编译器优化 权宜之计 高不小于7
    static int const unroll_size = unroll_h * unroll_w;
};

template <typename OgradTileConfig_, typename FilterTileConfig_>
struct SrcTileConfig {
    using OgradTileConfig = OgradTileConfig_;
    using FilterTileConfig = FilterTileConfig_;
    static int const unroll_h = (OgradTileConfig::unroll_h - 1) + FilterTileConfig::unroll_h;
    static int const unroll_w = (OgradTileConfig::unroll_w - 1) + FilterTileConfig::block_w;
    static int const block_w = (OgradTileConfig::block_w - 1) + FilterTileConfig::block_w;
    static int const block_h = (OgradTileConfig::block_h - 1) + FilterTileConfig::block_h;
    static int const unroll_size = unroll_h * unroll_w;
    static int const unroll_w_out = OgradTileConfig::unroll_w;
};

    
template <typename SrcTileConfig_>
struct SrcTileCount {
    using SrcTileConfig = SrcTileConfig_;
    static int const smem_h = SrcTileConfig::block_h;///
    static int const smem_w = SrcTileConfig::block_w;//保证宽为偶数，防止产生严重bank conflict
    // static int const a = (SrcTileConfig::unroll_w_out==8)? 0:1;
    static int const smem_w_bank = smem_w;
    static int const smem_size = smem_h * smem_w_bank;
};

template <typename OgradTileConfig_>
struct OgradTileCount {
    using OgradTileConfig = OgradTileConfig_;
    static int const smem_w = OgradTileConfig::block_w;//DIVUP(FilterTileConfig::block_w, 2) * 2;
    static int const smem_h = OgradTileConfig::block_h;//
    static int const smem_w_bank = smem_w+1;
    static int const smem_size = smem_h * smem_w_bank;
};

template <typename FilterTileConfig_, typename ThreadConfig_>
struct FilterTileCount {
    using FilterTileConfig = FilterTileConfig_;
    using ThreadConfig = ThreadConfig_;
    static int const smem_w = FilterTileConfig::block_w;//DIVUP(FilterTileConfig::block_w, 2) * 2;
    static int const smem_h = ThreadConfig::thread_y/(32/ThreadConfig::thread_x);//
    static int const smem_w_bank = smem_w;
    static int const smem_size = smem_h * smem_w_bank;
};

}  // namespace