#pragma once

/*
 * [窗口优化-公共 partition 计算]
 * 这个头文件只包含与张量布局、数据类型无关的整数计算，供 host 端
 * dispatcher、device kernel 以及 combine kernel 共用。窗口 mask 仍然在
 * 各自 kernel 中单独计算；这里只负责活动 KV page 和 partition 的范围。
 */

struct PagedAttentionActiveRange {
  int first_window_token;
  int first_active_block;
  int num_seq_blocks;
  int blocks_per_partition;
  int num_active_blocks;
  int num_active_partitions;
  int start_block_idx;
  int end_block_idx;
};

__host__ __device__ inline int paged_attention_window_min(
    int a, int b) {
  return a < b ? a : b;
}

__host__ __device__ inline int paged_attention_window_max(
    int a, int b) {
  return a > b ? a : b;
}

__host__ __device__ inline int paged_attention_window_divide_round_up(
    int a, int b) {
  return (a + b - 1) / b;
}

/*
 * [窗口优化-统一 active range]
 * partition_idx 是压缩后的 local partition 编号。函数先找到窗口覆盖的
 * 第一个 block，再从这个 block 开始重新编号 partition，避免使用原始
 * partition 编号倒推活动范围时漏掉未对齐的边界 block。
 */
__host__ __device__ inline PagedAttentionActiveRange
get_paged_attention_active_range(
    int seq_len,
    int mtp,
    int partition_window_size_left,
    int block_size,
    int partition_size,
    int partition_idx) {
  PagedAttentionActiveRange range;
  const int first_query_pos = seq_len - mtp;

  range.first_window_token = partition_window_size_left < 0
      ? 0
      : paged_attention_window_max(
            0,
            first_query_pos - partition_window_size_left);
  range.first_active_block =
      range.first_window_token / block_size;
  range.num_seq_blocks =
      paged_attention_window_divide_round_up(seq_len, block_size);
  range.blocks_per_partition = partition_size / block_size;
  range.num_active_blocks = paged_attention_window_max(
      0,
      range.num_seq_blocks - range.first_active_block);
  range.num_active_partitions =
      paged_attention_window_divide_round_up(
          range.num_active_blocks,
          range.blocks_per_partition);
  range.start_block_idx =
      range.first_active_block
      + partition_idx * range.blocks_per_partition;
  range.end_block_idx = paged_attention_window_min(
      range.start_block_idx + range.blocks_per_partition,
      range.num_seq_blocks);
  return range;
}

/*
 * [窗口优化-host partition 上界]
 * max_num_partitions 是所有 batch sequence 共用的 grid.z 和临时缓冲区上界，
 * 不是某一条 sequence 的精确 active partition 数。窗口从非对齐 block 开始
 * 时，left + mtp + block_size - 1 覆盖了向前取整产生的边界 block。
 */
inline int get_paged_attention_max_num_partitions(
    int max_seq_len,
    int mtp,
    int partition_window_size_left,
    int block_size,
    int partition_size,
    bool use_compaction) {
  const int max_active_tokens =
      use_compaction && partition_window_size_left >= 0
          ? paged_attention_window_min(
                max_seq_len,
                partition_window_size_left
                    + mtp
                    + block_size
                    - 1)
          : max_seq_len;
  return paged_attention_window_max(
      1,
      paged_attention_window_divide_round_up(
          max_active_tokens,
          partition_size));
}
