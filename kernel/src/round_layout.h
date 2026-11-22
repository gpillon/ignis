// ignis kernel leaf - GitHub #712 (spec #711): a DFlash2 verify round's layout, each lane's place in
// the round's buffers -- OURS. Pure host-side arithmetic: no CUDA and no model.
//
// A round's lanes are rows of its per-lane buffers (anchors, extents, slots, sampling configs, the
// accept's outputs), in two groups at one round shape each: the extended group (group 0), the lanes
// at the load's extended shape, then the base group (group 1), the lanes at its base shape. A lane
// owns `window + 1` consecutive columns of every flat column buffer (verify ids, positions, logits,
// hidden, feature taps, licensed tokens, rotation positions) from its column base, and `window`
// drafts of the drafts buffer from its draft base: the extended group's lanes from column 0 and
// draft 0, the base group's after them. A uniform round, every lane at one shape, is one group and
// an empty one, and lays its lanes out as a round of one shape always has: lane i's columns from
// i * (window + 1), its drafts from i * window.
//
// GitHub #713: a mixed round holds both groups, `w` extended lanes and `n` base lanes, the base
// group's columns from (extended window + 1) * w and its drafts from extended window * w. Its GDN
// layers record each group in its own shape's ReplaySSM records, over the one backing both shapes
// share: extended lane r at row r of the extended layout, base lane s at row 2w + s of a base layout
// laid at the extended one's stride, so that every layer's base rows lie past its extended rows, and
// the base group's fold runs behind 2w leading rows that commit no column. Record rows are not
// columns: where the rows of a layer are not the round's columns there (any window but 15), or no
// base layout has the extended one's stride, the round is refused, never padded.
#pragma once

#include <array>
#include <cstdint>

namespace ignis::round_layout {

// One group of a round's lanes, all at one round shape.
struct Group {
  std::uint32_t first_lane = 0;   // its first row in the round's per-lane buffers
  std::uint32_t lanes = 0;
  std::uint32_t window = 0;       // drafts a lane: its shape's window
  std::uint32_t first_column = 0; // its first lane's anchor column
  std::uint32_t first_draft = 0;  // its first lane's first draft
  bool extended = false;          // at the load's extended shape
  // GitHub #713: its first lane's row in its shape's ReplaySSM records (above).
  std::uint32_t first_record_row = 0;
  std::uint32_t lane_columns() const { return window + 1; }
  std::uint32_t columns() const { return lane_columns() * lanes; }
  std::uint32_t drafts() const { return window * lanes; }
};

// A verify round's layout: its extended group, then its base group.
struct Layout {
  std::array<Group, 2> groups{};

  std::uint32_t lanes() const { return groups[0].lanes + groups[1].lanes; }
  std::uint32_t columns() const { return groups[0].columns() + groups[1].columns(); }
  std::uint32_t drafts() const { return groups[0].drafts() + groups[1].drafts(); }
  // Every lane at one shape: one group holds them all.
  bool uniform() const { return groups[0].lanes == 0 || groups[1].lanes == 0; }
  // The group of row `lane`, and that lane's window, shape, column base and draft base.
  const Group &group_of(std::uint32_t lane) const { return lane < groups[0].lanes ? groups[0] : groups[1]; }
  std::uint32_t window_of(std::uint32_t lane) const { return group_of(lane).window; }
  bool extended(std::uint32_t lane) const { return group_of(lane).extended; }
  std::uint32_t column_of(std::uint32_t lane) const {
    const Group &group = group_of(lane);
    return group.first_column + (lane - group.first_lane) * group.lane_columns();
  }
  std::uint32_t draft_of(std::uint32_t lane) const {
    const Group &group = group_of(lane);
    return group.first_draft + (lane - group.first_lane) * group.window;
  }
  // The most columns one of its lanes holds.
  std::uint32_t widest_lane_columns() const {
    return groups[0].lanes > 0 ? groups[0].lane_columns() : groups[1].lane_columns();
  }
};

// `extended_lanes` lanes at the extended shape's window `extended_window`, then `base_lanes` at the
// base shape's `base_window`.
inline Layout layout_of(std::uint32_t extended_lanes, std::uint32_t extended_window, std::uint32_t base_lanes,
                        std::uint32_t base_window) {
  Layout layout{};
  layout.groups[0] = Group{0, extended_lanes, extended_window, 0, 0, true};
  layout.groups[1] = Group{extended_lanes,
                           base_lanes,
                           base_window,
                           (extended_window + 1) * extended_lanes,
                           extended_window * extended_lanes,
                           false,
                           2 * extended_lanes};
  return layout;
}

// A uniform round of `lanes` lanes at a shape of `window` drafts, the extended shape where `extended`.
inline Layout uniform(std::uint32_t lanes, std::uint32_t window, bool extended) {
  return extended ? layout_of(lanes, window, 0, 0) : layout_of(0, 0, lanes, window);
}

// GitHub #713 (spec #711): the most columns a verify round carries, its buffers' (8 lanes of 8, 4 of
// 16), and the mixes a mixed round takes, w extended lanes beside n base lanes at 16w + 8n columns
// or fewer: the window-15 geometry, which is the set at every window. A mix's index into a load's
// mixed graphs.
inline constexpr std::uint32_t kMaxColumns = 64;
inline constexpr std::uint32_t kMixes = 12;
inline constexpr std::array<std::array<std::uint32_t, 2>, kMixes> kMixTable = {
    {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {3, 1}, {3, 2}}};

// The index in kMixTable of `layout`'s mix, -1 for a uniform layout or a mix past the table.
inline int mix_index(const Layout &layout) {
  for (std::uint32_t i = 0; i < kMixes; ++i) {
    if (kMixTable[i][0] == layout.groups[0].lanes && kMixTable[i][1] == layout.groups[1].lanes) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// GitHub #713: the record capacity of a mixed round's base layout, rows of `base_width` columns laid
// at the stride of the extended records' `extended_capacity` rows of `extended_width` a layer, so
// that both layouts take one size of every plane over the backing they share and a layer's base
// row k is the extended layout's column 8k there; 0 where no whole number of base rows makes that
// stride, or where it takes more than `max_rows`.
inline std::uint32_t mixed_base_record_capacity(std::uint32_t extended_capacity, std::uint32_t extended_width,
                                                std::uint32_t base_width, std::uint32_t max_rows) {
  const std::uint32_t stride = extended_capacity * extended_width;
  if (base_width == 0 || stride % base_width != 0 || stride / base_width > max_rows) {
    return 0;
  }
  return stride / base_width;
}

// GitHub #713: whether each group of `layout` records where the round's one input projection writes
// its columns, its first record row at its first column, and holds its rows (the base group's behind
// its leading ones) within its records' capacity: the extended records' `extended_capacity`, the
// mixed base layout's `base_capacity`.
inline bool records_hold(const Layout &layout, std::uint32_t extended_capacity, std::uint32_t base_capacity) {
  for (const Group &group : layout.groups) {
    if (group.lanes == 0) {
      continue;
    }
    const std::uint32_t capacity = group.extended ? extended_capacity : base_capacity;
    if (group.first_record_row + group.lanes > capacity ||
        group.first_record_row * group.lane_columns() != group.first_column) {
      return false;
    }
  }
  return true;
}

// GitHub #713: the slots of a base group's `count` leading fold rows, which commit no column (the
// fold's own strict no-op, which reads and writes nothing): the lowest slots of [0, slot_count) that
// none of the group's `held` lanes holds, distinct, into `out`; false where the pool has too few.
inline bool leading_fold_slots(const std::int32_t *held, std::uint32_t held_count, std::uint32_t count,
                               std::int64_t slot_count, std::int32_t *out) {
  std::uint32_t found = 0;
  for (std::int64_t slot = 0; slot < slot_count && found < count; ++slot) {
    bool taken = false;
    for (std::uint32_t i = 0; i < held_count; ++i) {
      taken = taken || held[i] == slot;
    }
    if (!taken) {
      out[found++] = static_cast<std::int32_t>(slot);
    }
  }
  return found == count;
}

} // namespace ignis::round_layout
