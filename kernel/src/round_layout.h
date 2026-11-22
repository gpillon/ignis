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
                           false};
  return layout;
}

// A uniform round of `lanes` lanes at a shape of `window` drafts, the extended shape where `extended`.
inline Layout uniform(std::uint32_t lanes, std::uint32_t window, bool extended) {
  return extended ? layout_of(lanes, window, 0, 0) : layout_of(0, 0, lanes, window);
}

} // namespace ignis::round_layout
