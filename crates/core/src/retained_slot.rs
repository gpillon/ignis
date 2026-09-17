//! **Retained slots** (GitHub #211, ADR 0030, `CONTEXT.md`): the places in
//! the sequence pool, beside the [`N_DECODE_LANES`](crate::N_DECODE_LANES)
//! lane slots, that hold one mutable-state image each — a lane's own state
//! size — reserved at load.
//!
//! This is the allocator only: which retained slot is free, handed out, and
//! taken back. It knows nothing about devices, so its rules are pinned on the
//! CPU; the leaf's `ignis_seq_retained_store` / `ignis_seq_retained_load`
//! move the state in and out of the slot it names.

/// One retained slot handed out by [`RetainedSlots::take`]: an index in
/// `0..capacity`, never a lane's slot.
///
/// Neither `Copy` nor `Clone`: whoever holds it owns the slot until it gives
/// it back.
#[derive(Debug, PartialEq, Eq)]
pub struct RetainedSlot(u32);

impl RetainedSlot {
    /// The retained index the leaf's entry points take.
    pub fn index(&self) -> u32 {
        self.0
    }
}

/// A slot given back that this allocator does not have out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotTaken {
    pub index: u32,
}

impl std::fmt::Display for NotTaken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "retained slot {} is not taken from this allocator", self.index)
    }
}

impl std::error::Error for NotTaken {}

/// The free list of a pool's retained slots.
#[derive(Debug, Clone)]
pub struct RetainedSlots {
    /// Free indices, handed out from the back, lowest index first.
    free: Vec<u32>,
    taken: Vec<bool>,
}

impl RetainedSlots {
    /// `capacity` retained slots, every one free.
    pub fn new(capacity: u32) -> Self {
        Self {
            free: (0..capacity).rev().collect(),
            taken: vec![false; capacity as usize],
        }
    }

    /// How many retained slots the pool holds.
    pub fn capacity(&self) -> u32 {
        self.taken.len() as u32
    }

    /// How many are handed out.
    pub fn in_use(&self) -> u32 {
        self.capacity() - self.free.len() as u32
    }

    /// A free slot, or `None` when every one is handed out.
    pub fn take(&mut self) -> Option<RetainedSlot> {
        let index = self.free.pop()?;
        self.taken[index as usize] = true;
        Some(RetainedSlot(index))
    }

    /// Take `slot` back. Refused, with nothing changed, for a slot this
    /// allocator does not have out: one given back already, or one out of its
    /// range.
    pub fn give_back(&mut self, slot: RetainedSlot) -> Result<(), NotTaken> {
        let index = slot.0;
        match self.taken.get_mut(index as usize) {
            Some(taken) if *taken => {
                *taken = false;
                self.free.push(index);
                Ok(())
            }
            _ => Err(NotTaken { index }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_slot_is_taken_once_then_exhaustion_is_none() {
        let mut slots = RetainedSlots::new(3);
        assert_eq!(slots.capacity(), 3);
        assert_eq!(slots.in_use(), 0);
        let taken: Vec<_> = (0..3).map(|_| slots.take().expect("a free slot")).collect();
        let mut indices: Vec<_> = taken.iter().map(RetainedSlot::index).collect();
        indices.sort_unstable();
        assert_eq!(indices, [0, 1, 2]);
        assert_eq!(slots.in_use(), 3);
        assert_eq!(slots.take(), None);
    }

    #[test]
    fn a_slot_given_back_is_handed_out_again() {
        let mut slots = RetainedSlots::new(2);
        let first = slots.take().unwrap();
        let second = slots.take().unwrap();
        let index = first.index();
        slots.give_back(first).expect("taken");
        assert_eq!(slots.in_use(), 1);
        let again = slots.take().expect("the slot came back");
        assert_eq!(again.index(), index);
        assert_ne!(again.index(), second.index());
        assert_eq!(slots.take(), None);
    }

    #[test]
    fn a_slot_not_out_is_refused_and_changes_nothing() {
        let mut slots = RetainedSlots::new(2);
        let mut other = RetainedSlots::new(2);
        let held = slots.take().unwrap();
        // The same index from another allocator: this one already has it
        // out, so taking it back once is right...
        let twin = other.take().unwrap();
        assert_eq!(twin.index(), held.index());
        slots.give_back(held).expect("taken");
        // ...and a second give-back of that index is a double free.
        assert_eq!(slots.give_back(twin), Err(NotTaken { index: 0 }));
        assert_eq!(slots.in_use(), 0);
        let (a, b) = (slots.take().unwrap(), slots.take().unwrap());
        assert_ne!(a.index(), b.index(), "a refused give-back never listed a slot twice");
        assert_eq!(slots.take(), None);

        let mut wide = RetainedSlots::new(3);
        let _ = wide.take();
        let _ = wide.take();
        let out_of_range = wide.take().unwrap();
        assert_eq!(out_of_range.index(), 2);
        let err = slots.give_back(out_of_range).expect_err("index 2 of a 2-slot pool");
        assert_eq!(err, NotTaken { index: 2 });
        assert!(err.to_string().contains("retained slot 2"), "{err}");
    }

    #[test]
    fn no_retained_slots_hands_out_none() {
        let mut slots = RetainedSlots::new(0);
        assert_eq!(slots.capacity(), 0);
        assert_eq!(slots.take(), None);
    }
}
