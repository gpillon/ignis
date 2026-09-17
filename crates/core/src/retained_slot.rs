//! **Retained slots** (GitHub #211, ADR 0030, `CONTEXT.md`): the places in
//! the sequence pool, beside the [`N_DECODE_LANES`](crate::N_DECODE_LANES)
//! lane slots, that hold one mutable-state image each — a lane's own state
//! size — reserved at load.
//!
//! This is the allocator — which retained slot is free, handed out, and taken
//! back — and, since GitHub #215, the ledger of **who holds each one**
//! ([`RetainedSlotLedger`]). It knows nothing about devices, so its rules are
//! pinned on the CPU; the leaf's prefix publish and checkpoint capture move
//! the state into the slot the scheduler names.
//!
//! **A slot lives exactly as long as the backend's handle on what fills it.**
//! A holder is named the way the backend names that handle — a shared prefix
//! by its publisher and its length, a prompt checkpoint by its publisher — so
//! the scheduler gives a slot back at the one call that drops the handle, and
//! the two can never disagree about what still owns device state.

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

/// What holds a retained slot (GitHub #215): the image of a shared prefix or
/// of a prompt checkpoint, named as the backend names its handle on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainedHolder {
    /// A shared prefix — retained, chained or claimed by live siblings — named
    /// by the request that published it and the head's length (one request
    /// may publish two heads, #187 x #188).
    Prefix {
        publisher: crate::types::RequestId,
        tokens: u32,
    },
    /// A prompt checkpoint, named by the request that captured it (one per
    /// request).
    Checkpoint { publisher: crate::types::RequestId },
}

/// Why a publish or a capture was not taken (GitHub #215): retention is a
/// bet, and when the room for it is not there the request runs without
/// leaving reuse behind — it never waits and is never refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetainedSkip {
    /// No retained slot was free and no retained state could give one up.
    PublishNoSlot,
    /// The same, for a prompt checkpoint.
    CaptureNoSlot,
    /// A checkpoint's partial tail page is a KV page of the pool, and the pool
    /// had none spare.
    CaptureNoPage,
}

impl RetainedSkip {
    /// The log's spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            RetainedSkip::PublishNoSlot => "publish_skipped_no_slot",
            RetainedSkip::CaptureNoSlot => "capture_skipped_no_slot",
            RetainedSkip::CaptureNoPage => "capture_skipped_no_page",
        }
    }
}

/// The retained slots of a load and who holds each (GitHub #215, ADR 0030).
///
/// The one bound on retained state on the device: a publish or a capture
/// takes a slot here before the backend is asked for it, and gives it back
/// when the backend's handle goes.
#[derive(Debug)]
pub struct RetainedSlotLedger {
    slots: RetainedSlots,
    held: Vec<(RetainedHolder, RetainedSlot)>,
}

impl RetainedSlotLedger {
    /// A ledger over `capacity` retained slots, none held.
    pub fn new(capacity: u32) -> Self {
        Self {
            slots: RetainedSlots::new(capacity),
            held: Vec::new(),
        }
    }

    /// How many retained slots the load reserved.
    pub fn capacity(&self) -> u32 {
        self.slots.capacity()
    }

    /// How many are held.
    pub fn in_use(&self) -> u32 {
        self.slots.in_use()
    }

    /// A free slot for `holder`, as the index the leaf takes, or `None` when
    /// every one is held.
    ///
    /// Panics when `holder` already holds one: a second image under one name
    /// is a slot nothing would ever give back.
    pub fn take(&mut self, holder: RetainedHolder) -> Option<u32> {
        assert!(
            self.index_of(holder).is_none(),
            "{holder:?} already holds a retained slot"
        );
        let slot = self.slots.take()?;
        let index = slot.index();
        self.held.push((holder, slot));
        Some(index)
    }

    /// Give back the slot `holder` holds. `false`, with nothing changed, when
    /// it holds none — a handle whose image lives in KV-RAM, or one never
    /// given a slot.
    pub fn give_back(&mut self, holder: RetainedHolder) -> bool {
        let Some(pos) = self.held.iter().position(|(h, _)| *h == holder) else {
            return false;
        };
        let (_, slot) = self.held.swap_remove(pos);
        self.slots
            .give_back(slot)
            .expect("a held slot is taken from this allocator");
        true
    }

    /// The slot `holder` holds, if any.
    fn index_of(&self, holder: RetainedHolder) -> Option<u32> {
        self.held
            .iter()
            .find(|(h, _)| *h == holder)
            .map(|(_, slot)| slot.index())
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

    const BLOCK: RetainedHolder = RetainedHolder::Prefix {
        publisher: 1,
        tokens: 32,
    };
    const CHAIN: RetainedHolder = RetainedHolder::Prefix {
        publisher: 1,
        tokens: 48,
    };
    const OPENER: RetainedHolder = RetainedHolder::Checkpoint { publisher: 1 };

    #[test]
    fn a_ledger_hands_each_holder_its_own_slot_until_none_is_left() {
        let mut ledger = RetainedSlotLedger::new(2);
        let block = ledger.take(BLOCK).expect("a free slot");
        let chain = ledger.take(CHAIN).expect("another");
        assert_ne!(block, chain, "one request's two heads are two slots");
        assert_eq!(ledger.take(OPENER), None, "the third holder finds none");
        assert_eq!(ledger.in_use(), 2);
        assert_eq!(ledger.index_of(CHAIN), Some(chain));
        assert_eq!(ledger.index_of(OPENER), None);
    }

    #[test]
    fn giving_a_slot_back_frees_it_for_the_next_holder() {
        let mut ledger = RetainedSlotLedger::new(1);
        let index = ledger.take(BLOCK).unwrap();
        assert!(ledger.give_back(BLOCK));
        assert!(!ledger.give_back(BLOCK), "given back once");
        assert!(!ledger.give_back(OPENER), "a holder with no slot gives nothing back");
        assert_eq!(ledger.take(OPENER), Some(index));
        assert_eq!(ledger.in_use(), 1);
    }

    #[test]
    #[should_panic(expected = "already holds a retained slot")]
    fn a_holder_takes_one_slot_at_most() {
        let mut ledger = RetainedSlotLedger::new(2);
        ledger.take(OPENER);
        ledger.take(OPENER);
    }

    #[test]
    fn a_skip_is_spelled_as_the_log_names_it() {
        assert_eq!(RetainedSkip::PublishNoSlot.as_str(), "publish_skipped_no_slot");
        assert_eq!(RetainedSkip::CaptureNoSlot.as_str(), "capture_skipped_no_slot");
        assert_eq!(RetainedSkip::CaptureNoPage.as_str(), "capture_skipped_no_page");
    }
}
