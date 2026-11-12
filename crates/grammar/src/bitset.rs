//! A set of token ids over the whole vocabulary: the repair's permitted set
//! in its mask form, one bit per logits column (31,040 bytes at 248,320).

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TokenSet {
    words: Vec<u64>,
    len: usize,
}

impl TokenSet {
    pub fn new(len: usize) -> Self {
        Self { words: vec![0; len.div_ceil(64)], len }
    }

    /// The number of ids the set ranges over (the logits row).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    #[inline]
    pub fn insert(&mut self, id: u32) {
        self.words[id as usize >> 6] |= 1u64 << (id & 63);
    }

    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        (id as usize) < self.len && self.words[id as usize >> 6] & (1u64 << (id & 63)) != 0
    }

    pub fn union_with(&mut self, other: &TokenSet) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a |= *b;
        }
    }

    pub fn copy_from(&mut self, other: &TokenSet) {
        self.words.copy_from_slice(&other.words);
        self.len = other.len;
    }

    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let bit = w.trailing_zeros();
                w &= w - 1;
                Some((i as u32) * 64 + bit)
            })
        })
    }

    /// The bytes the mask takes on the wire to the leaf: one bit per id.
    pub fn byte_len(&self) -> usize {
        self.len.div_ceil(8)
    }

    /// The mask as words: id `i` is bit `i % 64` of word `i / 64`, the bits
    /// past the last id clear. The form a repair sends down (#586).
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn heap_bytes(&self) -> usize {
        self.words.len() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains_iter_and_count_agree() {
        let mut s = TokenSet::new(248_320);
        for id in [0u32, 63, 64, 1000, 248_319] {
            s.insert(id);
        }
        assert!(s.contains(63) && s.contains(64) && !s.contains(65));
        assert_eq!(s.count(), 5);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![0, 63, 64, 1000, 248_319]);
        assert_eq!(s.byte_len(), 31_040);
        assert_eq!(s.words().len(), 3880);
        assert_eq!(s.words()[0], 1 | 1 << 63);
        assert_eq!(s.words()[1], 1);
        assert_eq!(s.words()[3879], 1 << (248_319 % 64));
    }
}
