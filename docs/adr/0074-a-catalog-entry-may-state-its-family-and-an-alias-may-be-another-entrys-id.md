---
status: accepted
amends: [ADR-0047, ADR-0064]
---

# A catalog entry may state its family, and an alias may be another entry's id

The 16 GB build of the uncensored variant (GitHub #386) must answer to
`qwen3.8-27b-abliterated` on a 16 GB card, as `qwen3.8-27b-16g` answers to
`qwen3.8-27b`, so a client's configuration does not change with the card.
ADR 0064 allowed only a family's default id as an alias, because the catalog
declared no family (ADR 0047) and read an entry's family off its id or its
aliases; `qwen3.8-27b-abliterated` is neither, so nothing in the catalog said
its family. A catalog entry now may state `family` (the family's own id), and
must when an alias involves it, listed by it or naming its id, and neither its
id nor its aliases name one; an entry no alias involves needs none, so an
operator catalog written before this still loads. The artifact's header must
agree, checked where
ADR 0064 checks an alias today, so the header stays the authority and the
catalog's word is a claim it verifies. An alias may then be any served id of
an entry of the same family, answered on a card class below that entry's own.
Reading the family off an id's prefix was rejected: an operator's id is a
name, and a prefix match would be a coincidence the catalog trusted.
