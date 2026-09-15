# Playground: n-gram retrieval over attachments and memory (RAG-like)

## Why

Today the model reads attachments with `read_file` in 20,000-character
pieces and sees memory only as titles, then `memory_read`s whole bodies.
On long files and many notes that is slow and burns context. A small
retrieval step in the browser can hand the model the few passages that
match the question instead.

## Idea

- Index each session attachment and every memory note in the browser:
  split into passages (for example ~800 characters with overlap), build
  character or word n-grams (trigrams / bigrams), score with BM25 or
  TF-IDF on those n-grams.
- A tool, e.g. `search_files({query, limit?})` (and/or `memory_search`),
  returns the top passages with file name, character range and score, so
  `read_file` can fetch the surroundings.
- No network, no embedding model: pure JS in `web/src/tools/local/`,
  indexes rebuilt when attachments or notes change, kept in memory.

## Open questions

- Word n-grams vs character n-grams (Italian/English mixed text, code)?
- One tool for files and memory, or two?
- Should the top hits be injected automatically into the prompt for the
  user's message, or only on a tool call?
- Where does indexing run for big PDFs (main thread vs a worker)?

## Acceptance (draft)

- A query about a detail deep in a long attachment returns the passage
  holding it in the top 3, with its range.
- Index build for a 1 MB text file stays under ~200 ms off the main thread
  or does not block typing.
- Unit tests for passage splitting, n-gram scoring and ranking.
