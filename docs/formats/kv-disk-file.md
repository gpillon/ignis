# KV-disk file

One file holds one blob of the KV-disk tier (ADR 0045): a live sequence's
snapshot or a retained checkpoint. Kind 3, a retained prefix, is defined and
never written (#763). `FileHeader` in
`crates/runtime/src/kv_disk.rs` writes and reads it; the sizes are
`ignis_core::disk`'s (`DISK_HEADER_BYTES`, `DISK_IO_ALIGNMENT`). A process reads
only files it wrote itself, in the directory it holds locked, and nothing is
reused across a restart, so today the layout binds one process to itself. The
header still carries the whole identity of the load: that costs bytes, not
work, and a later step that keeps files across a restart or a model switch then
needs no format change.

## Layout

A 4 KiB header page, then the blob in windows of `window_bytes`, the last one
padded to the unbuffered-IO alignment (4 KiB). The body is written first and
the header page last: the header is the commit, and a file whose first page is
all zeros was torn before it.

The header page, every integer little-endian, unused bytes zero:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | magic, `IGNKVDK1` |
| 8 | 4 | format version, 1 |
| 12 | 4 | header bytes, 4096 |
| 16 | 1 | blob kind: 1 live, 2 checkpoint, 3 prefix |
| 20 | 4 | token count |
| 24 | 8 | blob length in bytes, before padding |
| 32 | 8 | window bytes |
| 40 | 4 | window count |
| 48 | 16 | match key (the empty key's for a live sequence) |
| 64 | 32 | structural artifact hash |
| 96 | 4 | KV format, its ABI code |
| 100 | 4 | layout version |
| 104 | 1 | 1 when the artifact has a sidecar payload hash |
| 108 | 32 | the sidecar's payload SHA-256, else zeros |
| 140 | 16 | RoPE scaling: factor, temperature, beta fast, beta slow, each an `f32`'s bits |
| 156 | 1 + 63 | drafter, a length byte then `backend:draft_tokens:head`; empty with no drafter |
| 220 | 1 + 255 | served model id, a length byte then UTF-8 |
| 512 | 4 x windows | CRC32 of each window, in order |
| 4092 | 4 | CRC32 of bytes 0 to 4091 |

The CRC table has room for 895 windows.

## What a reader refuses

A file is never restored, whole or in part, when:

- its header page is missing (a torn write) or fails its own CRC, its version
  or its layout;
- its identity differs from the load's in any field: served model id, artifact
  hash, KV format, layout version, drafter, sidecar payload hash, RoPE scaling
  (GitHub #205);
- its kind, match key, token count, length or CRC table is not the one the
  ledger recorded for the blob asked for;
- a window fails its CRC, found before that window's bytes reach the device.

A live blob's request then prefills again, which is logged as an error and
counted; a retained blob is discarded.

## Why CRC32

The check guards against torn and corrupt files, not adversaries: the
directory is the process's own. It must also keep pace with unbuffered reads
of about 3 GB/s on a CPU without SHA extensions, where SHA-256 runs slower
than the read it would guard
([the arithmetic](../findings/2026-10-07-kv-disk-arithmetic-on-the-rtx-5090-host.md)).
