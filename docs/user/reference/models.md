---
title: "Built-in models"
description: "The catalog compiled into ignis-server: every model it can fetch, the repository and revision it is pinned at, and every file's size and SHA-256."
sidebar:
  order: 3
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

The models the server knows how to fetch without an operator catalog. Each is pinned file by file, byte count and SHA-256, at a fixed revision of its Hugging Face repository, and nothing outside the binary can change a pin ([Get a model](../guides/models.md), [an operator catalog](../guides/operator-catalog.md)).

| Id | Repository | Size | Artifact |
|---|---|---|---|
| [`qwen3.8-27b`](#qwen38-27b) | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer) | 18.1 GiB | `qwen3_8_27b_nvfp4full-v2.ninfer` |
| [`qwen3.8-27b-abliterated`](#qwen38-27b-abliterated) | [`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) | 18.1 GiB | `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer` |
| [`qwen3.8-flash-next`](#qwen38-flash-next) | [`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis) | 67.8 GiB | `qwen3_8_flash_next_trellis_a25-v2.ninfer` |
| [`ignis-image-2.1-turbo`](#ignis-image-21-turbo) | [`gpillon/ignis-image-2.1-turbo`](https://huggingface.co/gpillon/ignis-image-2.1-turbo) | 15.6 GiB | `ignis_image_2_1_turbo-v2.ninfer` |
| [`qwen3.8-27b-16g`](#qwen38-27b-16g) | [`gpillon/Qwen3.8-27B-trellis-3bpw-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-ignis) | 12.5 GiB | `qwen3_8_27b_trellis_sc30-v2.ninfer` |
| [`qwen3.8-27b-abliterated-16g`](#qwen38-27b-abliterated-16g) | [`gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis) | 12.5 GiB | `qwen3_8_27b_trellis_sc30-v2-huihui-abliterated.ninfer` |
| [`qwen3-asr-1.7b`](#qwen3-asr-17b) | [`gpillon/Qwen3-ASR-1.7B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-1.7B-GGUF-ignis) | 1.5 GiB | `Qwen3-ASR-1.7B-Q4_K_M.gguf` |
| [`qwen3-asr-1.7b-q8`](#qwen3-asr-17b-q8) | [`gpillon/Qwen3-ASR-1.7B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-1.7B-GGUF-ignis) | 2.3 GiB | `Qwen3-ASR-1.7B-Q8_0.gguf` |
| [`qwen3-asr-0.6b`](#qwen3-asr-06b) | [`gpillon/Qwen3-ASR-0.6B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-0.6B-GGUF-ignis) | 0.9 GiB | `Qwen3-ASR-0.6B-Q8_0.gguf` |
| [`nemotron-3-diarization`](#nemotron-3-diarization) | [`gpillon/Nemotron-3-Diarization-GGUF-ignis`](https://huggingface.co/gpillon/Nemotron-3-Diarization-GGUF-ignis) | 0.2 GiB | `Nemotron-3-Diarization-F16.gguf` |

## qwen3.8-27b

[`gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer) at revision `e961b419b672e183aa55df8c4b975abc82006e8a`, 19,406,949,449 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_27b_nvfp4full-v2.ninfer.graft.json` | 6,981 | `b87e3c005fb1daf7d6b208da52790f95a3f478732f767f87417a5d7fc23491d1` |
| `qwen3_8_27b_nvfp4full-v2.ninfer` | 19,406,942,468 | `abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af` |

## qwen3.8-27b-abliterated

[`gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer`](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-abliterated-NInfer) at revision `ef3216949e42c42fc713feffb2e05849d57f0f3b`, 19,406,984,175 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer.graft.json` | 41,707 | `09b0df892c314563192b0a4f102139939d4ef6261fcd210439d4e2636d12e6fe` |
| `qwen3_8_27b_nvfp4full-v2-huihui-abliterated.ninfer` | 19,406,942,468 | `18954280c794cb2ff1fc24ada8158de1df11bf0a0a2ea63f109045af48905ef1` |

## qwen3.8-flash-next

[`gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis`](https://huggingface.co/gpillon/Qwen3.8-Flash-Next-trellis-a25-ignis) at revision `8d94db83cd8e58d6be31e796723e5e35b960638c`, 72,799,980,571 bytes in 4 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_flash_next_mtp_3p0-v2.ninfer.conversion.json` | 179,029 | `a5b2c10fe0745cfe6b8e2914135e44903cfb80acb3f0db21e549e76641f14127` |
| `qwen3_8_flash_next_trellis_a25-v2.ninfer.conversion.json` | 1,183,942 | `9858d9fb03dd66fa2558b5dc6aa6463ee70f00024ffbd746545b231c9514f7a0` |
| `qwen3_8_flash_next_mtp_3p0-v2.ninfer` | 1,037,905,920 | `432fe5c0838027047c5b46c4fde526f3d3ef3d126f00fed1213921f8705988d2` |
| `qwen3_8_flash_next_trellis_a25-v2.ninfer` | 71,760,711,680 | `12e3b265b0678ba935a4b05d1b612ba4b68b3a96c3026d818ad65838985fd864` |

## ignis-image-2.1-turbo

[`gpillon/ignis-image-2.1-turbo`](https://huggingface.co/gpillon/ignis-image-2.1-turbo) at revision `d6912417aa47320cd71c78e19d99872faa5cb63b`, 16,722,262,468 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `ignis_image_2_1_turbo-v2.ninfer.conversion.json` | 617,924 | `a97f04912b9a6d76e828efc02b7e458d0f45d00bc00d794cca39d703ad0f33ca` |
| `ignis_image_2_1_turbo-v2.ninfer` | 16,721,644,544 | `4205dd1932f0801804b0a544fe1340697e4e0a0dea4a276996bcb981006050af` |

## qwen3.8-27b-16g

[`gpillon/Qwen3.8-27B-trellis-3bpw-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-ignis) at revision `f5bbd2cc14c9ca573ef5553c100ac69410f1fe58`, 13,380,186,619 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_27b_trellis_sc30-v2.ninfer.conversion.json` | 309,495 | `210ab54bfe8b04e5fc4337ee3755c340ea3f21239aa625c505aa2406645a340d` |
| `qwen3_8_27b_trellis_sc30-v2.ninfer` | 13,379,877,124 | `2c4c9ee4d6b25c08da6c2d85840126e73d120564e230f926b6230b5b95ee8b3c` |

## qwen3.8-27b-abliterated-16g

[`gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis`](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-abliterated-ignis) at revision `94a8bfc0f47ce86403f274be03e2c0f79500727c`, 13,380,216,667 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `qwen3_8_27b_trellis_sc30-v2-huihui-abliterated.ninfer.conversion.json` | 339,543 | `b5b61863b218bf722e4786ec8ca5e7ffde03548068f92184f10ecd95f6fd0f61` |
| `qwen3_8_27b_trellis_sc30-v2-huihui-abliterated.ninfer` | 13,379,877,124 | `4929bc8e1fbc747092f96aebae6b81a0b619db452d05a56df3499fbe1afe9ac5` |

## qwen3-asr-1.7b

[`gpillon/Qwen3-ASR-1.7B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-1.7B-GGUF-ignis) at revision `f8a5263e398c1b3606abedc6b9c74dcbced9f93f`, 1,638,144,608 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` | 355,709,472 | `f325a44bdbc23b2df78ed35ad6342da3aabf1e618e1a9ffd1ef8861cb3b1b0f6` |
| `Qwen3-ASR-1.7B-Q4_K_M.gguf` | 1,282,435,136 | `5c5f878af090a70b25dc7a0f1090620e734654e9eceb6f4ac4656423c243974c` |

## qwen3-asr-1.7b-q8

[`gpillon/Qwen3-ASR-1.7B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-1.7B-GGUF-ignis) at revision `f8a5263e398c1b3606abedc6b9c74dcbced9f93f`, 2,520,744,544 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf` | 355,709,472 | `f325a44bdbc23b2df78ed35ad6342da3aabf1e618e1a9ffd1ef8861cb3b1b0f6` |
| `Qwen3-ASR-1.7B-Q8_0.gguf` | 2,165,035,072 | `9039c0cbbc13be5c25f2b34fcb70258f68f06b64f67a48831c7e13c72593e1e7` |

## qwen3-asr-0.6b

[`gpillon/Qwen3-ASR-0.6B-GGUF-ignis`](https://huggingface.co/gpillon/Qwen3-ASR-0.6B-GGUF-ignis) at revision `af02916ea7a532a667a907c2a163b16a8c04b433`, 1,019,141,984 bytes in 2 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `mmproj-Qwen3-ASR-0.6B-Q8_0.gguf` | 214,392,608 | `ee5e87436bcf8b1c550edd4300fd0516067d55b547611734d27fbc55ecaec3bc` |
| `Qwen3-ASR-0.6B-Q8_0.gguf` | 804,749,376 | `027ad3329549cd2c14e4ebd7d104e0390566e06570270c33062529474ec56015` |

## nemotron-3-diarization

[`gpillon/Nemotron-3-Diarization-GGUF-ignis`](https://huggingface.co/gpillon/Nemotron-3-Diarization-GGUF-ignis) at revision `d46d59d03be709db18e223b6bf92e8c32d14f422`, 198,805,760 bytes in 1 files.

| File | Bytes | SHA-256 |
|---|---|---|
| `Nemotron-3-Diarization-F16.gguf` | 198,805,760 | `f7a80981d178be5d683aee225166bc03806aac2d9b629f42f418d4e5fd2c8623` |
