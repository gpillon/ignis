---
title: "Counting a prompt"
description: "POST /v1/tokenize: how many tokens a chat body prefills, without serving it; and /v1/detokenize."
sidebar:
  order: 8
---

`POST /v1/tokenize` renders a chat body exactly as `/v1/chat/completions` would —
chat template, tool block, thinking controls, system block — and answers how many
tokens it prefills, **without submitting it**: no lane, no GPU, no KV page. The
`count` is the `usage.prompt_tokens` the same body reports when served, and
`max_model_len` is the context the load runs (`--model-max-context`, or what the 27B's
load adapted it to, [Adaptation](../configuration/index.md)), so one call says whether a body
fits.

```bash
curl http://127.0.0.1:8000/v1/tokenize \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"tools":[...],"return_token_ids":true}'
# {"count":1234,"max_model_len":262144,"token_ids":[...]}
```

Send `messages` **or** a raw `prompt` (tokenized with no template), never both.
`return_token_ids` and `return_text` add the ids and the rendered prompt.
A `response_format` of `json_schema` counts too: the server describes the
schema to the model after the conversation ([Structured
output](structured-output.md)), so the count includes that block, and
`ignis_describe_schema: false` counts the prompt without it. The format is
checked as chat checks it, so a schema chat would refuse is refused here.
Sampling fields are ignored, `stream` is refused, and a body with an `image_url`
part is a `400 media_not_countable`: an image's cost is its grid after
preparation, which only fetching and decoding it would tell — send the request.
`POST /v1/detokenize` takes `{"token_ids":[...]}` and answers `{"text":"..."}`; an
id outside the vocabulary is a `400` naming its index. The tokenizer normalizes
text to NFC, so a decomposed `e` + accent comes back as the composed `é` (the
`text` a call returns is the rendered prompt as sent, before that).
Neither route enters the scheduler; the count of calls is
`ignis_tokenize_requests_total{route}` on the metrics listener.
