---
title: "JSON Schema keywords"
description: "The JSON Schema keywords structured output enforces, its formats and annotations, the keywords it leaves inert without strict, and OpenAI's four limits."
sidebar:
  order: 7
  badge:
    text: generated
    variant: note
editUrl: false
---

:::note
Generated from the code by `make docs-reference` (`crates/server/src/user_docs.rs`); an edit here is overwritten, and `cargo test` fails while this page and the code disagree.
:::

What a `json_schema` in `response_format` may carry, as the grammar reads it. [Structured output](../api/structured-output.md) explains how an answer is held to it, what the grammar writes, and the refusals.

## Enforced

OpenAI's strict subset, held in every answer with `strict: true` or without it: a `pattern` that is a regular expression, a `format` among the nine below, a bound a decimal holds exactly ([otherwise inert](../api/structured-output.md#without-strict), or refused under `strict`).

`type`, `properties`, `required`, `additionalProperties`, `items`, `minItems`, `maxItems`, `enum`, `const`, `anyOf`, `$defs`, `definitions`, `$ref`, `pattern`, `format`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`

## Formats

`format` holds a string to one of these nine. Any other is refused under `strict: true` (`unsupported_format`) and inert without it.

`date-time`, `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`

## Annotations

Accepted anywhere, with `strict: true` or without it, and never enforced.

`title`, `description`, `$comment`, `examples`, `default`, `deprecated`, `readOnly`, `writeOnly`, `$schema`, `$id`

## Inert without `strict`

Any other keyword is refused under `strict: true` (`unsupported_keyword`, naming it). Without `strict` the schema is served and the keyword is inert: the answer is not held to it. These are JSON Schema 2020-12's, with the earlier drafts' `dependencies` and `additionalItems`:

`$anchor`, `$dynamicRef`, `$dynamicAnchor`, `$vocabulary`, `prefixItems`, `contains`, `patternProperties`, `dependentSchemas`, `propertyNames`, `if`, `then`, `else`, `allOf`, `oneOf`, `not`, `unevaluatedItems`, `unevaluatedProperties`, `maxLength`, `minLength`, `uniqueItems`, `maxContains`, `minContains`, `maxProperties`, `minProperties`, `dependentRequired`, `contentEncoding`, `contentMediaType`, `contentSchema`, `dependencies`, `additionalItems`

## Limits

OpenAI's limits on a schema, held with `strict: true` and without it. A schema past one is refused with the code below, naming the limit.

| Limit | At most | Code |
|---|---|---|
| properties | 5,000 | `too_many_properties` |
| nesting | 10 | `too_deeply_nested` |
| enum values | 1,000 | `too_many_enum_values` |
| characters | 120,000 | `too_many_characters` |
