//! The validator (GitHub #608, spec #582): a JSON value checked against a
//! JSON schema as JSON Schema (Draft 2020-12) reads it, each failure named by
//! its [`Kind`].
//!
//! It checks what the model's tool calls deliver against their tools'
//! schemas, and it is the oracle of the grammar's property tests: everything
//! the grammar accepts must be valid here. So it shares nothing with the
//! automaton but the JSON parser and the ECMA-262 pattern translation: its
//! numbers are exact decimals of its own, and its formats are written from
//! their RFCs, not from the grammar's regular expressions. Its verdicts are
//! held to `jsonschema`'s on a corpus (`tests/tool_arguments.rs`).
//!
//! What it reads:
//! - `type`, `enum`, `const`; an `integer` is a number written with neither a
//!   fraction nor an exponent (`1.0` is not one), as the independent
//!   validator reads exact numbers;
//! - `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`,
//!   `multipleOf`, in exact decimal arithmetic;
//! - `minLength`, `maxLength` (in characters), `pattern` (a search, not a
//!   whole match), and the nine formats the grammar holds: `date-time`,
//!   `time`, `date`, `duration`, `email`, `hostname`, `ipv4`, `ipv6`, `uuid`;
//! - `properties`, `patternProperties`, `additionalProperties`, `required`,
//!   `dependentRequired`, `minProperties`, `maxProperties`, `propertyNames`;
//! - `prefixItems`, `items` (its array form read as `prefixItems`),
//!   `minItems`, `maxItems`, `uniqueItems`;
//! - `allOf`, `anyOf`, `oneOf`, `not`, `if`/`then`/`else`, `true` and `false`
//!   schemas, and a `$ref` that points into the schema itself (`#`,
//!   `#/$defs/...`, `#/definitions/...`).
//!
//! Anything else is not checked: another format, a keyword such as `contains`
//! or `unevaluatedProperties`, a pattern that is not a regular expression
//! (lookaround, a backreference), a `$ref` to another document.

use std::cmp::Ordering;

use crate::json::Json;

/// What a violation breaks, coarsely: the kinds an operator counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// A property `required` (or `dependentRequired`) names is absent.
    Missing,
    /// A value of a type its schema's `type` does not admit. The schema's
    /// other keywords are not checked against it.
    Type,
    /// A value outside its `enum`, or other than its `const`.
    Enum,
    /// A property an object closed by `additionalProperties: false` does not
    /// declare.
    UnknownParameter,
    /// A value of an admitted type that breaks another of its keywords: a
    /// `pattern`, a `format`, a bound, a length, an item or property count,
    /// `multipleOf`, `uniqueItems`, `propertyNames`, `not`, a `false` schema,
    /// a `oneOf` matched more than once.
    Constraint,
}

impl Kind {
    /// The kind's name, as `ignis_tool_call_violations_total` labels it.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Missing => "missing",
            Kind::Type => "type",
            Kind::Enum => "enum",
            Kind::UnknownParameter => "unknown_parameter",
            Kind::Constraint => "constraint",
        }
    }
}

/// One failure of a value against its schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    pub kind: Kind,
    /// Where in the value, as a JSON pointer (`""` is the value itself).
    pub at: String,
    /// The keyword that failed (`"false"` for a `false` schema).
    pub keyword: &'static str,
}

/// Every violation of `value` against `schema`, in the order they were found;
/// none when it is valid. A `oneOf` or `anyOf` no member admits reports the
/// violations of the member that came closest (its type admitted the value,
/// fewest violations, first).
pub fn violations(schema: &Json, value: &Json) -> Vec<Violation> {
    let mut checker = Checker { root: schema, depth: 0, steps: 0 };
    let mut out = Vec::new();
    checker.check(schema, value, "", &mut out);
    out
}

/// Whether `value` is valid under `schema`.
pub fn is_valid(schema: &Json, value: &Json) -> bool {
    violations(schema, value).is_empty()
}

/// How deep subschemas and `$ref`s may nest while one value is checked: a
/// `$ref` cycle that never descends into the value stops here, unchecked.
/// This bound and the two below cap the check's own work on a client's
/// schema, and decide only how a call is counted, never what a response
/// carries, so they are not configuration fields (`CODING_STANDARDS.md`,
/// "Limits are config fields"); one that came to decide a response would be.
const MAX_DEPTH: usize = 256;

/// How many subschemas, and `uniqueItems` comparisons, one value's check may
/// take: past it the rest is not checked. A schema whose `anyOf` members
/// `$ref` each other can take exponential time, and the schema is the
/// client's.
const MAX_STEPS: usize = 200_000;

struct Checker<'a> {
    root: &'a Json,
    depth: usize,
    steps: usize,
}

fn violation(kind: Kind, at: &str, keyword: &'static str) -> Violation {
    Violation { kind, at: at.to_owned(), keyword }
}

/// A JSON pointer token, escaped.
fn child(at: &str, token: &str) -> String {
    format!("{at}/{}", crate::json::pointer_token(token))
}

fn entries(schema: &Json) -> &[(String, Json)] {
    match schema {
        Json::Obj(entries) => entries,
        _ => &[],
    }
}

fn number(value: &Json) -> Option<Dec> {
    value.number_text().and_then(Dec::parse)
}

impl<'a> Checker<'a> {
    fn check(&mut self, schema: &'a Json, value: &Json, at: &str, out: &mut Vec<Violation>) {
        let keywords = match schema {
            Json::Bool(true) => return,
            Json::Bool(false) => {
                out.push(violation(Kind::Constraint, at, "false"));
                return;
            }
            Json::Obj(entries) => entries,
            // Not a schema: nothing it could say.
            _ => return,
        };
        self.steps += 1;
        if self.depth >= MAX_DEPTH || self.steps > MAX_STEPS {
            return;
        }
        self.depth += 1;
        self.keywords(schema, keywords, value, at, out);
        self.depth -= 1;
    }

    /// The members of `subschemas` (an array of schemas) each checked on its
    /// own: their violations, one list per member.
    fn members(&mut self, subschemas: &'a Json, value: &Json, at: &str) -> Vec<Vec<Violation>> {
        let Json::Arr(members) = subschemas else { return Vec::new() };
        members
            .iter()
            .map(|member| {
                let mut found = Vec::new();
                self.check(member, value, at, &mut found);
                found
            })
            .collect()
    }

    fn keywords(&mut self, schema: &'a Json, keywords: &'a [(String, Json)], value: &Json, at: &str, out: &mut Vec<Violation>) {
        if let Some(Json::Str(reference)) = schema.get("$ref") {
            if let Some(target) = resolve(self.root, reference) {
                self.check(target, value, at, out);
            }
        }
        if let Some(types) = schema.get("type") {
            if !admits_type(types, value) {
                out.push(violation(Kind::Type, at, "type"));
                return;
            }
        }
        for (keyword, arg) in keywords {
            match keyword.as_str() {
                "enum" => {
                    if let Json::Arr(values) = arg {
                        if !values.iter().any(|v| equal(v, value)) {
                            out.push(violation(Kind::Enum, at, "enum"));
                        }
                    }
                }
                "const" => {
                    if !equal(arg, value) {
                        out.push(violation(Kind::Enum, at, "const"));
                    }
                }
                "allOf" => {
                    for found in self.members(arg, value, at) {
                        out.extend(found);
                    }
                }
                "anyOf" => {
                    let found = self.members(arg, value, at);
                    if !found.is_empty() && !found.iter().any(Vec::is_empty) {
                        out.extend(closest(found, at));
                    }
                }
                "oneOf" => {
                    let found = self.members(arg, value, at);
                    match found.iter().filter(|f| f.is_empty()).count() {
                        1 => {}
                        0 if !found.is_empty() => out.extend(closest(found, at)),
                        0 => {}
                        _ => out.push(violation(Kind::Constraint, at, "oneOf")),
                    }
                }
                "not" => {
                    let mut found = Vec::new();
                    self.check(arg, value, at, &mut found);
                    if found.is_empty() {
                        out.push(violation(Kind::Constraint, at, "not"));
                    }
                }
                "if" => {
                    let mut found = Vec::new();
                    self.check(arg, value, at, &mut found);
                    let branch = if found.is_empty() { schema.get("then") } else { schema.get("else") };
                    if let Some(branch) = branch {
                        self.check(branch, value, at, out);
                    }
                }
                _ => {}
            }
        }
        match value {
            Json::Num(_) => self.number_keywords(schema, value, at, out),
            Json::Str(s) => self.string_keywords(schema, s, at, out),
            Json::Arr(items) => self.array_keywords(schema, items, at, out),
            Json::Obj(entries) => self.object_keywords(schema, entries, at, out),
            Json::Null | Json::Bool(_) => {}
        }
    }

    fn number_keywords(&mut self, schema: &Json, value: &Json, at: &str, out: &mut Vec<Violation>) {
        let Some(v) = number(value) else { return };
        let bound = |keyword: &'static str, holds: fn(Ordering) -> bool, out: &mut Vec<Violation>| {
            if let Some(b) = schema.get(keyword).and_then(number) {
                if !holds(v.cmp(&b)) {
                    out.push(violation(Kind::Constraint, at, keyword));
                }
            }
        };
        bound("minimum", |o| o != Ordering::Less, out);
        bound("maximum", |o| o != Ordering::Greater, out);
        bound("exclusiveMinimum", |o| o == Ordering::Greater, out);
        bound("exclusiveMaximum", |o| o == Ordering::Less, out);
        if let (Some(m), Some(text), Some(m_text)) =
            (schema.get("multipleOf").and_then(number), value.number_text(), schema.get("multipleOf").and_then(Json::number_text))
        {
            if m.sign() > 0 && !v.is_multiple_of(&m, text, m_text) {
                out.push(violation(Kind::Constraint, at, "multipleOf"));
            }
        }
    }

    fn string_keywords(&mut self, schema: &Json, s: &str, at: &str, out: &mut Vec<Violation>) {
        let chars = s.chars().count();
        if let Some(min) = schema.get("minLength").and_then(count) {
            if chars < min {
                out.push(violation(Kind::Constraint, at, "minLength"));
            }
        }
        if let Some(max) = schema.get("maxLength").and_then(count) {
            if chars > max {
                out.push(violation(Kind::Constraint, at, "maxLength"));
            }
        }
        if let Some(Json::Str(pattern)) = schema.get("pattern") {
            if search(pattern, s) == Some(false) {
                out.push(violation(Kind::Constraint, at, "pattern"));
            }
        }
        if let Some(Json::Str(format)) = schema.get("format") {
            if formats::check(format, s) == Some(false) {
                out.push(violation(Kind::Constraint, at, "format"));
            }
        }
    }

    fn array_keywords(&mut self, schema: &'a Json, items: &[Json], at: &str, out: &mut Vec<Violation>) {
        let prefix = match (schema.get("prefixItems"), schema.get("items")) {
            (Some(Json::Arr(prefix)), _) => Some(prefix),
            (None, Some(Json::Arr(tuple))) => Some(tuple),
            _ => None,
        };
        let skipped = prefix.map_or(0, Vec::len);
        if let Some(prefix) = prefix {
            for (i, (sub, item)) in prefix.iter().zip(items).enumerate() {
                self.check(sub, item, &child(at, &i.to_string()), out);
            }
        }
        if let Some(rest) = schema.get("items").filter(|items| !matches!(items, Json::Arr(_))) {
            for (i, item) in items.iter().enumerate().skip(skipped) {
                self.check(rest, item, &child(at, &i.to_string()), out);
            }
        }
        if schema.get("minItems").and_then(count).is_some_and(|min| items.len() < min) {
            out.push(violation(Kind::Constraint, at, "minItems"));
        }
        if schema.get("maxItems").and_then(count).is_some_and(|max| items.len() > max) {
            out.push(violation(Kind::Constraint, at, "maxItems"));
        }
        if schema.get("uniqueItems") == Some(&Json::Bool(true)) {
            'pairs: for (i, a) in items.iter().enumerate() {
                for b in &items[i + 1..] {
                    self.steps += 1;
                    if self.steps > MAX_STEPS {
                        break 'pairs;
                    }
                    if equal(a, b) {
                        out.push(violation(Kind::Constraint, at, "uniqueItems"));
                        break 'pairs;
                    }
                }
            }
        }
    }

    fn object_keywords(&mut self, schema: &'a Json, members: &[(String, Json)], at: &str, out: &mut Vec<Violation>) {
        let properties = schema.get("properties").map(entries).unwrap_or_default();
        let patterns = schema.get("patternProperties").map(entries).unwrap_or_default();
        for (key, item) in members {
            let path = child(at, key);
            let mut declared = false;
            if let Some((_, sub)) = properties.iter().find(|(name, _)| name == key) {
                declared = true;
                self.check(sub, item, &path, out);
            }
            for (pattern, sub) in patterns {
                if search(pattern, key) == Some(true) {
                    declared = true;
                    self.check(sub, item, &path, out);
                }
            }
            if !declared {
                match schema.get("additionalProperties") {
                    Some(Json::Bool(false)) => out.push(violation(Kind::UnknownParameter, &path, "additionalProperties")),
                    Some(sub) => self.check(sub, item, &path, out),
                    None => {}
                }
            }
            if let Some(names) = schema.get("propertyNames") {
                let mut found = Vec::new();
                self.check(names, &Json::Str(key.clone()), &path, &mut found);
                if !found.is_empty() {
                    out.push(violation(Kind::Constraint, &path, "propertyNames"));
                }
            }
        }
        let has = |name: &str| members.iter().any(|(key, _)| key == name);
        if let Some(Json::Arr(required)) = schema.get("required") {
            for name in required.iter().filter_map(Json::as_str) {
                if !has(name) {
                    out.push(violation(Kind::Missing, &child(at, name), "required"));
                }
            }
        }
        for (trigger, needed) in schema.get("dependentRequired").map(entries).unwrap_or_default() {
            if let (true, Json::Arr(needed)) = (has(trigger.as_str()), needed) {
                for name in needed.iter().filter_map(Json::as_str) {
                    if !has(name) {
                        out.push(violation(Kind::Missing, &child(at, name), "dependentRequired"));
                    }
                }
            }
        }
        if schema.get("minProperties").and_then(count).is_some_and(|min| members.len() < min) {
            out.push(violation(Kind::Constraint, at, "minProperties"));
        }
        if schema.get("maxProperties").and_then(count).is_some_and(|max| members.len() > max) {
            out.push(violation(Kind::Constraint, at, "maxProperties"));
        }
    }
}

/// Of the members' violations (none empty), the closest member's: one whose
/// type admitted the value before one that did not, then the fewest, then
/// the first.
fn closest(found: Vec<Vec<Violation>>, at: &str) -> Vec<Violation> {
    found
        .into_iter()
        .min_by_key(|f| (f.iter().any(|v| v.kind == Kind::Type && v.at == at), f.len()))
        .unwrap_or_default()
}

/// A non-negative integer keyword's value (`minLength`, `maxItems`, ...).
fn count(value: &Json) -> Option<usize> {
    let d = number(value)?;
    if d.sign() < 0 {
        return Some(0);
    }
    d.to_integer().map(|n| usize::try_from(n).unwrap_or(usize::MAX))
}

/// Whether `value` has one of `types` (a name or a list of names). An
/// unknown name admits nothing; a list of none but unknown names, anything.
fn admits_type(types: &Json, value: &Json) -> bool {
    let one = |name: &str| -> Option<bool> {
        Some(match name {
            "null" => matches!(value, Json::Null),
            "boolean" => matches!(value, Json::Bool(_)),
            "object" => matches!(value, Json::Obj(_)),
            "array" => matches!(value, Json::Arr(_)),
            "string" => matches!(value, Json::Str(_)),
            "number" => matches!(value, Json::Num(_)),
            "integer" => value.number_text().is_some_and(|t| !t.contains(['.', 'e', 'E'])),
            _ => return None,
        })
    };
    let names: Vec<&str> = match types {
        Json::Str(name) => vec![name.as_str()],
        Json::Arr(list) => list.iter().filter_map(Json::as_str).collect(),
        _ => return true,
    };
    let known: Vec<bool> = names.iter().filter_map(|name| one(name)).collect();
    known.is_empty() || known.into_iter().any(|admits| admits)
}

/// JSON equality: numbers by value (`1` equals `1.0`), objects whatever
/// their key order, a boolean never a number.
fn equal(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Null, Json::Null) => true,
        (Json::Bool(x), Json::Bool(y)) => x == y,
        (Json::Num(x), Json::Num(y)) => match (Dec::parse(x), Dec::parse(y)) {
            (Some(x), Some(y)) => x == y,
            _ => x == y,
        },
        (Json::Str(x), Json::Str(y)) => x == y,
        (Json::Arr(x), Json::Arr(y)) => x.len() == y.len() && x.iter().zip(y).all(|(x, y)| equal(x, y)),
        (Json::Obj(x), Json::Obj(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.iter().any(|(k2, v2)| k == k2 && equal(v, v2)))
        }
        _ => false,
    }
}

/// The subschema a `$ref` names inside the schema itself: `#` or a JSON
/// pointer fragment from the root (`#/$defs/node`); `None` for any other
/// reference.
fn resolve<'a>(root: &'a Json, reference: &str) -> Option<&'a Json> {
    let pointer = reference.strip_prefix('#')?;
    if pointer.is_empty() {
        return Some(root);
    }
    let mut at = root;
    for token in pointer.strip_prefix('/')?.split('/') {
        let token = crate::json::unescape_pointer_token(&percent_decode(token)?);
        at = match at {
            Json::Obj(_) => at.get(&token)?,
            Json::Arr(items) => items.get(token.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(at)
}

/// A URI fragment's `%XX` escapes decoded.
fn percent_decode(token: &str) -> Option<String> {
    if !token.contains('%') {
        return Some(token.to_owned());
    }
    let bytes = token.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = std::str::from_utf8(bytes.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Whether ECMA-262 `pattern` finds a match anywhere in `s`; `None` when the
/// pattern is not a regular expression this validator reads.
fn search(pattern: &str, s: &str) -> Option<bool> {
    let translated = crate::regex::translate_ecma(pattern).ok()?;
    let regex = regex_automata::meta::Regex::new(&translated).ok()?;
    Some(regex.is_match(s))
}

/// An exact decimal: `digits` × 10^`exp`, negative when `neg`, with no
/// leading or trailing zero digit (zero has no digits and is not negative).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Dec {
    neg: bool,
    digits: Vec<u8>,
    exp: i64,
}

/// The furthest an exponent is read: beyond it every number compares the
/// same, and no real schema or argument comes near.
const MAX_EXP: i64 = 1 << 40;

impl Dec {
    /// A JSON number's text, exactly.
    fn parse(text: &str) -> Option<Dec> {
        let (neg, rest) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        let (mantissa, exp) = match rest.find(['e', 'E']) {
            Some(at) => {
                let e = &rest[at + 1..];
                let (sign, digits) = match e.strip_prefix('-') {
                    Some(d) => (-1, d),
                    None => (1, e.strip_prefix('+').unwrap_or(e)),
                };
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                let magnitude = digits.bytes().try_fold(0i64, |n, b| n.checked_mul(10)?.checked_add(i64::from(b - b'0')));
                (&rest[..at], sign * magnitude.unwrap_or(MAX_EXP).min(MAX_EXP))
            }
            None => (rest, 0),
        };
        let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if int.is_empty() || !int.bytes().chain(frac.bytes()).all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut digits: Vec<u8> = int.bytes().chain(frac.bytes()).map(|b| b - b'0').collect();
        let mut exp = exp - frac.len() as i64;
        let lead = digits.iter().take_while(|&&d| d == 0).count();
        digits.drain(..lead);
        while digits.last() == Some(&0) {
            digits.pop();
            exp += 1;
        }
        if digits.is_empty() {
            return Some(Dec { neg: false, digits, exp: 0 });
        }
        Some(Dec { neg, digits, exp })
    }

    fn sign(&self) -> i8 {
        match (self.digits.is_empty(), self.neg) {
            (true, _) => 0,
            (false, true) => -1,
            (false, false) => 1,
        }
    }

    /// The magnitudes compared.
    fn cmp_magnitude(&self, other: &Dec) -> Ordering {
        // The position of each leading digit decides, then the digits.
        let lead = |d: &Dec| d.digits.len() as i64 + d.exp;
        match (self.digits.is_empty(), other.digits.is_empty()) {
            (true, true) => return Ordering::Equal,
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        lead(self).cmp(&lead(other)).then_with(|| {
            let n = self.digits.len().max(other.digits.len());
            let at = |d: &Dec, i: usize| d.digits.get(i).copied().unwrap_or(0);
            (0..n).map(|i| at(self, i).cmp(&at(other, i))).find(|o| o.is_ne()).unwrap_or(Ordering::Equal)
        })
    }

    /// The value as an integer, when it is one that fits.
    fn to_integer(&self) -> Option<i128> {
        if self.exp < 0 {
            return None;
        }
        let mut n: i128 = 0;
        for &d in &self.digits {
            n = n.checked_mul(10)?.checked_add(i128::from(d))?;
        }
        for _ in 0..self.exp {
            n = n.checked_mul(10)?;
        }
        Some(if self.neg { -n } else { n })
    }

    /// Whether `self / m` is an integer: exact while `m`'s digits, aligned to
    /// the smaller exponent, fit 37 decimal digits and `self`'s are fewer
    /// than about ten thousand; past that, in floating point, as most
    /// validators run (`text` and `m_text` are the two numbers as written).
    fn is_multiple_of(&self, m: &Dec, text: &str, m_text: &str) -> bool {
        if self.digits.is_empty() {
            return true;
        }
        let e = self.exp.min(m.exp);
        let (pad_v, pad_m) = (self.exp - e, m.exp - e);
        if m.digits.len() as i64 + pad_m <= 37 && pad_v <= 10_000 {
            let modulus = m.digits.iter().fold(0u128, |n, &d| n * 10 + u128::from(d)) * 10u128.pow(pad_m as u32);
            let mut r = 0u128;
            for d in self.digits.iter().copied().chain(std::iter::repeat(0).take(pad_v as usize)) {
                r = (r * 10 + u128::from(d)) % modulus;
            }
            return r == 0;
        }
        match (text.parse::<f64>(), m_text.parse::<f64>()) {
            (Ok(v), Ok(m)) => {
                let q = v / m;
                q.is_finite() && q.fract() == 0.0
            }
            _ => false,
        }
    }
}

impl PartialOrd for Dec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dec {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.sign().cmp(&other.sign()) {
            Ordering::Equal if self.sign() < 0 => other.cmp_magnitude(self),
            Ordering::Equal => self.cmp_magnitude(other),
            unequal => unequal,
        }
    }
}

/// The nine formats, each written from its RFC (and, where the RFC leaves a
/// choice, as the independent validator reads it): `None` for a format this
/// validator does not check.
mod formats {
    pub(super) fn check(format: &str, s: &str) -> Option<bool> {
        Some(match format {
            "date" => date(s),
            "time" => time(&s.to_ascii_uppercase()),
            "date-time" => date_time(&s.to_ascii_uppercase()),
            "duration" => duration(s),
            "email" => email(s),
            "hostname" => hostname(s),
            "ipv4" => ipv4(s),
            "ipv6" => ipv6(s),
            "uuid" => uuid(s),
            _ => return None,
        })
    }

    /// `n` ASCII digits read as a number.
    fn digits(s: &str) -> Option<u32> {
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok()).flatten()
    }

    fn days_in_month(year: u32, month: u32) -> u32 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            _ if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
            _ => 28,
        }
    }

    /// RFC 3339 `full-date`: `YYYY-MM-DD`, a day of the calendar, year 0
    /// excluded.
    fn date(s: &str) -> bool {
        let b = s.as_bytes();
        if !s.is_ascii() || b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
            return false;
        }
        let (Some(year), Some(month), Some(day)) = (digits(&s[0..4]), digits(&s[5..7]), digits(&s[8..10])) else {
            return false;
        };
        year >= 1 && (1..=12).contains(&month) && day >= 1 && day <= days_in_month(year, month)
    }

    /// RFC 3339 `full-time` (upper case): `HH:MM:SS`, an optional fraction,
    /// then `Z` or an offset `±HH:MM`. No leap second.
    fn time(s: &str) -> bool {
        let b = s.as_bytes();
        if !s.is_ascii() || b.len() < 9 || b[2] != b':' || b[5] != b':' {
            return false;
        }
        let (Some(h), Some(m), Some(sec)) = (digits(&s[0..2]), digits(&s[3..5]), digits(&s[6..8])) else {
            return false;
        };
        if h > 23 || m > 59 || sec > 59 {
            return false;
        }
        let mut rest = &s[8..];
        if let Some(frac) = rest.strip_prefix('.') {
            let n = frac.bytes().take_while(u8::is_ascii_digit).count();
            if n == 0 {
                return false;
            }
            rest = &frac[n..];
        }
        if rest == "Z" {
            return true;
        }
        let Some(offset) = rest.strip_prefix(['+', '-']) else { return false };
        let ob = offset.as_bytes();
        if ob.len() != 5 || ob[2] != b':' {
            return false;
        }
        matches!((digits(&offset[0..2]), digits(&offset[3..5])), (Some(oh), Some(om)) if oh <= 23 && om <= 59)
    }

    /// RFC 3339 `date-time` (upper case): a date, `T`, a time.
    fn date_time(s: &str) -> bool {
        s.is_ascii() && s.len() > 11 && date(&s[..10]) && s.as_bytes()[10] == b'T' && time(&s[11..])
    }

    /// RFC 3339 appendix A: `P`, then weeks alone or date parts, then an
    /// optional `T` with time parts; units in order, none skipped between
    /// two that are used (no years then days), at least one on each side
    /// that is written.
    fn duration(s: &str) -> bool {
        let Some(body) = s.strip_prefix('P') else { return false };
        if body.is_empty() {
            return false;
        }
        /// The units used, in order, or `None`.
        fn parts(text: &str, units: &str) -> Option<Vec<usize>> {
            let mut used = Vec::new();
            let mut number = false;
            for c in text.chars() {
                if c.is_ascii_digit() {
                    number = true;
                } else if let (Some(unit), true) = (units.find(c), number) {
                    if used.last().is_some_and(|&last| last >= unit) {
                        return None;
                    }
                    used.push(unit);
                    number = false;
                } else {
                    return None;
                }
            }
            (!number).then_some(used)
        }
        let (date, time) = match body.split_once('T') {
            Some((date, time)) => (date, Some(time)),
            None => (body, None),
        };
        if !date.is_empty() {
            if date.ends_with('W') {
                if time.is_some() || parts(date, "W").as_deref() != Some(&[0]) {
                    return false;
                }
            } else {
                match parts(date, "YMD") {
                    Some(used) if !used.is_empty() && used != [0, 2] => {}
                    _ => return false,
                }
            }
        }
        if let Some(time) = time {
            match parts(time, "HMS") {
                Some(used) if !used.is_empty() && used != [0, 2] => {}
                _ => return false,
            }
        }
        true
    }

    /// RFC 5321 `Mailbox`: a local part, `@`, and a domain or an address
    /// literal. The local part is a quoted string, or `atext` and dots with
    /// the dot-string's placement left free (`.a`, `a..b`): what the grammar
    /// writes for `email`, which this validator must admit whole.
    fn email(s: &str) -> bool {
        let Some((local, domain)) = s.rsplit_once('@') else { return false };
        let atext = |c: char| c.is_ascii_alphanumeric() || "!#$%&'*+-/=?^_`{|}~".contains(c);
        let local_ok = if let Some(inner) = local.strip_prefix('"').and_then(|l| l.strip_suffix('"')) {
            let mut chars = inner.chars();
            let mut ok = true;
            while let Some(c) = chars.next() {
                ok &= match c {
                    '\\' => chars.next().is_some_and(|e| (' '..='~').contains(&e)),
                    '"' => false,
                    c => (' '..='~').contains(&c),
                };
            }
            ok
        } else {
            !local.is_empty() && local.chars().all(|c| c == '.' || atext(c))
        };
        let domain_ok = match domain.strip_prefix('[').and_then(|d| d.strip_suffix(']')) {
            Some(literal) => match literal.strip_prefix("IPv6:") {
                Some(v6) => ipv6(v6),
                None => ipv4(literal),
            },
            None => hostname(domain) && !domain.ends_with('.'),
        };
        local_ok && domain_ok
    }

    /// RFC 1123 host name: labels of letters, digits and inner hyphens, 1 to
    /// 63 characters each, 253 in all, a final dot allowed.
    fn hostname(s: &str) -> bool {
        let name = s.strip_suffix('.').unwrap_or(s);
        !name.is_empty()
            && name.len() <= 253
            && name.split('.').all(|label| {
                (1..=63).contains(&label.len())
                    && label.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                    && !label.starts_with('-')
                    && !label.ends_with('-')
            })
    }

    /// Four decimal octets, no leading zero.
    fn ipv4(s: &str) -> bool {
        let octets: Vec<&str> = s.split('.').collect();
        octets.len() == 4
            && octets.iter().all(|o| {
                (1..=3).contains(&o.len()) && (o.len() == 1 || !o.starts_with('0')) && digits(o).is_some_and(|n| n <= 255)
            })
    }

    /// RFC 4291 text form: eight groups of one to four hex digits, the last
    /// two of which may be an IPv4 address, and one `::` for at least one
    /// group of zeros. No zone.
    fn ipv6(s: &str) -> bool {
        let group = |g: &str| (1..=4).contains(&g.len()) && g.bytes().all(|b| b.is_ascii_hexdigit());
        let count = |part: &str, last: bool| -> Option<usize> {
            if part.is_empty() {
                return Some(0);
            }
            let groups: Vec<&str> = part.split(':').collect();
            let mut n = 0;
            for (i, g) in groups.iter().enumerate() {
                if last && i + 1 == groups.len() && g.contains('.') {
                    if !ipv4(g) {
                        return None;
                    }
                    n += 2;
                } else if group(g) {
                    n += 1;
                } else {
                    return None;
                }
            }
            Some(n)
        };
        match s.split_once("::") {
            Some((head, tail)) => {
                if tail.contains("::") {
                    return false;
                }
                // An IPv4 address ends the whole address, so never comes
                // before the `::`.
                matches!((count(head, false), count(tail, true)), (Some(h), Some(t)) if h + t < 8)
            }
            None => count(s, true) == Some(8),
        }
    }

    /// RFC 4122 text form: 8-4-4-4-12 hex digits.
    fn uuid(s: &str) -> bool {
        let b = s.as_bytes();
        b.len() == 36
            && b.iter().enumerate().all(|(i, &c)| match i {
                8 | 13 | 18 | 23 => c == b'-',
                _ => c.is_ascii_hexdigit(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(schema: &str, value: &str) -> Vec<Kind> {
        let mut k: Vec<Kind> =
            violations(&Json::parse(schema).unwrap(), &Json::parse(value).unwrap()).into_iter().map(|v| v.kind).collect();
        k.sort();
        k.dedup();
        k
    }

    #[test]
    fn a_violation_says_where_and_which_keyword() {
        let schema = Json::parse(r#"{"type":"object","properties":{"a":{"type":"array","items":{"type":"integer"}}},"additionalProperties":false}"#).unwrap();
        let found = violations(&schema, &Json::parse(r#"{"a":[1,"x"],"b/c":1}"#).unwrap());
        assert_eq!(
            found,
            vec![violation(Kind::Type, "/a/1", "type"), violation(Kind::UnknownParameter, "/b~1c", "additionalProperties")]
        );
    }

    #[test]
    fn a_type_failure_hides_the_rest_of_its_schema() {
        assert_eq!(kinds(r#"{"type":"integer","multipleOf":5,"enum":[5]}"#, "2.5"), vec![Kind::Type]);
        assert_eq!(kinds(r#"{"type":["string","null"],"enum":["a",null]}"#, "\"b\""), vec![Kind::Enum]);
    }

    #[test]
    fn a_ref_cycle_that_never_descends_stops_unchecked() {
        assert!(is_valid(&Json::parse(r##"{"$ref":"#"}"##).unwrap(), &Json::parse("1").unwrap()));
        let looped = r##"{"$defs":{"a":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a","type":"string"}"##;
        assert_eq!(kinds(looped, "1"), vec![Kind::Type]);
    }

    #[test]
    fn decimals_compare_and_divide_exactly() {
        let d = |t: &str| Dec::parse(t).unwrap();
        assert_eq!(d("1.0"), d("1"));
        assert_eq!(d("-0.0"), d("0"));
        assert_eq!(d("1e2"), d("100"));
        assert!(d("-2") < d("-1.5") && d("-1.5") < d("0") && d("0.001") < d("1e-2"));
        assert!(d("12345678901234567890123") > d("1.2345678901234567890122e22"));
        assert!(d("0.3").is_multiple_of(&d("0.1"), "0.3", "0.1"));
        assert!(!d("0.35").is_multiple_of(&d("0.1"), "0.35", "0.1"));
        assert!(d("1e30").is_multiple_of(&d("0.001"), "1e30", "0.001"));
    }

    #[test]
    fn a_pattern_ignis_cannot_read_is_not_checked() {
        assert!(is_valid(&Json::parse(r#"{"pattern":"(a)\\1"}"#).unwrap(), &Json::parse("\"xyz\"").unwrap()));
        assert!(!is_valid(&Json::parse(r#"{"pattern":"^a"}"#).unwrap(), &Json::parse("\"xyz\"").unwrap()));
    }

    /// The oracle admits everything the grammar writes (what the property
    /// tests of #611 and #612 build on): the documents of random walks over
    /// grammars of every format and of the keywords the strict subset holds,
    /// each valid here.
    #[test]
    fn every_document_the_grammar_writes_is_valid_here() {
        use crate::{random_document, Grammar, Vocab, VocabEntry};
        let vocab = Vocab::new((0..=255u8).map(|b| VocabEntry::Text(vec![b])).chain([VocabEntry::Control]).collect(), &[256]);
        let object = |properties: &str| {
            let names: Vec<String> = Json::parse(properties)
                .unwrap()
                .get("x")
                .map(|_| vec!["\"x\"".to_owned()])
                .unwrap_or_default();
            format!(r#"{{"type":"object","properties":{properties},"required":[{}],"additionalProperties":false}}"#, names.join(","))
        };
        let mut schemas: Vec<String> = crate::FORMATS
            .iter()
            .map(|format| object(&format!(r#"{{"x":{{"type":"string","format":"{format}"}}}}"#)))
            .collect();
        schemas.extend([
            object(r#"{"x":{"type":"string","pattern":"^[a-z]+-[0-9]{3}$"}}"#),
            object(r#"{"x":{"type":"string","pattern":"b"}}"#),
            object(r#"{"x":{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}}"#),
            object(r#"{"x":{"type":"number","multipleOf":0.25,"minimum":-3,"maximum":3}}"#),
            object(r#"{"x":{"type":"integer","multipleOf":5,"minimum":-100}}"#),
            object(r#"{"x":{"type":["string","null"],"enum":["a","b",null]}}"#),
            object(r#"{"x":{"anyOf":[{"type":"integer"},{"type":"boolean"},{"type":"null"}]}}"#),
            object(r#"{"x":{"type":"array","minItems":1,"maxItems":3,"items":{"type":"object","properties":{"old":{"type":"string"},"new":{"type":"string"}},"required":["old","new"],"additionalProperties":false}}}"#),
            r##"{"type":"object","properties":{"x":{"$ref":"#/$defs/node"}},"required":["x"],"additionalProperties":false,"$defs":{"node":{"type":"object","properties":{"name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["name","children"],"additionalProperties":false}}}"##.to_owned(),
        ]);
        let mut checked = 0;
        for schema in &schemas {
            let grammar = Grammar::json_schema(vocab.clone(), schema, true).unwrap_or_else(|r| panic!("{schema}: {r}"));
            let parsed = Json::parse(schema).unwrap();
            for seed in 0..200 {
                let Some(doc) = random_document(&grammar, seed, 40, 4000).unwrap() else { continue };
                let text = String::from_utf8(doc).unwrap();
                let value = Json::parse(&text).unwrap_or_else(|e| panic!("{schema}: {text}: {e}"));
                let found = violations(&parsed, &value);
                assert!(found.is_empty(), "the grammar of {schema} wrote {text}, which breaks {found:?}");
                checked += 1;
            }
        }
        assert!(checked > 3000, "{checked} documents checked");
    }

    #[test]
    fn the_formats_hold_their_rfcs() {
        let f = |format: &str, s: &str| formats::check(format, s).unwrap();
        assert!(f("email", "\"a b\"@example.com") && f("email", "a@[192.168.0.1]") && f("email", "a@[IPv6:2001:db8::1]"));
        assert!(!f("email", "a@b@c") && !f("email", "a@example.com.") && !f("email", "@example.com"));
        // The grammar's own email admits dots anywhere in the local part.
        assert!(f("email", ".a@example.com") && f("email", "a..b.@example.com"));
        assert!(f("ipv6", "1:2:3:4:5:6:1.2.3.4") && f("ipv6", "::1:2:3:4:5:6:7") && !f("ipv6", "1:2::3:4:5:6:7:8"));
        assert!(!f("ipv6", ":1::2") && !f("ipv6", "1::2:") && !f("ipv6", "00000::1") && !f("ipv6", "::1.2.3"));
        assert!(f("duration", "PT0S") && !f("duration", "P1W2D") && !f("duration", "P1D2Y") && !f("duration", "PT1H5S"));
        assert!(f("hostname", "123") && !f("hostname", &"a".repeat(64)) && !f("hostname", "a_b.example"));
        assert!(f("date", "0001-01-01") && !f("date", "0000-01-01") && !f("date", "2026-1-01"));
        assert_eq!(formats::check("uri", "not a uri"), None);
    }
}
