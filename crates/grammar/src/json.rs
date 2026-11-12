//! A JSON value that keeps what a schema means exactly:
//! - an object's key order: a strict schema's `properties` order is the order
//!   the answer writes its keys in, and `serde_json::Value` without
//!   `preserve_order` (which this workspace must not turn on: feature
//!   unification would reorder every other crate's maps) sorts them;
//! - a number's text: a bound or a `multipleOf` is held in exact decimal
//!   arithmetic, and a double would round one with more than 17 significant
//!   digits.
//!
//! Hence its own parser, small and strict (RFC 8259, nesting at most
//! [`MAX_DEPTH`]).

use crate::number::{format_scaled, parse_scaled};

/// The deepest a parsed document may nest (the parser's own recursion).
pub const MAX_DEPTH: usize = 128;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// A number as written (valid JSON number syntax).
    Num(String),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(text: &str) -> Result<Json, String> {
        let mut p = Parser { b: text.as_bytes(), i: 0 };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(p.err("trailing characters after the value"));
        }
        Ok(v)
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    /// A number's text, exactly as written.
    pub fn number_text(&self) -> Option<&str> {
        match self {
            Json::Num(t) => Some(t),
            _ => None,
        }
    }

    /// The canonical encoding a literal (`enum`, `const`, a key) is written
    /// in: no whitespace, serde_json's string escapes, a number in plain
    /// decimal without trailing zeros (`1.0` and `1e0` are `1`) when it fits
    /// the exact arithmetic, as written otherwise. The grammar accepts only
    /// this spelling of a literal.
    pub fn write_canonical(&self, out: &mut Vec<u8>) {
        match self {
            Json::Null => out.extend_from_slice(b"null"),
            Json::Bool(true) => out.extend_from_slice(b"true"),
            Json::Bool(false) => out.extend_from_slice(b"false"),
            Json::Num(t) => match parse_scaled(t) {
                Some(v) => out.extend_from_slice(format_scaled(v).as_bytes()),
                None => out.extend_from_slice(t.as_bytes()),
            },
            Json::Str(s) => write_string(s, out),
            Json::Arr(items) => {
                out.push(b'[');
                for (i, x) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    x.write_canonical(out);
                }
                out.push(b']');
            }
            Json::Obj(entries) => {
                out.push(b'{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    write_string(k, out);
                    out.push(b':');
                    v.write_canonical(out);
                }
                out.push(b'}');
            }
        }
    }
}

impl Json {
    /// Pretty-printed with two-space indents, keys in order, leaves in their
    /// canonical spelling: the shape a model writes a document in (the bench
    /// and the tests replay both this and the compact form).
    pub fn pretty(&self) -> String {
        fn go(j: &Json, indent: usize, out: &mut String) {
            let pad = |n: usize| " ".repeat(n);
            match j {
                Json::Obj(e) if !e.is_empty() => {
                    out.push_str("{\n");
                    for (i, (k, v)) in e.iter().enumerate() {
                        let mut key = Vec::new();
                        write_string(k, &mut key);
                        out.push_str(&pad(indent + 2));
                        out.push_str(std::str::from_utf8(&key).unwrap());
                        out.push_str(": ");
                        go(v, indent + 2, out);
                        out.push_str(if i + 1 < e.len() { ",\n" } else { "\n" });
                    }
                    out.push_str(&pad(indent));
                    out.push('}');
                }
                Json::Arr(a) if !a.is_empty() => {
                    out.push_str("[\n");
                    for (i, v) in a.iter().enumerate() {
                        out.push_str(&pad(indent + 2));
                        go(v, indent + 2, out);
                        out.push_str(if i + 1 < a.len() { ",\n" } else { "\n" });
                    }
                    out.push_str(&pad(indent));
                    out.push(']');
                }
                other => {
                    let mut b = Vec::new();
                    other.write_canonical(&mut b);
                    out.push_str(std::str::from_utf8(&b).unwrap());
                }
            }
        }
        let mut out = String::new();
        go(self, 0, &mut out);
        out
    }

    /// The canonical encoding as a string.
    pub fn compact(&self) -> String {
        let mut b = Vec::new();
        self.write_canonical(&mut b);
        String::from_utf8(b).unwrap()
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("{what} at byte {}", self.i)
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &[u8]) -> bool {
        if self.b[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth >= MAX_DEPTH {
            return Err(self.err(&format!("nesting deeper than {MAX_DEPTH}")));
        }
        match self.peek() {
            Some(b'{') => {
                self.i += 1;
                let mut entries = Vec::new();
                self.ws();
                if self.eat(b"}") {
                    return Ok(Json::Obj(entries));
                }
                loop {
                    self.ws();
                    if self.peek() != Some(b'"') {
                        return Err(self.err("expected a key"));
                    }
                    let k = self.string()?;
                    self.ws();
                    if !self.eat(b":") {
                        return Err(self.err("expected ':'"));
                    }
                    self.ws();
                    let v = self.value(depth + 1)?;
                    entries.push((k, v));
                    self.ws();
                    if self.eat(b",") {
                        continue;
                    }
                    if self.eat(b"}") {
                        return Ok(Json::Obj(entries));
                    }
                    return Err(self.err("expected ',' or '}'"));
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.eat(b"]") {
                    return Ok(Json::Arr(items));
                }
                loop {
                    self.ws();
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    if self.eat(b",") {
                        continue;
                    }
                    if self.eat(b"]") {
                        return Ok(Json::Arr(items));
                    }
                    return Err(self.err("expected ',' or ']'"));
                }
            }
            Some(b'"') => self.string().map(Json::Str),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ if self.eat(b"true") => Ok(Json::Bool(true)),
            _ if self.eat(b"false") => Ok(Json::Bool(false)),
            _ if self.eat(b"null") => Ok(Json::Null),
            _ => Err(self.err("expected a value")),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        self.i - start
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        self.eat(b"-");
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.err("expected a digit")),
        }
        if self.eat(b".") && self.digits() == 0 {
            return Err(self.err("expected a fraction digit"));
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                return Err(self.err("expected an exponent digit"));
            }
        }
        Ok(Json::Num(String::from_utf8(self.b[start..self.i].to_vec()).unwrap()))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u32::from_str_radix(h, 16).ok());
        let Some(v) = h else { return Err(self.err("expected four hex digits")) };
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out = Vec::new();
        loop {
            let Some(c) = self.peek() else { return Err(self.err("unterminated string")) };
            self.i += 1;
            match c {
                b'"' => return Ok(String::from_utf8(out).expect("the input is UTF-8 and escapes decode to chars")),
                b'\\' => {
                    let Some(e) = self.peek() else { return Err(self.err("unterminated escape")) };
                    self.i += 1;
                    let ch = match e {
                        b'"' | b'\\' | b'/' => e as char,
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..=0xDBFF).contains(&hi) {
                                if !self.eat(b"\\u") {
                                    return Err(self.err("a high surrogate without its low surrogate"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&lo) {
                                    return Err(self.err("a high surrogate without its low surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(cp).ok_or_else(|| self.err("a lone low surrogate"))?
                        }
                        _ => return Err(self.err("an unknown escape")),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                0x00..=0x1F => return Err(self.err("a control character in a string")),
                _ => out.push(c),
            }
        }
    }
}

/// serde_json's string escaping.
pub fn write_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for &b in s.as_bytes() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0c => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => out.extend_from_slice(format!("\\u{b:04x}").as_bytes()),
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical(text: &str) -> String {
        let mut out = Vec::new();
        Json::parse(text).unwrap().write_canonical(&mut out);
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn objects_keep_their_key_order() {
        assert_eq!(canonical(r#"{"zeta":1,"alpha":{"b":2,"a":[true,null,"x\n"]}}"#), r#"{"zeta":1,"alpha":{"b":2,"a":[true,null,"x\n"]}}"#);
    }

    #[test]
    fn numbers_keep_every_digit_written() {
        let j = Json::parse("[1234567890123456789.123456789, 1.0, 1e2, -0.50, 2.5E-3, 1e400]").unwrap();
        let Json::Arr(xs) = &j else { panic!() };
        assert_eq!(xs[0].number_text(), Some("1234567890123456789.123456789"));
        assert_eq!(canonical("[1234567890.123456789, 1.0, 1e2, -0.50, 2.5E-3, 1e400]"), "[1234567890.123456789,1,100,-0.5,0.0025,1e400]");
    }

    #[test]
    fn what_is_not_json_is_refused() {
        for bad in ["", "{", "[1,]", "{'a':1}", "01", "1.", "-", "\"\\x\"", "\"\\ud800\"", "\"\\udc00\"", "\"a\u{1}\"", "[1] x", "NaN", "{\"a\" 1}"] {
            assert!(Json::parse(bad).is_err(), "{bad:?}");
        }
        assert_eq!(Json::parse(r#""\ud83d\ude00 \u00e9""#).unwrap(), Json::Str("\u{1F600} \u{e9}".into()));
        let deep = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1));
        assert!(Json::parse(&deep).unwrap_err().contains("nesting"));
    }
}
