//! Where a value of a tool call ends (GitHub #610, spec #582, ADR 0094): the
//! one rule, over bytes, that the call parser and a **held tool**'s grammar
//! share, so that what the grammar accepts is what the parser delivers.
//!
//! The model writes its calls in the Qwen XML dialect, which has no escape
//! for its own tags:
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=K>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! A tool that is not held keeps the parser's first-occurrence cuts. For a
//! held tool a value ends only where the call continues: a **line** of the
//! value is a [`Step`] of the call only when it is a valid next step under
//! the tool's [`Parameters`] ([`line`]):
//! - `<parameter=K>`, for a K the tool declares and the call has not written;
//! - or `</function>` followed by `</tool_call>`, once every required
//!   parameter is written (the confirmed close);
//!
//! each after an optional `</parameter>`, the closer the model sometimes
//! forgets. Anything else is the value's text: a `</parameter>` line followed
//! by more text, an opener of a parameter already written or never declared,
//! a `</function>` line not followed by `</tool_call>`, and a close written
//! while a required parameter is missing. The residue: a value cannot hold
//! its own call's confirmed close.
//!
//! What the rule reads as a line, and how it reads the tags:
//! - A line starts at the value's first byte and after each `\n`; a tag
//!   counts only at a line's first byte, so a tag after spaces is text.
//! - Between the tags of one step, any run of spaces, tabs, `\r` and `\n`
//!   (none included): the 27B's tokenizer merges newlines with the
//!   whitespace beside them, so the rule reads bytes, never tokens.
//! - A parameter's name runs to the first `>`, as the parser reads it: a
//!   declared name holding a `>` is never opened, and a required parameter
//!   the tool does not declare is never written, so its call never closes.

/// The dialect's tags.
pub const CALL_OPEN: &str = "<tool_call>";
pub const CALL_CLOSE: &str = "</tool_call>";
pub const FUNCTION_OPEN: &str = "<function=";
pub const FUNCTION_CLOSE: &str = "</function>";
pub const PARAMETER_OPEN: &str = "<parameter=";
pub const PARAMETER_CLOSE: &str = "</parameter>";

/// What the rule knows of a tool: the parameters it declares, and those of
/// them a call writes before it may close. The caller says which are
/// required (a strict tool's nullable parameters may be omitted, for one).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Parameters {
    declared: Vec<String>,
    /// Each required name's place in `declared`, or `None` for a name the
    /// tool does not declare, which no call can write.
    required: Vec<Option<usize>>,
}

impl Parameters {
    /// The parameters `declared` (a name given twice counts once) and the
    /// names among them a call must write before it closes.
    pub fn new<D, R>(declared: D, required: R) -> Self
    where
        D: IntoIterator,
        D::Item: Into<String>,
        R: IntoIterator,
        R::Item: AsRef<str>,
    {
        let mut names: Vec<String> = Vec::new();
        for name in declared {
            let name = name.into();
            if !names.contains(&name) {
                names.push(name);
            }
        }
        let required = required.into_iter().map(|name| names.iter().position(|d| d == name.as_ref())).collect();
        Self { declared: names, required }
    }

    /// The declared parameters: what [`Step::Parameter`]'s `index` and the
    /// `written` flags of [`line`] point into.
    pub fn declared(&self) -> &[String] {
        &self.declared
    }

    /// Where `name` is among the declared parameters.
    pub fn index(&self, name: &str) -> Option<usize> {
        self.declared.iter().position(|d| d == name)
    }

    /// Whether a call that wrote `written` may close: every required
    /// parameter written.
    pub fn may_close(&self, written: &[bool]) -> bool {
        self.required.iter().all(|place| place.is_some_and(|i| written.get(i) == Some(&true)))
    }

    /// The declared, unwritten parameters an opener may name.
    fn openable<'a>(&'a self, written: &'a [bool]) -> impl Iterator<Item = (usize, &'a str)> + 'a {
        self.declared
            .iter()
            .enumerate()
            .filter(move |(i, name)| written.get(*i) != Some(&true) && !name.contains('>'))
            .map(|(i, name)| (i, name.as_str()))
    }
}

/// A valid next step of a call, read at a line's start: where the value
/// that line is in ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// `<parameter=K>`, the opener of the declared, unwritten parameter at
    /// `index`, after an optional `</parameter>`: K's value starts `len`
    /// bytes after the line's start.
    Parameter { index: usize, len: usize },
    /// `</function>` and `</tool_call>`, every required parameter written,
    /// after an optional `</parameter>`: the value, the function body and
    /// the call end at the line's start, and the call's closer ends `len`
    /// bytes after it.
    Close { len: usize },
}

/// What a line of a value is, read from its first byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Line {
    /// The value's text: no bytes that follow make it a step.
    Text,
    /// Not decided yet: some bytes that follow would make it a step, and
    /// others would leave it text.
    Pending,
    /// A step: what follows its `len` bytes does not change it.
    Step(Step),
}

/// The rule: what the line starting at `line[0]` is, in a value of a call to
/// a tool with these `parameters` that has written the parameters `written`
/// flags (by their place in [`Parameters::declared`]; a missing flag is
/// unwritten). `line` is every byte the caller has from the line's start,
/// the next lines included: a step may span lines. Once a line is text or a
/// step, more bytes never change it.
pub fn line(line: &[u8], parameters: &Parameters, written: &[bool]) -> Line {
    let opens = parameters.openable(written).next().is_some();
    let closes = parameters.may_close(written);
    let mut answer = Line::Text;
    if !opens && !closes {
        return answer;
    }
    // The step's own tags at the line's start, then after a forgotten
    // `</parameter>` and its whitespace.
    for forgotten in [false, true] {
        let mut at = Cursor { bytes: line, at: 0 };
        if forgotten {
            match at.literal(PARAMETER_CLOSE) {
                Lit::Whole => at.whitespace(),
                Lit::Short => return Line::Pending,
                Lit::No => break,
            }
        }
        let reads = [opens.then(|| at.clone().opener(parameters, written)), closes.then(|| at.clone().close())];
        for read in reads.into_iter().flatten() {
            match read {
                Read::Step(step) => return Line::Step(step),
                Read::Short => answer = Line::Pending,
                Read::No => {}
            }
        }
    }
    answer
}

/// Where a value that starts at `value[0]` ends under the rule: the first
/// line start in it (its first byte, then each byte after a `\n`) whose
/// [`line`] is a step, with that step; `None` while none is, a line still
/// [`Line::Pending`] at the end of `value` included.
pub fn value_end(value: &[u8], parameters: &Parameters, written: &[bool]) -> Option<(usize, Step)> {
    let mut at = 0;
    loop {
        if let Line::Step(step) = line(&value[at..], parameters, written) {
            return Some((at, step));
        }
        at += value[at..].iter().position(|&b| b == b'\n')? + 1;
    }
}

/// One step's reading so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    /// The bytes break it.
    No,
    /// The bytes ran out before it was whole: more could complete it.
    Short,
    Step(Step),
}

/// One tag's reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lit {
    No,
    Short,
    Whole,
}

#[derive(Clone)]
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn rest(&self) -> &[u8] {
        &self.bytes[self.at..]
    }

    /// `tag` next: read past it when whole.
    fn literal(&mut self, tag: &str) -> Lit {
        let rest = self.rest();
        let n = rest.len().min(tag.len());
        if rest[..n] != tag.as_bytes()[..n] {
            return Lit::No;
        }
        if n < tag.len() {
            return Lit::Short;
        }
        self.at += n;
        Lit::Whole
    }

    fn whitespace(&mut self) {
        self.at += self.rest().iter().take_while(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n')).count();
    }

    /// `<parameter=K>` for an openable K.
    fn opener(mut self, parameters: &Parameters, written: &[bool]) -> Read {
        match self.literal(PARAMETER_OPEN) {
            Lit::Whole => {}
            Lit::Short => return Read::Short,
            Lit::No => return Read::No,
        }
        let rest = self.rest();
        let Some(end) = rest.iter().position(|&b| b == b'>') else {
            let prefix = parameters.openable(written).any(|(_, name)| name.as_bytes().starts_with(rest));
            return if prefix { Read::Short } else { Read::No };
        };
        match parameters.openable(written).find(|(_, declared)| declared.as_bytes() == &rest[..end]) {
            Some((index, _)) => Read::Step(Step::Parameter { index, len: self.at + end + 1 }),
            None => Read::No,
        }
    }

    /// `</function>`, whitespace, `</tool_call>`.
    fn close(mut self) -> Read {
        match self.literal(FUNCTION_CLOSE) {
            Lit::Whole => {}
            Lit::Short => return Read::Short,
            Lit::No => return Read::No,
        }
        self.whitespace();
        match self.literal(CALL_CLOSE) {
            Lit::Whole => Read::Step(Step::Close { len: self.at }),
            Lit::Short => Read::Short,
            Lit::No => Read::No,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `write_file`: `path` and `content` required, `mode` optional.
    fn write_file() -> Parameters {
        Parameters::new(["path", "content", "mode"], ["path", "content"])
    }

    /// The `written` flags of these names.
    fn wrote(parameters: &Parameters, names: &[&str]) -> Vec<bool> {
        let mut written = vec![false; parameters.declared().len()];
        for name in names {
            written[parameters.index(name).unwrap()] = true;
        }
        written
    }

    fn read(text: &str, names: &[&str]) -> Line {
        let parameters = write_file();
        line(text.as_bytes(), &parameters, &wrote(&parameters, names))
    }

    fn opens(index: usize, text: &str) -> Line {
        Line::Step(Step::Parameter { index, len: text.len() })
    }

    fn closes(text: &str) -> Line {
        Line::Step(Step::Close { len: text.len() })
    }

    /// Where the value in `text` ends, the value having started at `text[0]`.
    fn end(text: &str, names: &[&str]) -> Option<(usize, Step)> {
        let parameters = write_file();
        value_end(text.as_bytes(), &parameters, &wrote(&parameters, names))
    }

    #[test]
    fn an_opener_of_a_declared_unwritten_parameter_is_a_step() {
        assert_eq!(read("<parameter=content>\nhello", &["path"]), opens(1, "<parameter=content>"));
        assert_eq!(read("<parameter=mode>", &["path", "content"]), opens(2, "<parameter=mode>"));
    }

    #[test]
    fn an_opener_of_a_written_or_undeclared_parameter_is_text() {
        assert_eq!(read("<parameter=path>\nb.txt\n", &["path"]), Line::Text);
        assert_eq!(read("<parameter=force>\ntrue\n", &["path"]), Line::Text);
        assert_eq!(read("<parameter=PATH>\n", &[]), Line::Text);
        assert_eq!(read("<parameter=path >\n", &[]), Line::Text);
        // In a value, so: a file about tool calls keeps its own openers.
        let value = "\n<parameter=path>\nb.txt\n</parameter>\n<parameter=mode>\nx";
        let step = "</parameter>\n<parameter=mode>";
        assert_eq!(end(value, &["path", "content"]), Some((value.find(step).unwrap(), Step::Parameter { index: 2, len: step.len() })));
    }

    #[test]
    fn a_value_keeps_a_parameter_closer_line_that_no_step_follows() {
        assert_eq!(read("</parameter>\nmore text\n", &["path"]), Line::Text);
        assert_eq!(read("</parameter>\n<parameter=path>\n", &["path"]), Line::Text);
        let value = "\nfirst\n</parameter>\nsecond\n</parameter>\n</function>\n</tool_call>";
        let at = value.rfind("</parameter>").unwrap();
        assert_eq!(end(value, &["path", "content"]), Some((at, Step::Close { len: value.len() - at })));
    }

    #[test]
    fn a_function_closer_not_followed_by_the_calls_closer_is_text() {
        assert_eq!(read("</function>\nmore\n</tool_call>", &["path", "content"]), Line::Text);
        assert_eq!(read("</function></parameter>", &["path", "content"]), Line::Text);
        assert_eq!(read("</function>\n</tool_call>", &["path", "content"]), closes("</function>\n</tool_call>"));
        let value = "\nsee </function>\n</function>\nstill text\n</parameter>\n</function>\n</tool_call>\nafter";
        let at = value.rfind("</parameter>").unwrap();
        assert_eq!(end(value, &["path", "content"]), Some((at, Step::Close { len: value.rfind('>').unwrap() + 1 - at })));
    }

    #[test]
    fn a_forgotten_parameter_closer_before_a_step_is_tolerated() {
        // No closer at all: the next opener, or the close, ends the value.
        assert_eq!(read("<parameter=content>", &["path"]), opens(1, "<parameter=content>"));
        assert_eq!(read("</function>\n</tool_call>", &["path", "content"]), closes("</function>\n</tool_call>"));
        // The closer written: one step with it.
        assert_eq!(read("</parameter>\n<parameter=content>", &["path"]), opens(1, "</parameter>\n<parameter=content>"));
        let text = "</parameter>\n</function>\n</tool_call>";
        assert_eq!(read(text, &["path", "content"]), closes(text));
    }

    #[test]
    fn a_close_with_a_required_parameter_missing_is_not_a_close() {
        for text in ["</function>\n</tool_call>", "</parameter>\n</function>\n</tool_call>"] {
            assert_eq!(read(text, &["path"]), Line::Text, "{text:?}");
            assert_eq!(read(text, &["path", "mode"]), Line::Text, "{text:?}");
            assert_eq!(read(text, &["path", "content"]), closes(text), "{text:?}");
        }
        // So the value runs on, and ends at the opener of the missing one.
        let value = "\nx\n</parameter>\n</function>\n</tool_call>\n<parameter=content>\ny";
        let at = value.find("<parameter=content>").unwrap();
        assert_eq!(end(value, &["path"]), Some((at, Step::Parameter { index: 1, len: 19 })));
        // A required parameter the tool does not declare is never written.
        let parameters = Parameters::new(["a"], ["a", "b"]);
        assert_eq!(line(b"</function>\n</tool_call>", &parameters, &[true]), Line::Text);
        // With none required, a call of no parameters closes.
        assert_eq!(line(b"</function>\n</tool_call>", &Parameters::new(["a"], [""; 0]), &[]), closes("</function>\n</tool_call>"));
    }

    #[test]
    fn a_value_cannot_hold_its_own_calls_confirmed_close() {
        // The documented residue: the value ends at the first confirmed close.
        let value = "\nan example:\n</function>\n</tool_call>\nand the rest\n</parameter>\n</function>\n</tool_call>";
        let at = value.find("</function>").unwrap();
        assert_eq!(end(value, &["path", "content"]), Some((at, Step::Close { len: "</function>\n</tool_call>".len() })));
    }

    #[test]
    fn newlines_merged_with_whitespace_around_the_tags_are_read_as_bytes() {
        // Blank lines and trailing whitespace between a step's tags.
        let text = "</parameter> \t\r\n\n<parameter=content>";
        assert_eq!(read(text, &["path"]), opens(1, text));
        let text = "</parameter>\n\n</function>\r\n\n</tool_call>";
        assert_eq!(read(text, &["path", "content"]), closes(text));
        assert_eq!(read("</parameter><parameter=content>", &["path"]), opens(1, "</parameter><parameter=content>"));
        // A value's whitespace before the line, merged into one token by the
        // tokenizer (`\n\n`, ` \n`, `\r\n`), stays the value's: the line
        // starts after the last `\n`.
        for value in ["\nx \n\n</parameter>\n<parameter=content>", "\nx\r\n</parameter>\n<parameter=content>", "\n\n\n<parameter=content>"] {
            assert_eq!(end(value, &["path"]).map(|(at, _)| at), value.find('<'), "{value:?}");
        }
        // A tag counts only at a line's first byte.
        assert_eq!(read(" </parameter>\n<parameter=content>", &["path"]), Line::Text);
        assert_eq!(end("\nx </parameter>\n  <parameter=content>\ny", &["path"]), None);
        // A value's first byte starts a line.
        assert_eq!(end("</parameter>\n<parameter=content>", &["path"]).map(|(at, _)| at), Some(0));
    }

    #[test]
    fn every_proper_prefix_of_a_step_is_pending_and_a_broken_one_is_text() {
        let steps = [
            ("<parameter=content>", &["path"][..]),
            ("</parameter>\n<parameter=mode>", &["path", "content"][..]),
            ("</parameter>\n</function>\n</tool_call>", &["path", "content"][..]),
            ("</function>\n</tool_call>", &["path", "content"][..]),
        ];
        for (step, written) in steps {
            for n in 1..step.len() {
                assert_eq!(read(&step[..n], written), Line::Pending, "{:?}", &step[..n]);
            }
        }
        assert_eq!(read("", &["path"]), Line::Pending);
        // Broken as soon as no step can follow.
        assert_eq!(read("<parameter=pa", &["path"]), Line::Text, "path is written");
        assert_eq!(read("<parameter=pa", &[]), Line::Pending);
        assert_eq!(read("</f", &["path"]), Line::Text, "content is missing, so no close");
        assert_eq!(read("</parameter>\n</f", &["path"]), Line::Text);
        assert_eq!(read("</parameter>\n", &["path", "content", "mode"]), Line::Pending, "it may close");
        assert_eq!(read("<a", &[]), Line::Text);
        assert_eq!(read("x", &[]), Line::Text);
        // Every parameter written and a required one undeclared: nothing can follow.
        let parameters = Parameters::new(["a"], ["b"]);
        assert_eq!(line(b"", &parameters, &[true]), Line::Text);
    }

    #[test]
    fn a_declared_name_holding_the_tags_end_is_never_opened() {
        let parameters = Parameters::new(["a>b", "a"], ["a"]);
        assert_eq!(line(b"<parameter=a>b>", &parameters, &[false, false]), Line::Step(Step::Parameter { index: 1, len: 13 }));
        assert_eq!(line(b"<parameter=a>b>", &parameters, &[false, true]), Line::Text);
    }

    /// The rule's answer for a line is stable: once text or a step, more
    /// bytes never change it. What a grammar that reads a line byte by byte
    /// relies on.
    #[test]
    fn a_lines_answer_never_changes_once_decided() {
        const PIECES: [&str; 14] = [
            "</parameter>", "<parameter=", "</function>", "</tool_call>", "path", "content", "mode", ">", "\n", " ", "\r",
            "<", "/", "x",
        ];
        let parameters = write_file();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..20_000 {
            let written: Vec<bool> = (0..3).map(|_| next() % 2 == 0).collect();
            let text: String = (0..next() % 8).map(|_| PIECES[next() as usize % PIECES.len()]).collect();
            let bytes = text.as_bytes();
            let whole = line(bytes, &parameters, &written);
            for n in 0..=bytes.len() {
                match line(&bytes[..n], &parameters, &written) {
                    Line::Text => assert_eq!(whole, Line::Text, "{:?} then {text:?}", &text[..n]),
                    Line::Step(step) => assert_eq!(whole, Line::Step(step), "{:?} then {text:?}", &text[..n]),
                    Line::Pending => {}
                }
            }
        }
    }
}
