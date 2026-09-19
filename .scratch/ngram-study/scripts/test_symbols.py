r"""Parser tests.

The point of these is not brace matching, which is easy, but the lexical
pitfalls that make a naive `fn (\w+)` regex report symbols that do not exist:
`fn` inside a doc comment, a `{` inside a raw string, a Rust lifetime that
looks like an opening char literal, a TS regex literal holding a quote.

Run with the study venv:  F:/ai/ngram-venv/Scripts/python.exe -m pytest -q
"""

from lexmask import RUST, C, TS, mask_source
from symbols import parse_file


def names(src, path):
    syms, problems = parse_file(path, src)
    return [s.name for s in syms], problems


def body(src, path, name):
    syms, _ = parse_file(path, src)
    s = next(x for x in syms if x.name == name)
    return src[s.body_span[0]:s.body_span[1]]


# ---------------------------------------------------------------- masking ---

def test_mask_preserves_offsets_and_lines():
    src = 'let a = "one";\n// two\nlet b = 3;\n'
    masked, problems = mask_source(src, RUST)
    assert problems == []
    assert len(masked) == len(src)
    assert masked.count("\n") == src.count("\n")
    assert "one" not in masked and "two" not in masked
    assert "let b = 3;" in masked


def test_rust_nested_block_comment():
    src = "/* outer /* inner */ still comment */ fn real() {}\n"
    got, problems = names(src, "a.rs")
    assert got == ["real"] and problems == []


def test_c_block_comment_does_not_nest():
    src = "/* outer /* inner */ void real(void) { }\n"
    got, _ = names(src, "a.c" if False else "a.cu")
    assert got == ["real"]


def test_rust_raw_string_with_braces_and_quotes():
    src = 'fn a() { let s = r#"fn fake() { " }"#; }\nfn b() {}\n'
    got, problems = names(src, "a.rs")
    assert got == ["a", "b"] and problems == []


def test_rust_lifetime_is_not_a_char_literal():
    src = "struct Wrap<'a> { r: &'a str }\nfn after() {}\n"
    got, problems = names(src, "a.rs")
    assert got == ["Wrap", "after"] and problems == []


def test_rust_char_literal_with_escape():
    src = "fn a() { let c = '\\''; let d = '\\u{1F600}'; }\nfn b() {}\n"
    got, problems = names(src, "a.rs")
    assert got == ["a", "b"] and problems == []


def test_doc_comment_keyword_is_not_a_symbol():
    src = "/// calls `fn ghost()` and `struct Ghost`\nfn real() {}\n"
    got, _ = names(src, "a.rs")
    assert got == ["real"]


def test_ts_template_literal_with_hole():
    src = "const a = `x ${ fake() } { y`;\nfunction real() {}\n"
    got, problems = names(src, "a.ts")
    assert "real" in got and "fake" not in got and problems == []


def test_ts_nested_template_literal():
    src = "const a = `${ `inner ${1} ` } `;\nfunction real() {}\n"
    got, problems = names(src, "a.ts")
    assert "real" in got and problems == []


def test_ts_regex_literal_holding_a_quote():
    src = "const re = /it's \\/ ok/g;\nfunction real() {}\n"
    got, problems = names(src, "a.ts")
    assert "real" in got and problems == []


def test_ts_division_is_not_a_regex():
    src = "const q = a / b; const r = c / d;\nfunction real() {}\n"
    got, problems = names(src, "a.ts")
    assert "real" in got and problems == []


# ---------------------------------------------------------------- extents ---

def test_rust_fn_body_extent():
    src = "fn outer() {\n    if x { y(); }\n}\nfn after() {}\n"
    assert body(src, "a.rs", "outer") == "fn outer() {\n    if x { y(); }\n}"


def test_rust_generic_fn_with_where_clause():
    src = ("fn g<T: Into<String>>(t: T) -> Vec<T>\nwhere T: Clone,\n"
           "{\n    vec![t]\n}\n")
    ext = body(src, "a.rs", "g")
    assert ext.startswith("fn g<") and ext.endswith("}")
    assert "vec![t]" in ext


def test_rust_const_with_struct_initialiser():
    src = "const C: Cfg = Cfg { a: 1 };\nfn after() {}\n"
    assert body(src, "a.rs", "C") == "const C: Cfg = Cfg { a: 1 };"


def test_rust_trait_method_declaration_without_body():
    src = "trait T {\n    fn required(&self);\n}\n"
    got, problems = names(src, "a.rs")
    assert got == ["T", "required"] and problems == []


def test_cuda_global_kernel():
    src = ("__global__ void my_kernel(float* p, int n) {\n"
           "    if (n > 0) { p[0] = 1.f; }\n}\n")
    ext = body(src, "a.cu", "my_kernel")
    assert ext.strip().startswith("__global__") and ext.endswith("}")


def test_c_struct_and_typedef():
    src = "typedef struct Inner { int a; } Outer;\nvoid f(void) {}\n"
    got, _ = names(src, "a.h")
    assert "Inner" in got and "Outer" in got and "f" in got


def test_c_call_site_is_not_a_definition():
    src = "void real(void) {\n    other(1, 2);\n    if (x) { }\n}\n"
    got, _ = names(src, "a.cu")
    assert got == ["real"]


def test_ts_arrow_const_extent():
    src = "const handler = (e: Event) => {\n  use(e);\n};\nconst after = 1;\n"
    assert body(src, "a.ts", "handler").endswith("};")


def test_ts_interface_and_class():
    src = "interface I { a: number }\nexport class K extends B { m() {} }\n"
    got, _ = names(src, "a.tsx")
    assert "I" in got and "K" in got


# ------------------------------------------------------------- complaints ---

def test_unclosed_body_is_reported_not_dropped_silently():
    src = "fn broken() {\n    if x {\n"
    got, problems = names(src, "a.rs")
    assert got == [] and any("unclosed" in p or "no body" in p for p in problems)


def test_unterminated_block_comment_is_reported():
    src = "fn a() {}\n/* never closed\n"
    _, problems = names(src, "a.rs")
    assert any("unterminated block comment" in p for p in problems)


def test_c_digit_separator_is_not_a_char_literal():
    src = "constexpr float kTheta = 10'000.0F;\nvoid real(void) {}\n"
    got, problems = names(src, "a.cu")
    assert got == ["real"] and problems == []


def test_jsx_apostrophe_in_text_is_not_a_string():
    src = ("export function Panel() {\n  return <p>this session's log</p>;\n}\n"
           "export function After() { return null; }\n")
    got, problems = names(src, "a.tsx")
    assert got == ["Panel", "After"] and problems == []


def test_ts_string_may_not_span_a_line():
    src = 'const a = "ok";\nreturn <b>say "hi"\n  and bye</b>;\nconst after = 1;\n'
    got, problems = names(src, "a.tsx")
    assert "a" in got and "after" in got and problems == []


def test_rust_string_may_span_a_line():
    src = 'const S: &str = "line one\nline two { fake";\nfn real() {}\n'
    got, problems = names(src, "a.rs")
    assert got == ["S", "real"] and problems == []
