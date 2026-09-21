"""Can the model be *told* to pad on the left?

The system text a `number` is put under declares the width and nothing about
alignment. This asks whether the ambiguity is prompt-level at all: the same
questions, the same widths, with the padding rule appended to the caller's own
`instructions` — no code change, and `number_system` untouched.

If the model obeys, the fix is one sentence in `number_system` (which only
`number` uses, so the measured pointing prompt is not disturbed). If it does
not, the schedule has to carry a terminator and `digits` has to mean "at most".
"""
import json, urllib.request

URL = "http://127.0.0.1:8000/v1/decide"
STATE = ("Help! My payouts have been failing for 3 days. I've emailed twice and "
         "nobody has replied. If this isn't fixed by Friday I'm moving to another provider.")
STATE_BIG = ("Our batch job processed 47 invoices today and rejected 128 of them. "
             "It has been running for 6 days.")
SPELL = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

CASES = [
    (STATE, "days", "For how many days have the payouts been failing?", 3),
    (STATE, "emails", "How many emails has the customer sent?", 2),
    (STATE_BIG, "invoices", "How many invoices were processed?", 47),
    (STATE_BIG, "rejected", "How many invoices were rejected?", 128),
]

def ask(state, criterion, digits):
    suffix = (f" Write the value right-aligned in the {SPELL[digits]}-digit field, "
              f"padded on the left with zeros.")
    body = {"state": state,
            "questions": {"q": {"type": "number", "instructions": criterion + suffix,
                                "digits": digits}}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["answers"]["q"]

hits = total = 0
for state, name, criterion, truth in CASES:
    print(f"\n{name}  (truth {truth})")
    for digits in range(1, 7):
        a = ask(state, criterion, digits)
        trace = " ".join(f"{d['digit']}({d['probability']:.2f})" for d in a["digits"])
        ok = a["number"] == truth
        # A field narrower than the truth can only truncate; not the question.
        if digits >= len(str(truth)):
            total += 1
            hits += ok
        print(f"  {digits} {a['number']:8}{'*' if ok else ' '}  {trace}")
print(f"\nright-aligned when the field can hold the value: {hits}/{total}")
