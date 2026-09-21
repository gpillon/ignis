"""Why does `number` answer 300 for three days?

The doc in `numbers.rs::DIGITS` says a width two or more past the value's own
makes the model pad on the left, and that the tell for the broken case is the
*first* digit's probability falling to 0.65-0.71. The user's trace shows
neither: `digits: 3` for a one-digit truth came back 300 with a first digit at
0.998.

So: the same state, the same two questions, every width 1..6, with the whole
per-digit trace kept. Three truths of different magnitudes, so "the value's
own width" is a variable and not a constant.
"""

import json
import urllib.request

URL = "http://127.0.0.1:8000/v1/decide"

STATE = (
    "Help! My payouts have been failing for 3 days. I've emailed twice and "
    "nobody has replied. If this isn't fixed by Friday I'm moving to another "
    "provider."
)

# (name, criterion, truth) — every truth is stated in the text above.
ASKS = [
    ("days", "For how many days have the payouts been failing?", 3),
    ("emails", "How many emails has the customer sent?", 2),
]

# A second state whose truths are two and three digits wide, so the width
# that "fits" is a different number.
STATE_BIG = (
    "Our batch job processed 47 invoices today and rejected 128 of them. "
    "It has been running for 6 days."
)
ASKS_BIG = [
    ("invoices", "How many invoices were processed?", 47),
    ("rejected", "How many invoices were rejected?", 128),
]


def ask(state, criterion, digits):
    body = {
        "state": state,
        "questions": {"q": {"type": "number", "instructions": criterion, "digits": digits}},
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))["answers"]["q"]


def main():
    out = open("width_probe.jsonl", "w", encoding="utf-8")
    for state, asks in ((STATE, ASKS), (STATE_BIG, ASKS_BIG)):
        for name, criterion, truth in asks:
            print(f"\n{name}  (truth {truth}, {len(str(truth))} digits)")
            print(f"  {'width':>5} {'answer':>8}  digits (p)")
            for digits in range(1, 7):
                answer = ask(state, criterion, digits)
                trace = " ".join(
                    f"{d['digit']}({d['probability']:.3f})" for d in answer["digits"]
                )
                mark = "*" if answer["number"] == truth else " "
                print(f"  {digits:5} {answer['number']:8}{mark} {trace}")
                out.write(
                    json.dumps(
                        {"item": name, "truth": truth, "digits": digits, "answer": answer}
                    )
                    + "\n"
                )
                out.flush()
    out.close()


if __name__ == "__main__":
    main()
