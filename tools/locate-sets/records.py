"""JSON arrays of records with one to find (spec 18, family `records`).

An array of 20, 80 or 300 records, each with 5 to 8 fields, of one of three
kinds: employees (found by the city they work from), support tickets (by
their title) and catalogue products (by their name). The target's selecting
value is unique in its array; the rest are drawn from filler pools. The
question selects the target:

- **lexical** -- by repeating a word of the value that no other record has;
- **paraphrase** -- by describing the value with none of the record's content
  words: a capital by its country, a ticket by what goes wrong, a product by
  what it is for;
- **absent** -- one question in six has the target replaced by a filler
  record of the same kind.

No record has a `type` field: an array whose every element is an object with
a string `type` is read by `/v1/decide` as content parts, not evidence
(`Evidence::read`), and these arrays have to be evidence.
"""

from common import assign_absent, paraphrase_clean, rare_shared

LENGTHS = (20, 80, 300)

FIRST = ("Ines", "Tomas", "Aiko", "Marek", "Leila", "Jonas", "Priya", "Mateo", "Sofia",
         "Kwame", "Hana", "Luca", "Noor", "Emil", "Clara", "Ravi", "Freya", "Omar",
         "Yuki", "Elena", "Bruno", "Zara", "Felix", "Maya", "Ivan", "Lena", "Diego",
         "Amara", "Nils", "Chiara")
LAST = ("Marchetti", "Novak", "Tanaka", "Kowalski", "Haddad", "Berg", "Iyer", "Rossi",
        "Mensah", "Sato", "Ferreira", "Weber", "Lindqvist", "Okafor", "Moreau", "Kim",
        "Costa", "Fischer", "Nakamura", "Petrov", "Silva", "Jensen", "Duarte", "Brandt",
        "Varga", "Oliveira", "Hughes", "Keller", "Abe", "Castro")
TEAMS = ("platform", "payments", "growth", "data", "mobile", "security", "support", "design")
ROLES = ("backend engineer", "frontend engineer", "data analyst", "product manager",
         "site reliability engineer", "designer", "support specialist", "QA engineer")
LANGUAGES = ("English", "Spanish", "German", "French", "Italian", "Japanese", "Hindi",
             "Polish", "Arabic", "Dutch")
# Capitals are only ever the target: a filler record works from a city that is
# not a national capital, so "the capital of X" names one record.
CAPITALS = {
    "Nairobi": "Kenya", "Lisbon": "Portugal", "Oslo": "Norway", "Canberra": "Australia",
    "Ottawa": "Canada", "Lima": "Peru", "Hanoi": "Vietnam", "Accra": "Ghana",
    "Warsaw": "Poland", "Dublin": "Ireland", "Santiago": "Chile", "Bogota": "Colombia",
    "Helsinki": "Finland", "Athens": "Greece", "Cairo": "Egypt", "Manila": "the Philippines",
    "Jakarta": "Indonesia", "Seoul": "South Korea", "Vienna": "Austria", "Prague": "the Czech Republic",
    "Budapest": "Hungary", "Reykjavik": "Iceland", "Wellington": "New Zealand",
    "Montevideo": "Uruguay", "Kathmandu": "Nepal",
}
CITIES = ("Lyon", "Porto", "Bergen", "Krakow", "Osaka", "Seattle", "Austin", "Munich",
          "Turin", "Valencia", "Gothenburg", "Leeds", "Denver", "Toronto", "Rotterdam",
          "Hamburg", "Milan", "Barcelona", "Chicago", "Melbourne", "Busan", "Graz",
          "Antwerp", "Marseille", "Bristol", "Aarhus", "Tampere", "Gdansk", "Brno", "Cork")

# Ticket titles to find: (title, what a paraphrase asks for).
TICKETS = {
    "Checkout button unresponsive on Safari": "Apple's web browser failing to complete purchases",
    "Password reset email never arrives": "users not receiving the message to recover their login",
    "Dark mode colors unreadable in settings": "the night theme having poor contrast on the preferences page",
    "CSV export truncates long names": "spreadsheet downloads cutting off lengthy labels",
    "Push notifications delayed by hours on Android": "mobile alerts arriving very late on Google's phone OS",
    "Invoice PDF shows wrong VAT rate": "the bill document applying an incorrect sales tax percentage",
    "Search returns nothing for accented names": "lookups failing for words with diacritics",
    "Two-factor codes rejected after daylight saving change": "one-time login numbers failing once the clocks moved",
    "Map widget crashes when zooming out": "the geographic view dying on reducing magnification",
    "Duplicate charges after double click": "customers billed twice for a single purchase when pressing quickly",
    "Calendar invites sent in wrong timezone": "meeting requests going out with the hour shifted",
    "Avatar upload rejects PNG files": "profile pictures refused in a common image format",
    "Session expires while typing long comments": "being logged out mid-way through writing a reply",
    "Bulk delete removes archived items too": "mass removal also wiping stored-away entries",
    "Tooltip overlaps dropdown on small screens": "hover hints covering menus on narrow displays",
    "Weekly report email has broken chart images": "the seven-day summary message showing missing graphs",
    "Autosave overwrites newer draft": "automatic saving clobbering a more recent version",
    "Screen reader skips table headers": "assistive software for blind users missing column titles",
    "Refund status stuck on pending": "money-back requests never leaving the waiting state",
    "Language switcher resets on reload": "the locale picker forgetting the choice after refreshing",
    "Dashboard freezes with over ten widgets": "the overview page hanging when many panels are added",
    "Coupon code accepted after expiry date": "discount vouchers working past their end",
    "Keyboard shortcuts conflict with browser defaults": "hotkeys clashing with built-in navigation",
    "Printing cuts off the last column": "paper output losing the rightmost field",
}
TICKET_AREAS = ("Login", "Sidebar", "Profile page", "Admin panel", "Onboarding", "Billing page",
                "Team settings", "Audit log", "API tokens", "File browser", "Chat widget",
                "Help center", "Status page", "Invite flow", "Tag editor")
TICKET_SYMPTOMS = ("loads slowly", "shows a blank state", "misaligned on tablet",
                   "missing translation", "typo in heading", "wrong icon", "flickers on hover",
                   "ignores sort order", "spinner never stops", "requires two taps",
                   "shows stale data", "scrolls to top unexpectedly")
STATUSES = ("open", "in progress", "blocked", "resolved")
PRIORITIES = ("low", "medium", "high", "urgent")
COMPONENTS = ("frontend", "backend", "infra", "mobile", "integrations")

# Products to find: (name, what a paraphrase asks for).
PRODUCTS = {
    "Burr coffee grinder": "turns roasted beans into powder for brewing",
    "Cast iron skillet": "is a heavy metal frying pan",
    "Noise-cancelling headphones": "blocks outside sound while you listen to music",
    "Standing desk frame": "lets you work at a raised height instead of sitting",
    "Robot vacuum": "cleans floors on its own",
    "Electric kettle": "boils water",
    "Yoga mat": "gives cushioning for floor exercise",
    "Soldering iron": "melts metal to join electronic parts",
    "Dehumidifier": "pulls moisture out of the air in a room",
    "Label printer": "makes sticky tags with text",
    "Tire pressure gauge": "measures how much air is in a car's wheels",
    "Sous vide circulator": "cooks food sealed in bags in a warm water bath",
    "Hiking backpack": "carries gear on long walks in the mountains",
    "Baby monitor": "lets parents hear an infant from another room",
    "Pasta maker": "rolls and cuts dough into noodles",
    "Bike lock": "secures a bicycle against theft",
    "Air fryer": "crisps food with hot circulating air instead of oil",
    "Laser rangefinder": "measures distance to a far object with a beam of light",
    "Sewing machine": "stitches fabric together",
    "Wireless doorbell": "chimes when a visitor presses a button at the entrance",
    "Pizza stone": "is a slab you bake flatbreads on for a crisp base",
    "Water flosser": "cleans between teeth with a jet",
    "Camping stove": "heats meals outdoors on gas",
    "Smoke detector": "sounds an alarm when there is a fire",
}
PRODUCT_ADJ = ("Compact", "Deluxe", "Classic", "Portable", "Heavy-duty", "Slim", "Pro", "Mini")
PRODUCT_NOUNS = ("desk lamp", "storage box", "extension cord", "wall clock", "picture frame",
                 "shower caddy", "cable organizer", "door mat", "coat hanger set", "notebook",
                 "phone stand", "mouse pad", "tea towel", "laundry basket", "spice rack")
CATEGORIES = ("home", "kitchen", "outdoor", "office", "electronics", "sports")
SUPPLIERS = ("Northwind", "Contoso", "Fabrikam", "Tailspin", "Wingtip", "Litware")


def _date(r):
    return f"20{r.randint(18, 26)}-{r.randint(1, 12):02}-{r.randint(1, 28):02}"


def _employee(r, index, city):
    record = {"id": 1000 + index, "name": f"{r.choice(FIRST)} {r.choice(LAST)}",
              "team": r.choice(TEAMS), "role": r.choice(ROLES), "city": city}
    extras = [("started", _date(r)), ("languages", ", ".join(r.sample(LANGUAGES, 2))),
              ("remote", r.random() < 0.4)]
    for key, value in extras[:r.randint(0, 3)]:
        record[key] = value
    return record


def _ticket(r, index, title):
    record = {"id": f"TCK-{r.randint(1000, 9999)}", "title": title,
              "status": r.choice(STATUSES), "priority": r.choice(PRIORITIES),
              "component": r.choice(COMPONENTS)}
    extras = [("reporter", f"{r.choice(FIRST)} {r.choice(LAST)}"), ("opened", _date(r)),
              ("votes", r.randint(0, 40))]
    for key, value in extras[:r.randint(0, 3)]:
        record[key] = value
    return record


def _product(r, index, name):
    record = {"sku": f"SKU-{r.randint(10000, 99999)}", "name": name,
              "category": r.choice(CATEGORIES), "price": round(r.uniform(4, 400), 2),
              "stock": r.randint(0, 300)}
    extras = [("supplier", r.choice(SUPPLIERS)), ("rating", round(r.uniform(2.5, 5), 1)),
              ("discontinued", r.random() < 0.1)]
    for key, value in extras[:r.randint(0, 3)]:
        record[key] = value
    return record


def _kind(r, kind):
    """The kind's record builder, target value, filler values, and question."""
    if kind == "employees":
        city = r.choice(sorted(CAPITALS))
        return (_employee, city, lambda: r.choice(CITIES),
                f"Which employee works from {city}?",
                f"Which employee works from the capital of {CAPITALS[city]}?")
    if kind == "tickets":
        title = r.choice(sorted(TICKETS))
        filler = lambda: f"{r.choice(TICKET_AREAS)} {r.choice(TICKET_SYMPTOMS)}"
        return (_ticket, title, filler, f"Which ticket is titled \"{title}\"?",
                f"Which ticket is about {TICKETS[title]}?")
    name = r.choice(sorted(PRODUCTS))
    filler = lambda: f"{r.choice(PRODUCT_ADJ)} {r.choice(PRODUCT_NOUNS)}"
    return (_product, name, filler, f"Which product is the {name.lower()}?",
            f"Which product {PRODUCTS[name]}?")


def _text(record):
    return " ".join(f"{k} {v}" for k, v in record.items())


def one_array(r, length, kind, split, absent):
    build, value, filler, lexical, paraphrase = _kind(r, kind)
    target_at = r.randrange(length)
    records = [build(r, i, value if i == target_at else filler()) for i in range(length)]
    question = lexical if split == "lexical" else paraphrase
    target = _text(records[target_at])
    rest = [_text(x) for i, x in enumerate(records) if i != target_at]
    if split == "lexical" and not rare_shared(question, target, rest):
        return None
    if split == "paraphrase" and not paraphrase_clean(question, target):
        return None
    if absent:
        records[target_at] = build(r, target_at, filler())
    return {"state": records, "instruction": question,
            "targets": [] if absent else [target_at], "distractors": [], "event": value}


def generate(r, count, prefix="records"):
    absent = assign_absent(r, count)
    kinds = ("employees", "tickets", "products")
    rows = []
    for i in range(count):
        length = LENGTHS[i % len(LENGTHS)]
        split = ("lexical", "paraphrase")[i % 2]
        kind = kinds[(i // 6) % len(kinds)]
        for _ in range(50):
            row = one_array(r, length, kind, split, i in absent)
            if row is not None:
                break
        else:
            raise SystemExit(f"records: {kind} ({split}) failed its word checks 50 times")
        rows.append({"id": f"{prefix}-{i:03}", "family": "records", "kind": kind, "split": split,
                     "absent": i in absent, "segments": length, **row})
    return rows
