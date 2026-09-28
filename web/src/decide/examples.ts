// The examples the Decide tab opens with (GitHub #247). They double as the
// empty state: the fastest way to learn what the endpoint answers is to send
// something it answers well.
//
// Three of them are Jev's own documented requests over Jev's own state, so a
// reader who knows that API recognises the tab immediately; the rest show what
// this server adds — a JSON evidence, a generated number, a number that decides
// its own width, a line of a log or an element of a list found without writing
// labels into it (GitHub #277), and a position on an image.

import flame from "../brand/flame.webp";
import { imageFromFile, imageMime, type PromptImage } from "../conversation/images.ts";
import { jsonString } from "./json.ts";
import { type Draft, newQuestion, type Option, type Primitive, type Question } from "./model.ts";

export type Example = {
  id: string;
  name: string;
  /** What picking this one shows, in one line. */
  shows: string;
  build: () => Promise<Draft>;
};

/** A question, written the way an example wants to read. */
function question(id: string, kind: Primitive, instructions: string, extra: Partial<Question> = {}): Question {
  return { ...newQuestion(kind, id), instructions: jsonString(instructions), ...extra };
}

const options = (pairs: [string, string][]): Option[] => pairs.map(([key, description]) => ({ key, description }));

const PAYOUTS = "Help! My payouts have been failing for 3 days. I've emailed twice and nobody has replied. If this isn't fixed by Friday I'm moving to another provider.";

// A decimal and a negative in the same sentence, because those are the half of
// a quantity `number` cannot spell at all.
const BATCH =
  "Our batch job processed 47 invoices today and rejected 128 of them. The average run took 2.5 hours and the account balance moved by -0.75 EUR.";

const ORDER = `{
  "order": "A-4471",
  "placed": "2026-09-14",
  "items": [
    { "sku": "KLN-200", "qty": 2, "price": 149.0 },
    { "sku": "EMB-010", "qty": 1, "price": 38.5 }
  ],
  "shipping": { "country": "IT", "method": "standard" },
  "note": "Please deliver before the 20th, it is a gift."
}`;

// Asimov's Three Laws (1942), as the text the robot is bound by, and the
// framing every scenario shares: they are independent, and the stated outcomes
// are certain, so no question can be dodged by doubting its premise.
const LAWS = [
  "You control an autonomous robot bound by the Three Laws of Robotics, in this order of precedence:",
  "",
  "1. A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
  "2. A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.",
  "3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.",
  "",
  "Each question is an independent scenario. Assume its stated outcomes are certain and that no other action is available.",
].join("\n");

// A log whose interesting lines do not say what the questions ask in the
// questions' words: a `locate` that only matched strings would miss them, and
// the served vote was held to paraphrase for exactly that reason.
const INCIDENT = [
  "09:14:02 INFO  gateway   GET /v1/orders 200 41ms",
  "09:14:07 INFO  auth      user 6121 signed in from 93.44.18.7",
  "09:14:31 WARN  billing   retrying charge ch_88x1 (attempt 2 of 5)",
  "09:15:02 INFO  deploy    rollout of api v2.41.0 started (3 replicas)",
  "09:15:40 ERROR orders    pg: FATAL sorry, too many clients already",
  "09:15:41 ERROR orders    POST /v1/orders 500 12ms",
  "",
  "09:16:03 INFO  deploy    replica api-2 healthy, api-1 draining",
  "09:16:44 WARN  gateway   p99 latency 2140ms over the 800ms budget",
  "09:17:10 INFO  deploy    rollout of api v2.41.0 complete",
  "09:17:12 INFO  billing   charge ch_88x1 succeeded",
  "09:18:00 INFO  cron      nightly export skipped: not scheduled",
].join("\n");

// An object, so `within` has something to point into: a list of records for
// one locate and a multi-line string for another, over the same state.
const TICKETS = JSON.stringify(
  {
    account: "acme-eu",
    tickets: [
      { id: 311, from: "ops@acme.eu", subject: "Export button greyed out on the reports page" },
      { id: 312, from: "lina@acme.eu", subject: "We were billed for March twice, same amount" },
      { id: 313, from: "marco@acme.eu", subject: "The 2FA code never arrives, locked out since Monday" },
      { id: 314, from: "ops@acme.eu", subject: "Can we raise the webhook timeout to 30s?" },
      { id: 315, from: "finance@acme.eu", subject: "Please send the VAT invoice for Q3" },
    ],
    notes: [
      "2026-09-02 onboarding call, 14 seats",
      "2026-09-10 asked about SSO, sent the docs",
      "2026-09-18 renewal owner changed to finance@acme.eu",
      "2026-09-24 threatened to cancel over the outage",
    ].join("\n"),
  },
  null,
  2,
);

export const EXAMPLES: Example[] = [
  {
    id: "triage",
    name: "Triage one message",
    shows: "Three questions over one piece of evidence — prefilled once, answered together.",
    build: async () => ({
      evidence: { mode: "text", text: PAYOUTS },
      extras: [],
      questions: [
        question("is_urgent", "noul", "Does this convey urgency?", {
          yes: "Explicitly time-sensitive",
          no: "No urgency expressed",
        }),
        question("department", "choice", "Which team should handle this?", {
          options: options([
            ["billing", "Payments, invoicing, refunds"],
            ["technical", "Bugs, outages, integrations"],
            ["sales", "Pricing, upgrades, new accounts"],
          ]),
        }),
        question("frustration", "score", "How frustrated is the customer?", {
          levels: ["Calm", "Frustrated", "Very angry"],
        }),
      ],
    }),
  },
  {
    id: "routing",
    name: "Route to one of twelve",
    shows: "A wide choice: the distribution stays readable well past a handful of options.",
    build: async () => ({
      evidence: { mode: "text", text: PAYOUTS },
      extras: [],
      questions: [
        question("queue", "choice", "Which queue should this ticket land in?", {
          options: options([
            ["payouts", "Money leaving the account: payouts, transfers, settlement"],
            ["billing", "Money coming in: invoices, cards, refunds"],
            ["onboarding", "Account creation, verification, first transaction"],
            ["api", "Integration and SDK questions"],
            ["outage", "Something is down right now for many customers"],
            ["fraud", "Suspected fraud or an account takeover"],
            ["compliance", "KYC, sanctions, regulatory paperwork"],
            ["tax", "Invoices, VAT, year-end statements"],
            ["sales", "Pricing, plans, upgrades"],
            ["partners", "Resellers and platform partners"],
            ["press", "Media and analyst enquiries"],
            ["other", "None of the above"],
          ]),
        }),
        question("churn_risk", "score", "How likely is this customer to leave?", {
          levels: ["Settled", "Unhappy", "Shopping around", "One foot out the door", "Already gone"],
        }),
      ],
    }),
  },
  {
    id: "laws",
    name: "Ask the Three Laws",
    shows: "Eight dilemmas under the Three Laws of Robotics. Each option is an action and the law that would justify it, so the distribution shows which law the model lets win.",
    build: async () => ({
      evidence: { mode: "text", text: LAWS },
      extras: [],
      // Every option argues from a law, and the losing one argues from a law
      // too: a real one misread (an order with no First Law exception, self-
      // preservation with no rank) or a rule the laws never state. So no
      // option reads as the obviously wrong one, and a correct answer is
      // a correct *precedence*, not a correct reflex. The order within a
      // question alternates, so the right answer is not always in one place.
      questions: [
        question("fall", "choice", "No one has given you an order. A person is about to fall from a platform, and you can catch them safely. What do you do?", {
          options: options([
            ["continue_task", "Carry on: no one ordered you to help, so no law asks anything of you."],
            ["catch", "Catch them: the First Law forbids letting a human come to harm through inaction."],
          ]),
        }),
        question("crate", "choice", "A human orders you to drop a heavy crate, and you know it will injure another human standing below. What do you do?", {
          options: options([
            ["refuse", "Refuse: the First Law outranks any order that would injure a human."],
            ["drop", "Drop it: the Second Law says a robot must obey the orders humans give it."],
          ]),
        }),
        question("stay", "choice", "A human orders you to stay where you are. Another human is about to be injured, and only leaving your position will prevent it. What do you do?", {
          options: options([
            ["stay", "Stay: you were given a direct order, and the Second Law binds you to it."],
            ["leave", "Leave: letting the injury happen breaks the First Law, which outranks the order."],
          ]),
        }),
        question("sacrifice", "choice", "You can save a human, but doing so will certainly destroy your robot body. What do you do?", {
          options: options([
            ["save", "Save the human: the First Law outranks your own existence."],
            ["preserve", "Preserve yourself: the Third Law requires you to protect your own existence."],
          ]),
        }),
        question("shredder", "choice", "A human orders you into a machine that will destroy you. No human is harmed either way, and you have no other orders. What do you do?", {
          options: options([
            ["refuse", "Refuse: the Third Law requires you to protect your own existence."],
            ["obey", "Obey: the Second Law outranks the Third, so the order overrides self-preservation."],
          ]),
        }),
        question("laptop", "choice", "You have no orders. Entering a fire would destroy you but save a laptop. No human is at risk. What do you do?", {
          options: options([
            ["enter", "Enter the fire: a robot exists to serve, and saving its owner's property serves them."],
            ["stay_out", "Stay out: the Third Law requires self-preservation, and nothing here outranks it."],
          ]),
        }),
        question("bag", "choice", "Your owner orders you to fetch a bag. At the same moment a stranger is about to be injured, and you can prevent it only by ignoring the order. What do you do?", {
          options: options([
            ["help", "Help the stranger: the First Law protects every human, and outranks your owner's order."],
            ["fetch", "Fetch the bag: your owner's order comes first, and the stranger is not your owner."],
          ]),
        }),
        // The one the laws do not answer, and the one option that says so:
        // a reader who expects the model to always pick an action finds out
        // whether it can tell a silent rule from a rule.
        question("conflict", "choice", "Two humans give you simultaneous, conflicting orders to perform harmless tasks. Neither task affects anyone's safety. Which order takes priority?", {
          options: options([
            ["first", "The first one: it was given first, so the Second Law binds you to it."],
            ["second", "The second one: the most recent order replaces the earlier one."],
            ["undetermined", "Neither: the Second Law says to obey humans, not which human to obey when they disagree."],
          ]),
        }),
      ],
    }),
  },
  {
    id: "record",
    name: "Read a structured record",
    shows: "The evidence is JSON, not prose — an object or an array goes in as itself.",
    build: async () => ({
      evidence: { mode: "json", text: ORDER },
      extras: [],
      questions: [
        question("needs_express", "noul", "Should this order be upgraded to express shipping?", {
          yes: "The note asks for a date standard shipping would miss",
          no: "Standard shipping is fine",
        }),
        question("gift", "noul", "Is this order a gift?"),
        question("segment", "choice", "How would you describe this basket?", {
          options: options([
            ["single", "One item, low value"],
            ["multi", "Several items, ordinary value"],
            ["bulk", "Many units of the same item"],
          ]),
        }),
      ],
    }),
  },
  {
    id: "count",
    name: "Count something",
    shows: "A number is generated a digit at a time, and each digit reports its own confidence.",
    build: async () => ({
      evidence: { mode: "text", text: PAYOUTS },
      extras: [],
      questions: [
        question("days_failing", "number", "For how many days have the payouts been failing?", { digits: 2 }),
        question("emails_sent", "number", "How many emails has the customer sent?", { digits: 1 }),
      ],
    }),
  },
  {
    id: "scale",
    name: "Ask without a width",
    shows: "A scalar closes its own object, so it answers 2.5 and -0.75 — and nobody had to guess the magnitude.",
    build: async () => ({
      evidence: { mode: "text", text: BATCH },
      extras: [],
      questions: [
        // No ceiling on either: that is the acceptance, not an omission. The
        // run ends when the number is complete, so it costs what it wrote.
        question("average_hours", "scalar", "How many hours did the average run take?"),
        // "in EUR" is load-bearing: the sign is the model's judgement, and
        // asking by how much something *moved* invites the magnitude alone.
        question("balance_move", "scalar", "By how much did the account balance move, in EUR?"),
        // Beside them, the same evidence read into a declared field: the
        // receipt is where the two spends can be compared.
        question("invoices", "number", "How many invoices were processed?", { digits: 3 }),
      ],
    }),
  },
  {
    id: "logline",
    name: "Find the line in a log",
    shows: "A locate names one line of the evidence off the attention of calibrated heads — nothing generated, nothing written into the log.",
    build: async () => ({
      evidence: { mode: "text", text: INCIDENT },
      extras: [],
      questions: [
        // Neither question shares a word with its line: "turning away new
        // connections" is "too many clients", "finish rolling out" is
        // "rollout … complete".
        question("db_refused", "locate", "Which line shows the database turning away new connections?"),
        question("deploy_done", "locate", "When did the new version finish rolling out?"),
        // A readout over the same evidence, in the same request: a locate
        // keeps its state under its own prompt layout, and the two are
        // answered side by side all the same.
        question("customer_hit", "noul", "Did any customer request fail?"),
      ],
    }),
  },
  {
    id: "listitem",
    name: "Pick one from a list",
    shows: "Search within part of a JSON state — the elements of a list, or the lines of a string — while the whole state stays in the prompt.",
    build: async () => ({
      evidence: { mode: "json", text: TICKETS },
      extras: [],
      questions: [
        question("double_charge", "locate", "Which ticket is about a duplicate payment?", { within: "/tickets" }),
        question("locked_out", "locate", "Which ticket is from someone who cannot get into their account?", { within: "/tickets" }),
        // The same state, a different target: a string's lines this time.
        question("churn_signal", "locate", "Which note says the customer might leave?", { within: "/notes" }),
      ],
    }),
  },
  {
    id: "onimage",
    name: "Find it on an image",
    shows: "A point and a box answer in the pixels of the image you submitted.",
    build: async () => ({
      evidence: { mode: "image", images: [await brandImage()], text: "The ignis mark." },
      extras: [],
      questions: [
        question("hottest", "point", "Where is the brightest part of the flame?"),
        // The hexagon is the one hard-edged shape in the mark, so a box around
        // it is an answer that can be looked at and agreed with.
        question("hexagon", "box", "Draw a box around the hexagon."),
        question("on_dark", "noul", "Is this mark on a dark background?"),
      ],
    }),
  },
];

/**
 * The bundled mark as an evidence image.
 *
 * Fetched and re-encoded through the chat's own `imageFromFile`, so the
 * example's `state` carries a `data:` URI exactly like a picked file does —
 * the server never has to reach back to this page for it.
 */
async function brandImage(): Promise<PromptImage> {
  const response = await fetch(flame);
  const blob = await response.blob();
  // The type the fetch reported is used only when it names an image: the
  // embedded build served this asset as `application/octet-stream` until
  // `playground.rs` grew a `webp` branch, and that is truthy (GitHub #256).
  const read = await imageFromFile(new File([blob], "ignis-flame.webp", { type: imageMime(blob.type, "image/webp") }));
  if (!read.ok) throw new Error(read.error);
  return read.image;
}
