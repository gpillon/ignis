//! Reasoning text for the loop guard's tests (ADR 0052): prose whose windows
//! score clean and a verbatim loop the guard rewinds.
//!
//! Included by `#[path]` from `src/loop_guard.rs`'s unit tests and from
//! `responses.rs`, so the guard and its surfaces are tested on one text.

#![allow(dead_code)]

/// Plausible reasoning that does not repeat: a running account of numbered,
/// varied sentences, a line break after every fourth. Its windows score
/// 0.45-0.49 normalized, clean with room to spare, as real reasoning's
/// mostly do.
pub fn prose(chars: usize) -> String {
    const WORDS: &[&str] = &[
        "Rome", "founded", "kings", "republic", "senate", "consuls", "tribunes", "plebeians", "wars", "Carthage",
        "Hannibal", "Scipio", "Gracchi", "Marius", "Sulla", "Pompey", "Caesar", "Augustus", "empire", "Nero",
        "Flavians", "Trajan", "Hadrian", "crisis", "Diocletian", "Constantine", "Christianity", "Goths", "Vandals",
        "Odoacer", "popes", "Charlemagne", "communes", "Avignon", "Renaissance", "Borgia", "sack", "Baroque",
        "Bernini", "Napoleon", "unification", "capital", "Fascism", "war", "liberation", "republic", "Jubilee",
        "aqueducts", "Forum", "Colosseum", "Pantheon", "Tiber", "Palatine", "legions", "Latin", "law", "Cicero",
        "Vergil", "Ostia", "plague", "Gregory", "Leo", "Vatican", "Sistine", "Michelangelo", "Raphael", "Trastevere",
        "Garibaldi", "Mazzini", "Cavour", "Mussolini", "Lateran", "Olympics", "Fellini", "Tridentine", "Jesuits",
        "Inquisition", "Galileo", "Campidoglio", "Aventine", "Esquiline", "Quirinal", "Viminal", "Caelian", "Janiculum",
    ];
    let mut out = String::new();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut n = 0;
    while out.chars().count() < chars {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let len = 6 + (state % 9) as usize;
        out.push_str(&format!("{n}. "));
        for k in 0..len {
            let word = WORDS[((state >> ((k * 5) % 60)) as usize + k * 7 + n) % WORDS.len()];
            out.push_str(word);
            out.push(if k + 1 == len { '.' } else { ' ' });
        }
        out.push(if n % 4 == 3 { '\n' } else { ' ' });
        n += 1;
    }
    out.chars().take(chars).collect()
}

/// A verbatim loop, `chars` long.
pub fn looping(chars: usize) -> String {
    "the \"Dark Ages\" (the \"Dark Ages\", the \"Dark Ages\"), ".chars().cycle().take(chars).collect()
}

/// A source of reasoning sentences for the repeat signal's tests, in its
/// own words and language.
#[derive(Debug, Clone, Copy)]
pub enum Source {
    /// A coding agent thinking aloud, in English.
    Agent,
    /// A history told in English.
    History,
    /// A coding agent thinking aloud, in Italian.
    Italian,
}

impl Source {
    pub const ALL: [Source; 3] = [Source::Agent, Source::History, Source::Italian];

    /// The parts each sentence is made of, in order.
    fn parts(self) -> [&'static [&'static str]; 5] {
        match self {
            Source::Agent => [
                &[
                    "Let me check whether", "I should confirm that", "It looks like", "So far it seems that",
                    "I need to verify that", "The log suggests that", "My guess is that", "It is possible that",
                    "I am not sure that", "The reviewer thinks that",
                ],
                &[
                    "the parser", "the cache", "the request handler", "the retry loop", "the config loader",
                    "the session store", "the token counter", "the build script", "the migration", "the scheduler",
                    "the test fixture", "the error path",
                ],
                &[
                    "still reads", "never resets", "drops", "rewrites", "keeps", "skips", "validates", "copies",
                    "locks", "flushes",
                ],
                &[
                    "the old field", "the stale entry", "the last batch", "the whole buffer", "the default value",
                    "the open handle", "the second header", "the pending write", "the cached plan", "the error code",
                ],
                &[
                    "after a restart", "when the queue is full", "on the first call", "before the lock is taken",
                    "under load", "once the timeout fires", "for an empty input", "in the slow path",
                ],
            ],
            Source::History => [
                &[
                    "In the end", "That year", "Soon after", "Against all odds", "For a while", "Once again",
                    "By then", "Without warning", "At last", "Even so",
                ],
                &[
                    "the senate", "the consuls", "the legions", "the tribunes", "the emperor", "the plebeians",
                    "the pope", "the city", "the Goths", "the merchants", "the Gracchi", "the Flavians",
                ],
                &[
                    "rebuilt", "abandoned", "defended", "taxed", "flooded", "fortified", "sold", "crowned", "besieged",
                    "restored",
                ],
                &[
                    "the Forum", "the aqueducts", "the Palatine", "the harbour at Ostia", "the Colosseum", "the walls",
                    "the Pantheon", "the bridges on the Tiber", "the old temples", "the granaries",
                ],
                &[
                    "after the great fire", "during the long siege", "in a single summer",
                    "against the advice of the priests", "with money from the provinces", "while the plague spread",
                    "before the winter came", "to win back the crowd",
                ],
            ],
            Source::Italian => [
                &[
                    "Bisogna capire se", "Mi sembra che", "Devo controllare che", "Il registro dice che", "Forse",
                    "Non sono sicuro che", "Il revisore pensa che", "Prima di tutto", "A quanto pare",
                    "Resta da vedere se",
                ],
                &[
                    "il parser", "la cache", "il gestore delle richieste", "il ciclo dei tentativi",
                    "il caricatore della configurazione", "l'archivio delle sessioni", "il contatore dei token",
                    "lo script di build", "la migrazione", "lo scheduler", "il caso di prova", "il percorso d'errore",
                ],
                &[
                    "legge ancora", "non azzera mai", "scarta", "riscrive", "conserva", "salta", "convalida", "copia",
                    "blocca", "svuota",
                ],
                &[
                    "il vecchio campo", "la voce scaduta", "l'ultimo lotto", "l'intero buffer",
                    "il valore predefinito", "il file aperto", "la seconda intestazione", "la scrittura in sospeso",
                    "il piano in cache", "il codice d'errore",
                ],
                &[
                    "dopo un riavvio", "quando la coda è piena", "alla prima chiamata", "prima di prendere il lock",
                    "sotto carico", "quando scade il timeout", "con un input vuoto", "nel percorso lento",
                ],
            ],
        }
    }

    /// `count` sentences of this source, no two the same, each ending in a
    /// full stop: 40 to 130 characters, no digit and no backtick, so the
    /// detector scores them as written.
    pub fn sentences(self, count: usize) -> Vec<String> {
        let parts = self.parts();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15 ^ self as u64;
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        while out.len() < count {
            let words: Vec<&str> = parts
                .iter()
                .map(|choices| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    choices[(state >> 11) as usize % choices.len()]
                })
                .collect();
            let sentence = format!("{}.", words.join(" "));
            if seen.insert(sentence.clone()) {
                out.push(sentence);
            }
        }
        out
    }
}
