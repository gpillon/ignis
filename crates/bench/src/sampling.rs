//! The opt-in sampling a bench run can send (GitHub #578, spec #574).
//!
//! Every request the harness sends is greedy by default (`temperature: 0`,
//! `seed: 0`), which is what every recorded baseline was measured with and
//! what keeps two engines given the same work. A decode cell that asks how
//! the engine runs as an agent samples it (temperature 1, top-p 0.95,
//! top-k 20) sets a [`Sampling`] on the endpoint (`--temperature`,
//! `--top-p`, `--top-k`, `--sample-seed` on `replay` and `g5`): the body
//! then carries those three settings and a seed of its own per request.
//!
//! The seed is derived from the run's base seed and the request's id, so two
//! launches of one command send the same bodies and a replay's lines (and a
//! cell's lanes) each draw their own stream. Without the option nothing in a
//! body changes.

use serde::{Deserialize, Serialize};

/// The sampling a run sends on every request, as the run's record states it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    /// The base the per-request seeds derive from.
    pub base_seed: u64,
}

impl Sampling {
    /// The sampling the command line asked for: `None` when none of the four
    /// flags is given (the greedy contract), `Some` when `--temperature`,
    /// `--top-p` and `--top-k` all are (`--sample-seed` defaults to 0), and
    /// an error naming the flag otherwise. Values are the flags' text.
    pub fn from_flags(
        temperature: Option<&str>,
        top_p: Option<&str>,
        top_k: Option<&str>,
        seed: Option<&str>,
    ) -> Result<Option<Self>, String> {
        if temperature.is_none() && top_p.is_none() && top_k.is_none() && seed.is_none() {
            return Ok(None);
        }
        fn need<'a>(flag: &str, v: Option<&'a str>) -> Result<&'a str, String> {
            v.ok_or_else(|| format!("--temperature, --top-p and --top-k go together: --{flag} is missing"))
        }
        let temperature = need("temperature", temperature)?;
        let top_p = need("top-p", top_p)?;
        let top_k = need("top-k", top_k)?;
        let temperature = temperature
            .parse::<f64>()
            .ok()
            .filter(|t| t.is_finite() && *t > 0.0)
            .ok_or_else(|| format!("--temperature {temperature}: not a number above 0"))?;
        let top_p = top_p
            .parse::<f64>()
            .ok()
            .filter(|p| *p > 0.0 && *p <= 1.0)
            .ok_or_else(|| format!("--top-p {top_p}: not a number in (0, 1]"))?;
        let top_k = top_k
            .parse::<u32>()
            .ok()
            .filter(|k| *k >= 1)
            .ok_or_else(|| format!("--top-k {top_k}: not a whole number from 1"))?;
        let base_seed = match seed {
            Some(v) => v.parse::<u64>().map_err(|_| format!("--sample-seed {v}: not a whole number"))?,
            None => 0,
        };
        Ok(Some(Self { temperature, top_p, top_k, base_seed }))
    }

    /// The seed request `id` is sent: a pure function of the base seed and
    /// the id (FNV-1a over the id, mixed with the base by splitmix64), kept
    /// below 2^62 so it is a positive JSON integer on any reader.
    pub fn seed_for(&self, id: &str) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in id.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let mut z = hash ^ self.base_seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) & ((1 << 62) - 1)
    }

    /// `body` with this sampling in place of the greedy contract, for the
    /// request `id`.
    pub fn apply(&self, body: &mut serde_json::Value, id: &str) {
        body["temperature"] = serde_json::json!(self.temperature);
        body["top_p"] = serde_json::json!(self.top_p);
        body["top_k"] = serde_json::json!(self.top_k);
        body["seed"] = serde_json::json!(self.seed_for(id));
    }

    /// One line for the terminal.
    pub fn describe(&self) -> String {
        format!(
            "sampled: temperature {}, top-p {}, top-k {}, base seed {} (a seed per request from the base and the request id)",
            self.temperature, self.top_p, self.top_k, self.base_seed
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(t: &str, p: &str, k: &str, s: Option<&str>) -> Result<Option<Sampling>, String> {
        Sampling::from_flags(Some(t), Some(p), Some(k), s)
    }

    #[test]
    fn no_flag_is_the_greedy_contract() {
        assert_eq!(Sampling::from_flags(None, None, None, None), Ok(None));
    }

    #[test]
    fn the_three_settings_go_together_and_are_checked() {
        let s = flags("1", "0.95", "20", None).unwrap().unwrap();
        assert_eq!((s.temperature, s.top_p, s.top_k, s.base_seed), (1.0, 0.95, 20, 0));
        assert!(Sampling::from_flags(Some("1"), None, Some("20"), None).unwrap_err().contains("--top-p"));
        assert!(Sampling::from_flags(None, None, None, Some("3")).is_err());
        for (t, p, k) in [("0", "0.95", "20"), ("x", "0.95", "20"), ("1", "0", "20"), ("1", "1.5", "20"), ("1", "0.9", "0")] {
            assert!(flags(t, p, k, None).is_err(), "{t} {p} {k}");
        }
        assert!(flags("1", "0.95", "20", Some("-1")).is_err());
    }

    #[test]
    fn the_body_carries_the_sampling_and_a_seed_per_request() {
        let s = flags("0.6", "0.95", "20", Some("7")).unwrap().unwrap();
        let mut a = serde_json::json!({ "temperature": 0, "seed": 0, "max_tokens": 5 });
        s.apply(&mut a, "thr-1-1-0");
        assert_eq!(a["temperature"], 0.6);
        assert_eq!(a["top_p"], 0.95);
        assert_eq!(a["top_k"], 20);
        assert_eq!(a["max_tokens"], 5);
        let mut again = serde_json::json!({ "temperature": 0, "seed": 0 });
        s.apply(&mut again, "thr-1-1-0");
        assert_eq!(a["seed"], again["seed"], "two launches send the same body");
        let mut other = serde_json::json!({});
        s.apply(&mut other, "thr-1-1-1");
        assert_ne!(a["seed"], other["seed"], "each request draws its own stream");
        let other_base = Sampling { base_seed: 8, ..s };
        assert_ne!(s.seed_for("x"), other_base.seed_for("x"));
        assert!(a["seed"].as_u64().unwrap() < 1 << 62);
    }
}
