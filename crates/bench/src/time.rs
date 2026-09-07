//! Wall-clock stamps for a TTFT record ([`crate::ttft::Record::date`]) and
//! a fresh measurement session id ([`crate::ttft::TtftConfig::session`]).
//!
//! Written out rather than pulled in: a date string in a record is the only
//! calendar work this crate does, and it is worth less than a dependency.

use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds since the Unix epoch (0 if the clock is before it).
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Format Unix `secs` as an RFC 3339 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// The civil-from-days conversion is the standard one (Howard Hinnant's),
/// exact for every date this project will ever stamp.
pub fn utc_timestamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let time = secs % 86_400;
    // Shift the epoch to 0000-03-01 so leap days land at the end of the era.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        time / 3_600,
        (time % 3_600) / 60,
        time % 60
    )
}

/// A session identifier for a fresh measurement session: the UTC
/// timestamp, compacted. The operator passes the *same* one to both
/// engines' runs — that shared value is what lets the gate check confirm
/// the two records are live/live (ADR 0015).
pub fn new_session_id() -> String {
    utc_timestamp(unix_now()).replace(['-', ':'], "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timestamp_is_rfc_3339_utc() {
        assert_eq!(utc_timestamp(0), "1970-01-01T00:00:00Z");
        // A leap-year date well past the epoch, and one on a month
        // boundary (the civil-from-days conversion's awkward case).
        assert_eq!(utc_timestamp(1788870896), "2026-09-08T12:34:56Z");
        assert_eq!(utc_timestamp(1709251200), "2024-03-01T00:00:00Z");
    }
}
