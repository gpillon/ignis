//! `--expose <mode>` / `IGNIS_EXPOSE`: reach the server from outside this
//! machine without opening a port (ADR 0028). One mode today,
//! `cloudflare-quick` — an anonymous `https://*.trycloudflare.com` URL
//! proxied to the local listener — named as a mode so a later one (a named
//! tunnel, a reverse proxy) is a new variant, not a new flag.
//!
//! An exposed server always requires an API key: `config::resolve` turns an
//! unset key into `auto` whenever a mode is set, so this module never sees
//! an open API.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

/// How the server is exposed beyond its bind address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expose {
    /// A Cloudflare quick tunnel: no account, a random hostname per start,
    /// gone when the process exits.
    CloudflareQuick,
}

impl Expose {
    /// Every mode's name, as `--expose` spells it.
    pub const NAMES: &[&str] = &["cloudflare-quick"];

    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "cloudflare-quick" => Ok(Self::CloudflareQuick),
            other => Err(format!(
                "unknown expose mode `{other}` (expected one of: {})",
                Self::NAMES.join(", ")
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CloudflareQuick => "cloudflare-quick",
        }
    }
}

/// The local port a tunnel should forward to, for a listener bound at
/// `local`. The quick tunnel dials `127.0.0.1:<port>`, so the listener has
/// to accept IPv4 loopback: bound to it, or to every IPv4 interface. Any
/// other address would give a public URL that answers nothing but 502s, so
/// it is refused before the tunnel is requested.
pub fn origin_port(local: SocketAddr) -> Result<u16, String> {
    match local.ip() {
        IpAddr::V4(ip) if ip.is_loopback() || ip == Ipv4Addr::UNSPECIFIED => Ok(local.port()),
        other => Err(format!(
            "`--expose` forwards to 127.0.0.1, which a listener on {other} does not accept \
             (bind 127.0.0.1:<port> or 0.0.0.0:<port>)"
        )),
    }
}

/// A live exposure. Keep it for as long as the server serves: dropping it
/// tears the tunnel down.
pub struct Exposure {
    url: String,
    location: String,
    tunnel: cloudflare_quick_tunnel::QuickTunnelHandle,
}

impl Exposure {
    /// The public base URL (no trailing slash).
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Where the provider terminated the connection (a Cloudflare POP such
    /// as `mxp01`), for the startup log.
    pub fn location(&self) -> &str {
        &self.location
    }

    /// Unregister from the provider and wait for it to finish.
    pub async fn shutdown(self) -> Result<(), String> {
        self.tunnel.shutdown().await.map_err(|e| e.to_string())
    }
}

/// Open `mode` towards the listener bound at `local`. Returns once the
/// public URL routes (for a quick tunnel: requested, dialled and
/// registered — bounded by the crate's 30 s handshake timeout).
pub async fn start(mode: Expose, local: SocketAddr) -> Result<Exposure, String> {
    let port = origin_port(local)?;
    match mode {
        Expose::CloudflareQuick => {
            let tunnel = cloudflare_quick_tunnel::QuickTunnelManager::new(port)
                .start()
                .await
                .map_err(|e| format!("cloudflare quick tunnel: {e}"))?;
            Ok(Exposure {
                url: tunnel.url.trim_end_matches('/').to_owned(),
                location: tunnel.location.clone(),
                tunnel,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_mode_parses_back_to_its_name() {
        for name in Expose::NAMES {
            assert_eq!(Expose::parse(name).expect("known mode").as_str(), *name);
        }
    }

    #[test]
    fn an_unknown_mode_is_refused_naming_the_known_ones() {
        let err = Expose::parse("ngrok").expect_err("unknown");
        assert!(err.contains("ngrok") && err.contains("cloudflare-quick"), "{err}");
    }

    #[test]
    fn a_listener_on_loopback_or_every_ipv4_interface_can_be_tunnelled() {
        for addr in ["127.0.0.1:8000", "0.0.0.0:8000"] {
            assert_eq!(origin_port(addr.parse().unwrap()), Ok(8000), "{addr}");
        }
    }

    #[test]
    fn a_listener_the_tunnel_cannot_reach_on_127_0_0_1_is_refused() {
        for addr in ["192.168.1.10:8000", "[::1]:8000", "[::]:8000"] {
            let err = origin_port(addr.parse().unwrap()).expect_err(addr);
            assert!(err.contains("--expose") && err.contains("127.0.0.1"), "{err}");
        }
    }
}
