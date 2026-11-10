//! Cloudflare edge discovery: DNS SRV
//! (`_v2-origintunneld._tcp.argotunnel.com`) with a DNS-over-TLS
//! fallback through `1.1.1.1:853`. Mirrors the semantics of
//! `cloudflared/edgediscovery/allregions/discovery.go`.
//!
//! The result is a list of `EdgeAddr`s (resolved IPs + port 7844)
//! the caller can hand to `quic_dial::dial_any`. Order is shuffled
//! per-resolution so two adjacent processes don't pin the same
//! edge, and an in-memory cache with a 1h TTL keeps repeated
//! reconnects from hammering DNS.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hickory_resolver::config::{
    LookupIpStrategy, NameServerConfig, ResolverConfig, ResolverOpts, GOOGLE,
};
use hickory_resolver::net::runtime::TokioRuntimeProvider;
use hickory_resolver::proto::rr::RData;
use hickory_resolver::TokioResolver;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::error::TunnelError;

/// SRV record we resolve to discover the v2 origintunneld pool.
pub const SRV_NAME: &str = "_v2-origintunneld._tcp.argotunnel.com";

/// Server name for the DoT fallback resolver.
pub const DOT_SERVER_NAME: &str = "cloudflare-dns.com";

/// DoT endpoint address (Cloudflare public resolver).
pub const DOT_SERVER_ADDR: &str = "1.1.1.1:853";

/// Default in-memory cache TTL for resolved edges.
pub const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(3600);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IpVersionFilter {
    #[default]
    Auto,
    V4Only,
    V6Only,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeIpVersion {
    V4,
    V6,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeAddr {
    pub ip: IpAddr,
    pub port: u16,
    pub version: EdgeIpVersion,
}

impl EdgeAddr {
    pub fn socket(&self) -> SocketAddr {
        SocketAddr::new(self.ip, self.port)
    }

    fn from_ip(ip: IpAddr, port: u16) -> Self {
        let version = if ip.is_ipv4() {
            EdgeIpVersion::V4
        } else {
            EdgeIpVersion::V6
        };
        Self { ip, port, version }
    }

    fn matches(&self, filter: IpVersionFilter) -> bool {
        matches!(
            (filter, self.version),
            (IpVersionFilter::Auto, _)
                | (IpVersionFilter::V4Only, EdgeIpVersion::V4)
                | (IpVersionFilter::V6Only, EdgeIpVersion::V6)
        )
    }
}

/// One-shot discovery without caching. Google's public resolver
/// group first (not the host's configuration); on failure / empty
/// answer, falls back to DoT via `1.1.1.1`.
pub async fn discover(filter: IpVersionFilter) -> Result<Vec<EdgeAddr>, TunnelError> {
    let primary = match build_primary_resolver() {
        Ok(r) => resolve_srv(&r, filter).await,
        Err(e) => Err(e),
    };
    match primary {
        Ok(edges) if !edges.is_empty() => return Ok(edges),
        Ok(_) => warn!("primary resolver returned zero edges; falling back to DoT"),
        Err(e) => warn!(error = %e, "primary resolver SRV failed; falling back to DoT"),
    }

    let dot = build_dot_resolver()?;
    let edges = resolve_srv(&dot, filter).await?;
    if edges.is_empty() {
        return Err(TunnelError::Discovery(format!(
            "DoT fallback also returned no edges for {SRV_NAME}"
        )));
    }
    Ok(edges)
}

/// In-memory cache around `discover`. Re-resolves once the TTL
/// expires; otherwise hands out a fresh shuffle of the previous
/// result so callers see a different head edge across calls.
#[derive(Clone)]
pub struct EdgeRegistry {
    inner: Arc<RwLock<Option<Cached>>>,
    ttl: Duration,
}

struct Cached {
    edges: Vec<EdgeAddr>,
    expires_at: Instant,
    filter: IpVersionFilter,
}

impl EdgeRegistry {
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_CACHE_TTL)
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(None)),
            ttl,
        }
    }

    pub async fn get_or_refresh(
        &self,
        filter: IpVersionFilter,
    ) -> Result<Vec<EdgeAddr>, TunnelError> {
        {
            let guard = self.inner.read().await;
            if let Some(c) = guard.as_ref() {
                if c.filter == filter && c.expires_at > Instant::now() {
                    debug!(count = c.edges.len(), "edge cache hit");
                    return Ok(shuffled(&c.edges));
                }
            }
        }
        let edges = discover(filter).await?;
        let mut guard = self.inner.write().await;
        *guard = Some(Cached {
            edges: edges.clone(),
            expires_at: Instant::now() + self.ttl,
            filter,
        });
        Ok(shuffled(&edges))
    }
}

impl Default for EdgeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Internals ─────────────────────────────────────────────────────────────────

// The primary is hickory's public Google group over UDP and TCP:
// what hickory 0.24's `ResolverConfig::default()` was.
fn primary_config() -> ResolverConfig {
    ResolverConfig::udp_and_tcp(&GOOGLE)
}

// hickory 0.26 asks AAAA and A together by default; 0.24 asked A
// first and AAAA only when A failed. Keep 0.24's order.
fn primary_opts() -> ResolverOpts {
    let mut opts = ResolverOpts::default();
    opts.ip_strategy = LookupIpStrategy::Ipv4thenIpv6;
    opts
}

fn build_primary_resolver() -> Result<TokioResolver, TunnelError> {
    TokioResolver::builder_with_config(primary_config(), TokioRuntimeProvider::default())
        .with_options(primary_opts())
        .build()
        .map_err(|e| TunnelError::Discovery(format!("primary resolver: {e}")))
}

fn dot_config() -> Result<ResolverConfig, TunnelError> {
    let addr: SocketAddr = DOT_SERVER_ADDR
        .parse()
        .map_err(|e| TunnelError::Discovery(format!("DoT addr parse: {e}")))?;
    let mut ns = NameServerConfig::tls(addr.ip(), Arc::from(DOT_SERVER_NAME));
    for conn in &mut ns.connections {
        conn.port = addr.port();
    }
    Ok(ResolverConfig::from_name_servers(vec![ns]))
}

fn dot_opts() -> ResolverOpts {
    let mut opts = primary_opts();
    opts.timeout = Duration::from_secs(15);
    opts
}

// The DoT client verifies 1.1.1.1's certificate against the webpki
// (Mozilla) roots. hickory's own TLS config fills its root store only
// under features this crate never enabled, so it is given one here.
fn dot_root_store() -> rustls::RootCertStore {
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    roots
}

fn dot_tls_config() -> Result<rustls::ClientConfig, TunnelError> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    Ok(rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| TunnelError::Discovery(format!("DoT TLS config: {e}")))?
        .with_root_certificates(dot_root_store())
        .with_no_client_auth())
}

fn build_dot_resolver() -> Result<TokioResolver, TunnelError> {
    TokioResolver::builder_with_config(dot_config()?, TokioRuntimeProvider::default())
        .with_options(dot_opts())
        .with_tls_config(dot_tls_config()?)
        .build()
        .map_err(|e| TunnelError::Discovery(format!("DoT resolver: {e}")))
}

async fn resolve_srv(
    resolver: &TokioResolver,
    filter: IpVersionFilter,
) -> Result<Vec<EdgeAddr>, TunnelError> {
    let srv = resolver
        .srv_lookup(SRV_NAME)
        .await
        .map_err(|e| TunnelError::Discovery(format!("SRV {SRV_NAME}: {e}")))?;

    let mut edges = Vec::new();
    for rec in srv.answers().iter().filter_map(|r| match &r.data {
        RData::SRV(srv) => Some(srv),
        _ => None,
    }) {
        let target = rec.target.to_utf8();
        let target = target.trim_end_matches('.');
        let port = rec.port;
        match resolver.lookup_ip(target).await {
            Ok(ips) => {
                for ip in ips.iter() {
                    let edge = EdgeAddr::from_ip(ip, port);
                    if edge.matches(filter) {
                        edges.push(edge);
                    }
                }
            }
            Err(e) => warn!(target, error = %e, "IP resolution failed for SRV target"),
        }
    }
    Ok(edges)
}

fn shuffled(input: &[EdgeAddr]) -> Vec<EdgeAddr> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    Instant::now().elapsed().as_nanos().hash(&mut h);
    let n = input.len().max(1);
    let offset = (h.finish() as usize) % n;
    let mut out = Vec::with_capacity(input.len());
    out.extend_from_slice(&input[offset..]);
    out.extend_from_slice(&input[..offset]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hickory_resolver::config::ProtocolConfig;
    use std::net::Ipv4Addr;

    fn fake(ip: u8) -> EdgeAddr {
        EdgeAddr {
            ip: IpAddr::V4(Ipv4Addr::new(198, 41, 192, ip)),
            port: 7844,
            version: EdgeIpVersion::V4,
        }
    }

    #[test]
    fn filter_matches_auto() {
        let e = fake(1);
        assert!(e.matches(IpVersionFilter::Auto));
        assert!(e.matches(IpVersionFilter::V4Only));
        assert!(!e.matches(IpVersionFilter::V6Only));
    }

    #[test]
    fn shuffle_preserves_set() {
        let input: Vec<_> = (0..8).map(fake).collect();
        let out = shuffled(&input);
        assert_eq!(out.len(), input.len());
        let mut in_ips: Vec<_> = input.iter().map(|e| e.ip).collect();
        let mut out_ips: Vec<_> = out.iter().map(|e| e.ip).collect();
        in_ips.sort();
        out_ips.sort();
        assert_eq!(in_ips, out_ips);
    }

    /// The primary resolver asks what hickory 0.24's
    /// `ResolverConfig::default()` asked: Google's public group, plain
    /// UDP then TCP on port 53, an empty NXDOMAIN trusted.
    #[test]
    fn primary_is_googles_public_group_over_udp_and_tcp() {
        let cfg = primary_config();
        let ips: Vec<IpAddr> = cfg.name_servers().iter().map(|ns| ns.ip).collect();
        let want: Vec<IpAddr> = [
            "8.8.8.8",
            "8.8.4.4",
            "2001:4860:4860::8888",
            "2001:4860:4860::8844",
        ]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
        assert_eq!(ips, want);
        for ns in cfg.name_servers() {
            assert!(ns.trust_negative_responses);
            let conns: Vec<(ProtocolConfig, u16)> = ns
                .connections
                .iter()
                .map(|c| (c.protocol.clone(), c.port))
                .collect();
            assert_eq!(conns, vec![(ProtocolConfig::Udp, 53), (ProtocolConfig::Tcp, 53)]);
        }
    }

    /// The DoT fallback is one server, 1.1.1.1 on 853 over TLS, named
    /// `cloudflare-dns.com` for the certificate check, with a 15 s timeout.
    #[test]
    fn dot_fallback_is_cloudflare_over_tls_on_853() {
        let cfg = dot_config().unwrap();
        assert_eq!(cfg.name_servers().len(), 1);
        let ns = &cfg.name_servers()[0];
        assert_eq!(ns.ip, IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1)));
        assert_eq!(ns.connections.len(), 1);
        assert_eq!(ns.connections[0].port, 853);
        assert_eq!(
            ns.connections[0].protocol,
            ProtocolConfig::Tls {
                server_name: Arc::from("cloudflare-dns.com")
            }
        );
        assert_eq!(dot_opts().timeout, Duration::from_secs(15));
    }

    /// hickory 0.26 defaults to asking AAAA and A together; 0.24 asked A
    /// first and AAAA only when that failed. Both resolvers keep 0.24's
    /// order, so the edge list holds the addresses it held before.
    #[test]
    fn both_resolvers_ask_ipv4_first() {
        assert_eq!(primary_opts().ip_strategy, LookupIpStrategy::Ipv4thenIpv6);
        assert_eq!(dot_opts().ip_strategy, LookupIpStrategy::Ipv4thenIpv6);
    }

    /// Under 0.24 the DoT client's root store was filled only by
    /// hickory features the crate did not enable, so every handshake
    /// had nothing to verify against. The fallback now carries the
    /// webpki roots, and its resolver builds with them.
    #[test]
    fn dot_fallback_trusts_the_public_roots() {
        assert!(!dot_root_store().is_empty());
        build_dot_resolver().unwrap();
    }

    #[tokio::test]
    async fn registry_serves_cached_within_ttl() {
        let reg = EdgeRegistry::with_ttl(Duration::from_secs(60));
        {
            let mut g = reg.inner.write().await;
            *g = Some(Cached {
                edges: vec![fake(7), fake(8)],
                expires_at: Instant::now() + Duration::from_secs(60),
                filter: IpVersionFilter::Auto,
            });
        }
        let got = reg.get_or_refresh(IpVersionFilter::Auto).await.unwrap();
        assert_eq!(got.len(), 2);
    }

    /// Real edge discovery. Gated so CI doesn't pound DNS on every
    /// PR; opt-in with `CFQT_LIVE_TESTS=1`.
    #[tokio::test]
    #[ignore]
    async fn live_discover_returns_edges() {
        if std::env::var_os("CFQT_LIVE_TESTS").is_none() {
            eprintln!("skip: set CFQT_LIVE_TESTS=1 to run");
            return;
        }
        let edges = discover(IpVersionFilter::Auto).await.unwrap();
        assert!(!edges.is_empty(), "should resolve at least one edge");
        for e in &edges {
            assert_eq!(e.port, 7844);
        }
    }

    /// The DoT fallback on its own, against the real 1.1.1.1:853: the
    /// TLS handshake verifies and the SRV answer resolves to edges.
    /// Same opt-in as above.
    #[tokio::test]
    #[ignore]
    async fn live_dot_fallback_returns_edges() {
        if std::env::var_os("CFQT_LIVE_TESTS").is_none() {
            eprintln!("skip: set CFQT_LIVE_TESTS=1 to run");
            return;
        }
        let dot = build_dot_resolver().unwrap();
        let edges = resolve_srv(&dot, IpVersionFilter::Auto).await.unwrap();
        assert!(!edges.is_empty(), "DoT should resolve at least one edge");
        for e in &edges {
            assert_eq!(e.port, 7844);
        }
    }
}
