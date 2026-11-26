//! PNG compression assertions shared by unit and HTTP tests.

/// The first IDAT stream's zlib FLEVEL: 0 for png 0.18.1's Fast encoder,
/// 2 for Balanced. This identifies the route without a wall-clock threshold.
pub fn compression_level(bytes: &[u8]) -> u8 {
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "PNG signature");
    let mut offset = 8;
    while offset + 12 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        let payload = bytes.get(offset + 8..offset + 8 + length).expect("whole PNG chunk");
        if &bytes[offset + 4..offset + 8] == b"IDAT" {
            assert!(length >= 2, "IDAT contains its zlib header");
            return payload[1] >> 6;
        }
        offset += length + 12;
    }
    panic!("PNG has an IDAT stream");
}
