//! Pure rules and the wire format, kept apart from COM so they can be unit tested.

/// UTF-16 units of context read before the cursor.
pub(crate) const MAX_CONTEXT: i32 = 1000;
/// A text change counts as typing if it follows an editing key within this time.
pub(crate) const EDIT_WINDOW_MS: u64 = 500;

const VK_BACK: u32 = 0x08;
const VK_SHIFT: u32 = 0x10;
const VK_CONTROL: u32 = 0x11;
const VK_MENU: u32 = 0x12;
const VK_CAPITAL: u32 = 0x14;
const VK_SPACE: u32 = 0x20;
const VK_DELETE: u32 = 0x2E;
const VK_LWIN: u32 = 0x5B;
const VK_RWIN: u32 = 0x5C;
const VK_LSHIFT: u32 = 0xA0;
const VK_RMENU: u32 = 0xA5;

/// Keys that change no text and do not mean the user has moved on.
pub(crate) fn is_modifier(vk: u32) -> bool {
    matches!(vk, VK_SHIFT | VK_CONTROL | VK_MENU | VK_CAPITAL | VK_LWIN | VK_RWIN | VK_LSHIFT..=VK_RMENU)
}

/// Keys that type or delete text. Everything else (arrows, Enter, shortcuts) dismisses the bar.
pub(crate) fn is_edit_key(vk: u32, ctrl: bool, alt: bool) -> bool {
    if vk == VK_BACK {
        return !alt;
    }
    if ctrl || alt {
        return false;
    }
    matches!(vk, VK_SPACE | VK_DELETE
        | 0x30..=0x39       // digits
        | 0x41..=0x5A       // letters
        | 0x60..=0x6F       // numeric keypad
        | 0xBA..=0xC0       // ; = , - . / `
        | 0xDB..=0xDF)      // [ \ ] '
}

/// No bar when the current line is blank, so Tab keeps moving between empty fields and indenting code.
pub(crate) fn wants_prediction(text: &[u16]) -> bool {
    text.iter()
        .rev()
        .take_while(|&&c| !matches!(c, 0x0A | 0x0B | 0x0D | 0x2028 | 0x2029))
        .any(|&c| !matches!(c, 0x09 | 0x20 | 0xA0))
}

/// Request: a little-endian u32 length followed by the text in UTF-8.
pub(crate) fn encode_request(text: &[u16]) -> Vec<u8> {
    let body = String::from_utf16_lossy(text).into_bytes();
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

#[derive(Debug, PartialEq)]
pub(crate) struct Reply {
    /// Length of the partly typed word in UTF-16 units.
    pub partial: usize,
    pub words: Vec<String>,
}

/// Reply body: lines, the partial word's length and then the candidate words.
pub(crate) fn parse_reply(body: &[u8]) -> Option<Reply> {
    let text = std::str::from_utf8(body).ok()?;
    let mut lines = text.split('\n');
    let partial = lines.next()?.parse().ok()?;
    let words = lines.filter(|w| !w.is_empty()).map(str::to_owned).collect();
    Some(Reply { partial, words })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16s(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    #[test]
    fn typing_keys_are_edit_keys() {
        for vk in [0x41, 0x5A, 0x30, 0x20, 0x08, 0xBE, 0xDE, 0x2E] {
            assert!(is_edit_key(vk, false, false), "{vk:#x}");
        }
        assert!(is_edit_key(VK_BACK, true, false), "Ctrl+Backspace deletes a word");
    }

    #[test]
    fn navigation_and_shortcuts_are_not_edit_keys() {
        for vk in [0x25, 0x26, 0x27, 0x28, 0x0D, 0x09, 0x1B, 0x70] {
            assert!(!is_edit_key(vk, false, false), "{vk:#x}");
        }
        assert!(!is_edit_key(0x56, true, false), "Ctrl+V");
        assert!(!is_edit_key(0x31, false, true), "Alt+1");
    }

    #[test]
    fn modifiers_are_recognised() {
        for vk in [0x10, 0x11, 0x12, 0x14, 0x5B, 0x5C, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5] {
            assert!(is_modifier(vk), "{vk:#x}");
        }
        assert!(!is_modifier(0x41));
    }

    #[test]
    fn blank_lines_get_no_prediction() {
        assert!(!wants_prediction(&u16s("")));
        assert!(!wants_prediction(&u16s("   ")));
        assert!(!wants_prediction(&u16s("Hello there\n")));
        assert!(!wants_prediction(&u16s("Hello there\r\n    ")));
        assert!(wants_prediction(&u16s("Hello")));
        assert!(wants_prediction(&u16s("Thank you ")));
        assert!(wants_prediction(&u16s("First line\n  second ")));
    }

    #[test]
    fn request_is_length_prefixed_utf8() {
        assert_eq!(encode_request(&u16s("hé")), vec![3, 0, 0, 0, b'h', 0xC3, 0xA9]);
        assert_eq!(encode_request(&[]), vec![0, 0, 0, 0]);
        // A lone surrogate must not fail the request.
        assert_eq!(encode_request(&[0xD800]), vec![3, 0, 0, 0, 0xEF, 0xBF, 0xBD]);
    }

    #[test]
    fn reply_is_parsed() {
        assert_eq!(
            parse_reply(b"2\nknow\nknew"),
            Some(Reply { partial: 2, words: vec!["know".into(), "knew".into()] })
        );
        assert_eq!(parse_reply(b"0"), Some(Reply { partial: 0, words: vec![] }));
        assert_eq!(parse_reply(b"x\nword"), None);
        assert_eq!(parse_reply(&[0xFF]), None);
    }
}
