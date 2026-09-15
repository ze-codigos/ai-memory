//! Naming for a workstream adopted from a session the launcher did not start.
//!
//! The desktop app titles a session from its first turn, so the good name only
//! exists once the user has spoken. When there is no app title — a bare
//! `claude`, an IDE extension — a slug of that first prompt stands in, flagged
//! `provisional` so the hook can ask for a better one.
//!
//! Never fails: a workstream cannot be opened without a name, and the standing
//! policy is that memory never blocks the developer.

use std::path::{Path, PathBuf};

/// Three words names the work without turning into a sentence.
const SLUG_WORDS: usize = 3;
/// What `run --new` and `rename-workstream --to` already validate.
const NAME_MAX: usize = 128;
/// Last resort: a prompt with no usable character in it.
const SLUG_FALLBACK: &str = "sessao";
/// The app's session store, relative to the platform config dir.
const APP_SESSIONS: &str = "Claude/claude-code-sessions";
/// `<config>/Claude/claude-code-sessions/<account>/<workspace>/<id>.json` — two
/// levels of UUID we never see in the environment. Bounded so a layout change
/// cannot turn this into a walk of the whole config directory.
const SESSION_SEARCH_DEPTH: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedName {
    pub name: String,
    /// A slug stands in until something better is known; an app title does not.
    pub provisional: bool,
}

/// Kebab-case slug of the prompt's first words.
pub(crate) fn slugify_prompt(prompt: &str) -> String {
    let slug = prompt
        .chars()
        .map(deaccent)
        .collect::<String>()
        .split_whitespace()
        .map(|word| {
            word.chars()
                .filter(char::is_ascii_alphanumeric)
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .take(SLUG_WORDS)
        .collect::<Vec<_>>()
        .join("-");
    if slug.is_empty() {
        return SLUG_FALLBACK.to_string();
    }
    truncate_name(&slug)
}

/// Minimal transliteration, only what PT-BR actually produces. Deliberately not
/// full Unicode decomposition: the goal is a readable workstream name, not
/// correct normalisation of arbitrary text.
fn deaccent(c: char) -> char {
    match c {
        'á' | 'à' | 'â' | 'ã' | 'ä' => 'a',
        'é' | 'ê' | 'è' | 'ë' => 'e',
        'í' | 'î' | 'ì' | 'ï' => 'i',
        'ó' | 'ô' | 'õ' | 'ò' | 'ö' => 'o',
        'ú' | 'û' | 'ù' | 'ü' => 'u',
        'ç' => 'c',
        'Á' | 'À' | 'Â' | 'Ã' | 'Ä' => 'A',
        'É' | 'Ê' | 'È' | 'Ë' => 'E',
        'Í' | 'Î' | 'Ì' | 'Ï' => 'I',
        'Ó' | 'Ô' | 'Õ' | 'Ò' | 'Ö' => 'O',
        'Ú' | 'Û' | 'Ù' | 'Ü' => 'U',
        'Ç' => 'C',
        other => other,
    }
}

/// Cut on a character boundary; the limit is in bytes.
fn truncate_name(name: &str) -> String {
    if name.len() <= NAME_MAX {
        return name.to_string();
    }
    let mut cut = NAME_MAX;
    while cut > 0 && !name.is_char_boundary(cut) {
        cut -= 1;
    }
    name[..cut].to_string()
}

/// The title the desktop app generated for this session, if it exists yet.
///
/// `config_dir` is the platform config root (`dirs::config_dir()`), which is
/// the Electron `userData` parent on all three platforms: `~/.config` on
/// Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows.
pub(crate) fn desktop_title(host_session_id: &str, config_dir: &Path) -> Option<String> {
    let root = config_dir.join(APP_SESSIONS);
    let wanted = format!("{host_session_id}.json");
    let file = find_file(&root, &wanted, SESSION_SEARCH_DEPTH)?;
    let raw = std::fs::read_to_string(file).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let title = value.get("title")?.as_str()?.trim();
    if title.is_empty() {
        return None;
    }
    let cleaned = sanitize(title);
    if cleaned.is_empty() {
        return None;
    }
    Some(truncate_name(&cleaned))
}

/// Drop what name validation rejects: control characters and separators.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_control() || c == '/' || c == '\\' {
                ' '
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn find_file(root: &Path, wanted: &str, depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, wanted, depth - 1) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(wanted) {
            return Some(path);
        }
    }
    None
}

/// The workstream's name. Always returns something valid.
pub(crate) fn resolve_name(
    host_session_id: Option<&str>,
    config_dir: &Path,
    first_prompt: &str,
) -> ResolvedName {
    if let Some(name) = host_session_id.and_then(|id| desktop_title(id, config_dir)) {
        return ResolvedName {
            name,
            provisional: false,
        };
    }
    ResolvedName {
        name: slugify_prompt(first_prompt),
        provisional: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_session(config_dir: &Path, host: &str, body: &str) {
        let dir = config_dir.join("Claude/claude-code-sessions/conta/ws");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{host}.json")), body).unwrap();
    }

    #[test]
    fn slug_takes_first_three_words_kebab_lowercase() {
        assert_eq!(
            slugify_prompt("Quero conseguir usar o meu setup"),
            "quero-conseguir-usar"
        );
    }

    #[test]
    fn slug_strips_accents_and_punctuation() {
        assert_eq!(
            slugify_prompt("Corrigir emissão, urgente!"),
            "corrigir-emissao-urgente"
        );
    }

    #[test]
    fn slug_never_empty() {
        assert_eq!(slugify_prompt("!!! ???"), "sessao");
    }

    #[test]
    fn slug_is_bounded() {
        assert!(slugify_prompt(&"palavra ".repeat(100)).len() <= 128);
    }

    #[test]
    fn desktop_title_reads_app_session_file_under_config_dir() {
        let tmp = tempfile::tempdir().unwrap();
        app_session(
            tmp.path(),
            "local_abc",
            r#"{"title":"Ajuste no checkout","titleSource":"auto"}"#,
        );
        assert_eq!(
            desktop_title("local_abc", tmp.path()).as_deref(),
            Some("Ajuste no checkout")
        );
    }

    #[test]
    fn desktop_title_none_without_title_field() {
        let tmp = tempfile::tempdir().unwrap();
        app_session(tmp.path(), "local_abc", r#"{"cliSessionId":"nat-1"}"#);
        assert_eq!(desktop_title("local_abc", tmp.path()), None);
    }

    #[test]
    fn desktop_title_none_for_unknown_session() {
        let tmp = tempfile::tempdir().unwrap();
        app_session(tmp.path(), "local_abc", r#"{"title":"Ajuste"}"#);
        assert_eq!(desktop_title("local_OUTRA", tmp.path()), None);
    }

    #[test]
    fn desktop_title_strips_slashes_and_control_chars() {
        let tmp = tempfile::tempdir().unwrap();
        app_session(tmp.path(), "local_abc", "{\"title\":\"a/b\\tc\"}");
        assert_eq!(
            desktop_title("local_abc", tmp.path()).as_deref(),
            Some("a b c")
        );
    }

    #[test]
    fn resolve_prefers_desktop_title_and_is_not_provisional() {
        let tmp = tempfile::tempdir().unwrap();
        app_session(tmp.path(), "local_abc", r#"{"title":"Ajuste no checkout"}"#);
        let r = resolve_name(Some("local_abc"), tmp.path(), "qualquer prompt");
        assert_eq!(r.name, "Ajuste no checkout");
        assert!(!r.provisional);
    }

    #[test]
    fn resolve_slug_is_provisional() {
        let tmp = tempfile::tempdir().unwrap();
        let r = resolve_name(None, tmp.path(), "Corrigir o timeout do nexus");
        assert_eq!(r.name, "corrigir-o-timeout");
        assert!(r.provisional);
    }
}
