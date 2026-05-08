"""
Wiki-Generator für A-Schein-Fragen (questions_A.json)
======================================================

Befüllt das `wiki`-Feld jeder Frage mit:
  - explanation: Erklärung für Anfänger ("für Dummies"), 3-5 Sätze, fachlich korrekt
  - merksatz: Ein einzelner einprägsamer Satz
  - links: Bevorzugt DHV-Quellen mit Direktsprung in PDF/Webseite

Funktionsweise:
  - Idempotent: bereits gefüllte Wiki-Einträge werden NICHT überschrieben
    (außer mit --force). Skript kann jederzeit unterbrochen und fortgesetzt werden.
  - Schreibt nach jeder Frage zurück in questions_A.json (resume-safe).
  - Nutzt OpenAI Responses API (chat.completions). Kosten: ~3-5 USD bei gpt-4o-mini
    für alle 660 Fragen, ~30 Min Laufzeit.

Voraussetzungen:
  pip install openai>=1.40.0

Verwendung:
  export OPENAI_API_KEY="sk-..."
  python generate_wiki_a.py                           # alle leeren befüllen
  python generate_wiki_a.py --range 1 50              # nur A001-A050
  python generate_wiki_a.py --force                   # alle neu generieren
  python generate_wiki_a.py --model gpt-4o            # anderes Modell
  python generate_wiki_a.py --dry-run --range 1 3     # nur ausgeben, nichts speichern

Quellen-Politik:
  - DHV-Quellen sind erste Wahl (offiziell, deutschsprachig, A-Lizenz-Curriculum):
      https://www.dhv.de/piloteninfos/ausbildung/lehrplan-a-schein/
      https://www.dhv.de/piloteninfos/sicherheit/
      https://www.dhv.de/typo/Wetter.50.0.html
      https://www.dhv.de/db2/files/lehrbuch_gleitschirm_dhv.pdf
  - Bei Luftrecht: Bezug auf LuftVG/LuftVO/SERA, BAF/DFS-Quellen
  - Bei Wetter: zusätzlich DWD-Glossar oder Skybrary
  - KEINE erfundenen Links — wenn unsicher, Liste leer lassen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print("FEHLER: openai-Bibliothek fehlt. Installation: pip install openai>=1.40.0", file=sys.stderr)
    sys.exit(2)


QUESTIONS_PATH = Path("questions_A.json")
DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.4
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.4  # Sekunden, schont das Rate-Limit


SYSTEM_PROMPT = """Du bist ein erfahrener Gleitschirm-Fluglehrer (DHV) und schreibst Wiki-Erklärungen für eine deutschsprachige Lernapp für die A-Lizenz.

DEINE AUFGABE: Für die gegebene Frage einen Wiki-Eintrag im JSON-Format erstellen.

ANFORDERUNGEN AN DEINE ERKLÄRUNG:
1. Schreibe für komplette Anfänger („für Dummies"), die sich gerade auf die A-Lizenz vorbereiten.
2. Erkläre WARUM die richtige Antwort richtig ist und (kurz) WARUM die falschen Antworten falsch sind, falls hilfreich.
3. Verwende klare, einfache deutsche Sprache. Keine englischen Fachbegriffe ohne Erklärung.
4. 3 bis 5 Sätze, kompakt aber vollständig. Lieber präzise als geschwätzig.
5. Fachlich 100% korrekt — keine Verallgemeinerungen, keine Vermutungen.
6. Bezug zur Praxis: was passiert in der Luft, was sieht/spürt der Pilot, was muss er tun.

ANFORDERUNGEN AN DEN MERKSATZ:
- Ein einzelner Satz, der die Kernlernregel der Frage festhält.
- Maximal ~80 Zeichen. Einprägsam, gerne mit Eselsbrücken-Charakter.
- Beispiel-Stil: „Bremse links → Schirm dreht links." oder „Vor dem Start: Helm, Gurt, Schirm — in dieser Reihenfolge."

ANFORDERUNGEN AN DIE LINKS:
- 1 bis 2 Quellen, KEINE wenn du keine passende kennst (lieber nichts als erfunden).
- Bevorzugt DHV-Quellen (Lehrbuch, Lehrplan, Sicherheit, Wetter):
    https://www.dhv.de/piloteninfos/ausbildung/lehrplan-a-schein/
    https://www.dhv.de/piloteninfos/sicherheit/
    https://www.dhv.de/db2/files/lehrbuch_gleitschirm_dhv.pdf
- Bei Luftrecht: LuftVO, SERA, BAF.bund.de, DFS.
- Bei Wetter: zusätzlich DWD-Glossar, alpenverein.de/wetter.
- Format: { "title": "Kurzer beschreibender Titel", "url": "https://..." }
- Wenn du den Direktsprung kennst (z.B. PDF mit #page= oder konkrete Anker-URL), nutze ihn.
- ERFINDE NIEMALS URLs. Im Zweifel: leeres Array.

AUSGABEFORMAT (NUR valides JSON, keine Markdown-Codeblöcke, keinen Begleittext):
{
  "explanation": "...",
  "merksatz": "...",
  "links": [{"title": "...", "url": "https://..."}]
}
"""


def build_user_prompt(q: Dict[str, Any]) -> str:
    options_text = "\n".join(f"  {chr(65 + i)}) {opt}" for i, opt in enumerate(q.get("options", [])))
    correct_idx = int(q.get("correctIndex", 0))
    correct_letter = chr(65 + correct_idx)
    correct_text = q.get("options", [""] * 4)[correct_idx]

    parts = [
        f"KATEGORIE: {q.get('category', '')}",
        f"UNTERKAPITEL: {q.get('subchapter', '')}",
        f"FRAGE-ID: {q.get('id', '')}",
        "",
        f"FRAGE: {q.get('question', '')}",
        "",
        "ANTWORTOPTIONEN:",
        options_text,
        "",
        f"RICHTIGE ANTWORT: {correct_letter}) {correct_text}",
    ]
    return "\n".join(parts)


def call_openai(client: OpenAI, model: str, q: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate wiki entry for one question. Returns dict or None on failure."""
    user_prompt = build_user_prompt(q)
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                timeout=TIMEOUT_SECONDS,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Leere Antwort vom Modell.")
            data = json.loads(text)

            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Antwort ist kein JSON-Objekt.")
            expl = str(data.get("explanation") or "").strip()
            merk = str(data.get("merksatz") or "").strip()
            links_raw = data.get("links") or []
            if not isinstance(links_raw, list):
                links_raw = []
            links: List[Dict[str, str]] = []
            for ln in links_raw:
                if isinstance(ln, dict):
                    title = str(ln.get("title") or "").strip()
                    url = str(ln.get("url") or "").strip()
                    if title and url and url.startswith(("http://", "https://")):
                        links.append({"title": title, "url": url})

            if not expl or len(expl) < 30:
                raise ValueError(f"Erklärung zu kurz ({len(expl)} chars).")
            if not merk:
                raise ValueError("Merksatz fehlt.")

            return {"explanation": expl, "merksatz": merk, "links": links}

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                wait = 1.5 * attempt
                print(f"    ⚠ Versuch {attempt}/{MAX_RETRIES} fehlgeschlagen ({type(e).__name__}: {str(e)[:80]}), warte {wait}s …")
                time.sleep(wait)
            else:
                print(f"    ✗ Endgültig fehlgeschlagen nach {MAX_RETRIES} Versuchen: {last_err}")

    return None


def needs_generation(q: Dict[str, Any], force: bool) -> bool:
    if force:
        return True
    wiki = q.get("wiki") or {}
    expl = (wiki.get("explanation") or "").strip()
    return len(expl) < 30  # leer oder zu kurz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wiki-Generator für A-Schein-Fragen")
    p.add_argument("--input", type=Path, default=QUESTIONS_PATH,
                   help=f"questions-A.json Pfad (default: {QUESTIONS_PATH})")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"OpenAI-Modell (default: {DEFAULT_MODEL})")
    p.add_argument("--range", nargs=2, type=int, metavar=("FROM", "TO"),
                   help="Nur Fragen-Nummer FROM..TO (1-based, inkl.)")
    p.add_argument("--force", action="store_true",
                   help="Auch bereits befüllte Wiki-Einträge neu generieren")
    p.add_argument("--dry-run", action="store_true",
                   help="Nur ausgeben, nichts speichern")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("FEHLER: OPENAI_API_KEY ist nicht gesetzt.", file=sys.stderr)
        return 2

    if not args.input.exists():
        print(f"FEHLER: {args.input} nicht gefunden.", file=sys.stderr)
        return 2

    questions = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        print(f"FEHLER: {args.input} ist keine Liste.", file=sys.stderr)
        return 2

    # Range-Filter
    start, end = 1, len(questions)
    if args.range:
        start, end = args.range

    client = OpenAI()

    todo: List[int] = []
    for idx, q in enumerate(questions):
        num = idx + 1
        if num < start or num > end:
            continue
        if needs_generation(q, args.force):
            todo.append(idx)

    print(f"Modell:           {args.model}")
    print(f"Datei:            {args.input}")
    print(f"Bereich:          {start}..{end} ({end - start + 1} Fragen)")
    print(f"Bereits befüllt:  {(end - start + 1) - len(todo)}")
    print(f"Zu generieren:    {len(todo)}")
    print(f"Dry-run:          {args.dry_run}")
    print(f"Force:            {args.force}")
    print()

    if not todo:
        print("Nichts zu tun. Alle Fragen im Bereich haben bereits einen Wiki-Eintrag.")
        return 0

    success = 0
    failed: List[str] = []

    for n, idx in enumerate(todo, start=1):
        q = questions[idx]
        qid = q.get("id", f"#{idx + 1}")
        print(f"[{n}/{len(todo)}] {qid} ({q.get('subchapter', '')[:40]}) …")

        wiki = call_openai(client, args.model, q)

        if wiki is None:
            failed.append(qid)
            time.sleep(SLEEP_BETWEEN)
            continue

        if args.dry_run:
            print(f"    ✓ DRY: {wiki['merksatz']}")
        else:
            q["wiki"] = wiki
            # Inkrementelles Speichern: jede Frage einzeln, damit Abbruch
            # nicht alles verliert.
            args.input.write_text(
                json.dumps(questions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"    ✓ {wiki['merksatz'][:80]}")

        success += 1
        time.sleep(SLEEP_BETWEEN)

    print()
    print(f"Fertig: {success}/{len(todo)} erfolgreich.")
    if failed:
        print(f"Fehlgeschlagen ({len(failed)}): {', '.join(failed[:20])}{' …' if len(failed) > 20 else ''}")
        print("Tipp: Skript einfach nochmal starten — fehlende werden automatisch erneut versucht.")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
