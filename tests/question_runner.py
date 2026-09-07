"""Score the answers, not the counts.

Every other number this project records describes the graph: how many nodes,
how many edges, how many calls resolved. None of them describes the only thing
a user ever sees, which is what comes back when they ask a question. A stage
was once marked finished while returning `llm.py` for a question whose answer
was in `detect.py`, and every recorded metric was correct at the time.

So this runner asks the questions in `questions.json` -- written by hand, with
the right answer decided by opening the source first -- and scores one thing:

    is the named answer the FIRST result, and is it in the top three.

The questions are asked exactly as a person would type them, whole sentences
included, because that is what `graph-paat query how are files classified`
receives and the gap between an English question and a code identifier is a
real part of the problem.

    python -m tests.question_runner              # score, and compare to the baseline
    python -m tests.question_runner --record     # accept the current ranks as truth
    python -m tests.question_runner django       # one corpus
    python -m tests.question_runner --misses     # print what came back instead
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from graphpaat.build import assemble
from graphpaat.query import match

HERE = Path(__file__).parent
QUESTIONS = HERE / "questions.json"
BASELINE = HERE / "baselines" / "questions.json"
# The held-out set, written by agents who never saw this tool, and never tuned
# against. `--heldout` swaps both files: a separate baseline so the two scores
# can never be confused for one another.
HELDOUT = HERE / "questions-heldout.json"
HELDOUT_BASELINE = HERE / "baselines" / "questions-heldout.json"
CONFIG = HERE / "corpora.json"

TOP_N = 3           # "in the top three" -- an agent reads a handful, not one


def load_questions(path: Path | None = None) -> dict[str, list[dict]]:
    return json.loads((path or QUESTIONS).read_text())["questions"]


def load_corpora() -> dict[str, Path]:
    if not CONFIG.exists():
        print(f"no {CONFIG.name}; copy corpora.sample.json and set the paths",
              file=sys.stderr)
        raise SystemExit(2)
    return {name: Path(path) for name, path in json.loads(CONFIG.read_text()).items()}


def where(node: dict) -> str:
    """How an answer is written in questions.json: `path/to/file.ext:Label`."""
    return f"{node['file']}:{node['label']}"


def ask(graph: dict, question: dict, node_of: dict[str, dict]) -> tuple[int, list[str]]:
    """Rank of the first acceptable answer, and what was actually returned.

    Returns -1 when no acceptable answer is in the top N, which is a miss.
    """
    wanted = set(question["answers"])
    seeds, _ = match(graph, question["q"].split(), limit=TOP_N)
    got = [where(node_of[s]) for s in seeds if s in node_of]
    for rank, place in enumerate(got):
        if place in wanted:
            return rank, got
    return -1, got


def score_corpus(name: str, root: Path, questions: list[dict]) -> tuple[dict, list[str]]:
    """Ask one corpus its questions. Returns (rank per question, problems)."""
    built = assemble(root)
    graph = built.payload()
    node_of = {n["id"]: n for n in graph["nodes"]}
    known = {where(n) for n in graph["nodes"]}

    ranks, problems = {}, []
    for question in questions:
        # A question whose answer is not in the graph is a broken question, not
        # a low score. Say so loudly: it means the answer was mistyped, or the
        # corpus moved underneath it, and every run after this would quietly
        # count an unanswerable question as a failure of ranking.
        missing = [a for a in question["answers"] if a not in known]
        if missing:
            problems.append(f"  {name}: no such node {missing} "
                            f"for \"{question['q']}\"")
            continue
        rank, got = ask(graph, question, node_of)
        ranks[question["q"]] = {"rank": rank, "got": got}
    return ranks, problems


def summarise(ranks: dict) -> tuple[int, int, int]:
    first = sum(1 for r in ranks.values() if r["rank"] == 0)
    top = sum(1 for r in ranks.values() if r["rank"] >= 0)
    return first, top, len(ranks)


def main(argv: list[str]) -> int:
    record = "--record" in argv
    show_misses = "--misses" in argv
    heldout = "--heldout" in argv
    only = [a for a in argv if not a.startswith("--")]

    source, baseline = ((HELDOUT, HELDOUT_BASELINE) if heldout
                        else (QUESTIONS, BASELINE))
    questions = load_questions(source)
    corpora = load_corpora()
    if only:
        questions = {k: v for k, v in questions.items() if k in only}

    results: dict[str, dict] = {}
    problems: list[str] = []
    misses: list[str] = []

    for name in sorted(questions):
        root = corpora.get(name)
        if root is None:
            problems.append(f"  {name}: has questions but is not in corpora.json")
            continue
        if not root.exists():
            print(f"{name:12s} SKIP  (not installed at {root})")
            continue

        ranks, found = score_corpus(name, root, questions[name])
        problems.extend(found)
        results[name] = ranks
        first, top, total = summarise(ranks)
        print(f"{name:12s} first {first:2d}/{total:2d}   top{TOP_N} {top:2d}/{total:2d}")
        for q, r in ranks.items():
            if r["rank"] != 0:
                got = r["got"][0] if r["got"] else "nothing"
                misses.append(f'  {name}: "{q}"\n'
                              f'      wanted {" or ".join(_wanted(questions[name], q))}\n'
                              f'      got    {got}')

    if not results:
        print("no corpora available to score")
        return 0

    every = {q: r for ranks in results.values() for q, r in ranks.items()}
    first, top, total = summarise(every)
    print(f"\n{'ALL':12s} first {first}/{total} ({100 * first // max(total, 1)}%)"
          f"   top{TOP_N} {top}/{total} ({100 * top // max(total, 1)}%)")

    if show_misses and misses:
        print(f"\n{len(misses)} question(s) not answered first:\n")
        print("\n".join(misses))

    if problems:
        print("\nBroken questions -- these are not scores, they are bugs:\n")
        print("\n".join(problems))
        return 1

    flat = {name: {q: r["rank"] for q, r in ranks.items()}
            for name, ranks in results.items()}
    if record or not baseline.exists():
        baseline.parent.mkdir(exist_ok=True)
        baseline.write_text(json.dumps(flat, indent=1, sort_keys=True) + "\n")
        print(f"\nrecorded {total} question(s) to {baseline.name}")
        return 0

    return _compare(flat, json.loads(baseline.read_text()))


def _wanted(corpus_questions: list[dict], q: str) -> list[str]:
    for question in corpus_questions:
        if question["q"] == q:
            return question["answers"]
    return []


def _compare(now: dict, before: dict) -> int:
    """A question that got worse is a regression, whatever the total did.

    Totals hide swaps: one question improving while another breaks leaves the
    percentage flat. So every question is compared on its own.
    """
    worse, better = [], []
    for name, ranks in sorted(now.items()):
        for q, rank in sorted(ranks.items()):
            was = before.get(name, {}).get(q)
            if was is None or was == rank:
                continue
            # -1 is a miss, so it sorts below every real rank.
            got_worse = rank == -1 or (was != -1 and rank > was)
            (worse if got_worse else better).append(
                f"  {name}: \"{q}\"  {_rank(was)} -> {_rank(rank)}")

    for line in better:
        print(f"BETTER {line}")
    if worse:
        print(f"\n{len(worse)} question(s) answered worse than the baseline:\n")
        print("\n".join(worse))
        print("\nIf every change above is intended, re-run with --record.")
        return 1
    if better:
        print("\nNothing got worse. Re-run with --record to accept the improvements.")
    return 0


def _rank(rank: int) -> str:
    return "miss" if rank == -1 else f"#{rank + 1}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
