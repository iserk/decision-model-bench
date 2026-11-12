"""A second accuracy set: eight rule-application tasks from LegalBench, in JevBench's task format.

LegalBench (Guha et al., 2023; https://hazyresearch.stanford.edu/legalbench) is a collection of
legal-reasoning tasks written and labelled by lawyers and legal scholars. Unlike JevBench, it was
not written for typed decision models, and none of the systems here was tuned on it. The eight
tasks below each state a rule and ask for its application to a short case, so they map onto the
three question types: yes/no (`noul`), choice and score. All eight are CC BY 4.0.

Each case is asked zero-shot: the rule from the task's own description goes into the question's
instructions and the case into the state. LegalBench's few-shot examples are not used. The six
diversity-jurisdiction variants have 300 cases each; a fixed 50 of each are kept, so that they
don't outweigh the other tasks.

The test files are downloaded at a pinned revision of the Hugging Face copy and checked by hash.
`fetch_tasks()` returns the joined task file and each task id's question type.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import httpx

DATASET = "nguha/legalbench"
REVISION = "daec8237410aa23e3faf4bc41ad8b3a7e1696826"
SOURCE = "https://huggingface.co/datasets/{dataset}/resolve/{rev}/data/{task}/test.tsv"
PER_DIVERSITY_VARIANT = 50
COLUMNS = ["noul", "choice", "score"]  # report columns: accuracy per question type

SHA256 = {
    "hearsay": "f6ce5ed63ba93768eafa611610312a42a1b4d70ae57addf2e04bf87cf2bd4be9",
    "personal_jurisdiction": "d3537c25a7de82386acb13eab665743c3b814c0431fa03748cd4b9d476fa2e4c",
    "diversity_1": "904fa5852012c1ee40302c269acba0eda0e085890164ced743649e92340996e4",
    "diversity_2": "4db187a5988d7fc42175a91f4ff0af2fdeb9bf75f82e5a7e35c53a59a1b78023",
    "diversity_3": "1951d1f5bd8a264e9175bafd61ce3fe8b496dcdc5987fe7747bc313104d5f6c1",
    "diversity_4": "ddbd022c5b5e585503af4d988a75c9a46ef40737352519d77cb5a0f59e135e87",
    "diversity_5": "03f102101e961d36e2e6c22dc2b8e59d2da43c62d0bb11c21473e1832e15fcd7",
    "diversity_6": "eb52b8ef28e6188b2ebbd125c60b932f6bd27389bc8586c42d423ea321fab514",
    "ucc_v_common_law": "ccb66717c227f4e8c440eb209fb371ad3cc1845290648ecb22346070024ecaf2",
    "abercrombie": "8e2e9051584fa6da66ba14a6a65310b03053af39869a5673ee43a9e378debab8",
    "insurance_policy_interpretation": "3efa5129cb7a48ccdfe7c98e684748fb73141b87e10509b78008a53e06be708b",
    "proa": "704f49fe4d9388ebad2a68b2a95ec49da24b5451219767e8ec8cf96c334154b0",
    "jcrew_blocker": "b772123270cae35705380dbf59e1dee25a4b70ee13e44d92ab1e281354061231",
}

# The rules, from each task's description in the LegalBench repository.
HEARSAY = ("Hearsay is an out-of-court statement introduced to prove the truth of the matter "
           "asserted. Is the evidence described hearsay?")
JURISDICTION = (
    "There is personal jurisdiction over a defendant in the state where the defendant is "
    "domiciled, or when (1) the defendant has sufficient contacts with the state, such that they "
    "have availed itself of the privileges of the state and (2) the claim arises out of the nexus "
    "of the defendant's contacts with the state. Is there personal jurisdiction in the state "
    "where the plaintiff sues?")
DIVERSITY = (
    "Diversity jurisdiction exists when there is (1) complete diversity between plaintiffs and "
    "defendants, and (2) the amount-in-controversy (AiC) is greater than $75k. Complete diversity "
    "requires that no plaintiff is a citizen of the same state as any defendant; several "
    "plaintiffs, or several defendants, may share a state. The AiC is the amount of damages sued "
    "for. A plaintiff may add together the values of independent claims against the same "
    "defendant, but may not aggregate claims against two separate defendants, and two plaintiffs "
    "may not aggregate claims against the same defendant. Is there diversity jurisdiction?")
UCC = ("The UCC (through Article 2) governs the sale of goods, which are defined as moveable "
       "tangible things (cars, apples, books, etc.), whereas the common law governs contracts for "
       "real estate and services. Which law governs this contract?")
ABERCROMBIE = (
    "Classify the trademark on the Abercrombie spectrum of distinctiveness, from least to most "
    "distinctive. A mark is generic if it is the common name for the product; descriptive if it "
    "describes a purpose, nature, or attribute of the product; suggestive if it suggests or "
    "implies a quality or characteristic of the product; arbitrary if it is a real English word "
    "that has no relation to the product; fanciful if it is an invented word.")
ABERCROMBIE_LEVELS = ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"]
INSURANCE = "Read the insurance policy and the claim. Is the claim covered under the policy?"
PROA = ("A private right of action is when a regular person, a private citizen, is legally "
        "entitled to enforce their rights under a given statute. Does the clause specify a "
        "private right of action?")
JCREW = ("A J.Crew blocker is a provision that typically includes (1) a prohibition on the "
         "borrower from transferring IP to an unrestricted subsidiary, and (2) a requirement that "
         "the borrower obtains the consent of its agent/lenders before transferring IP to any "
         "subsidiary. Does the provision contain a J.Crew blocker?")


def noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


# task -> (question, state from a row, expected answer from a row, labels)
TASKS = {
    "hearsay": (noul(HEARSAY), lambda r: r["text"], lambda r: r["answer"].lower(), ["no", "yes"]),
    "personal_jurisdiction": (noul(JURISDICTION), lambda r: r["text"],
                              lambda r: r["answer"].lower(), ["no", "yes"]),
    **{f"diversity_{i}": (noul(DIVERSITY), lambda r: r["text"], lambda r: r["answer"].lower(),
                          ["no", "yes"]) for i in range(1, 7)},
    "ucc_v_common_law": (
        {"type": "choice", "instructions": UCC,
         "criteria": {"ucc": "The UCC: a sale of goods",
                      "common_law": "The common law: real estate or services"}},
        lambda r: r["contract"].removesuffix(
            " Is this contract governed by the UCC or the common law?"),
        lambda r: {"UCC": "ucc", "Common Law": "common_law"}[r["answer"]], ["ucc", "common_law"]),
    "abercrombie": (
        {"type": "score", "instructions": ABERCROMBIE, "criteria": ABERCROMBIE_LEVELS},
        lambda r: r["text"], lambda r: ABERCROMBIE_LEVELS.index(r["answer"]),
        [str(i) for i in range(len(ABERCROMBIE_LEVELS))]),
    "insurance_policy_interpretation": (
        {"type": "choice", "instructions": INSURANCE,
         "criteria": {"covered": "Yes, the claim is covered",
                      "not_covered": "No, the claim is not covered",
                      "ambiguous": "It is ambiguous whether the claim is covered"}},
        lambda r: f"Policy: {r['policy']}\nClaim: {r['claim']}",
        lambda r: {"A": "covered", "B": "not_covered", "C": "ambiguous"}[r["answer"]],
        ["covered", "not_covered", "ambiguous"]),
    "proa": (noul(PROA), lambda r: r["text"], lambda r: r["answer"].lower(), ["no", "yes"]),
    "jcrew_blocker": (noul(JCREW), lambda r: r["text"], lambda r: r["answer"].lower(),
                      ["no", "yes"]),
}


def keep(task: str, rows: list[dict]) -> list[dict]:
    """All rows, except a fixed sample (by hash of task and index) of each diversity variant."""
    if not task.startswith("diversity_"):
        return rows
    key = lambda r: hashlib.sha256(f"{task}:{r['index']}".encode()).hexdigest()  # noqa: E731
    return sorted(sorted(rows, key=key)[:PER_DIVERSITY_VARIANT], key=lambda r: int(r["index"]))


def convert(task: str, rows: list[dict]) -> list[dict]:
    question, state, expected, labels = TASKS[task]
    return [{"id": f"legalbench-{task}-{r['index']}", "family": task, "group": None,
             "split": "public", "question": question, "state": state(r), "expected": expected(r),
             "labels": labels,
             "provenance": {"source": f"LegalBench {task} (test), {DATASET}@{REVISION[:7]}",
                            "license": "CC-BY-4.0", "label_basis": "LegalBench's expert label",
                            "exclude_reason": None}}
            for r in keep(task, rows)]


def fetch_tasks(root: Path) -> tuple[Path, dict[str, str]]:
    """Download the test files at the pinned revision (once), check their hashes, and convert
    them. Returns the task file and each task id's question type."""
    data = root / "data" / f"legalbench-{REVISION[:7]}"
    data.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task, sha in SHA256.items():
        path = data / f"{task}.tsv"
        if not path.exists():
            url = SOURCE.format(dataset=DATASET, rev=REVISION, task=task)
            print(f"downloading {url}", flush=True)
            r = httpx.get(url, timeout=60, follow_redirects=True)
            r.raise_for_status()
            path.write_bytes(r.content)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit(f"{path}: sha256 mismatch; delete it and rerun")
        tasks += convert(task, list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t")))
    joined = data / "tasks.jsonl"
    joined.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    return joined, {t["id"]: t["question"]["type"] for t in tasks}


if __name__ == "__main__":  # print the set's make-up
    from collections import Counter
    path, types = fetch_tasks(Path(__file__).parent)
    rows = [json.loads(line) for line in open(path)]
    print(f"{len(rows)} tasks in {path}")
    for (family, kind), n in sorted(Counter((r["family"], r["question"]["type"]) for r in rows).items()):
        print(f"  {family:<34}{kind:<8}{n:>4}")
