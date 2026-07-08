"""Assert-based self-check for the papers/extraction logic in main.py.

Run directly: `poetry run python -m test_main`
"""
from datetime import datetime
from unittest.mock import patch

import main


class FakeCursor:
    """In-memory stand-in for a psycopg cursor, just enough for these queries."""

    def __init__(self, papers, runs=None):
        self._papers = papers
        self._runs = runs if runs is not None else []
        self._tags = []
        self._next_run_id = max((r["id"] for r in self._runs), default=0) + 1
        self._next_tag_id = 1
        self._result = None

    def _latest_run(self, paper_id):
        matches = [r for r in self._runs if r["paper_id"] == paper_id]
        return matches[-1] if matches else None

    def execute(self, sql, params=()):
        sql = " ".join(sql.split())
        if sql.startswith("SELECT p.id, p.title"):
            rows = []
            for p in self._papers:
                run = self._latest_run(p["id"])
                rows.append((
                    p["id"], p["title"], p["authors"], p["query"], p["open_access"], p["pdf_url"],
                    run["status"] if run else None,
                    run["error"] if run else None,
                    run["started_at"] if run else None,
                ))
            self._result = rows
        elif sql.startswith("SELECT pdf_url"):
            match = next((p for p in self._papers if p["id"] == params[0]), None)
            self._result = (match["pdf_url"],) if match else None
        elif sql.startswith("INSERT INTO extraction_runs"):
            paper_id, status = params
            run_id = self._next_run_id
            self._next_run_id += 1
            self._runs.append({
                "id": run_id, "paper_id": paper_id, "status": status,
                "error": None, "raw_result": None, "started_at": datetime.now(), "finished_at": None,
            })
            self._result = (run_id,)
        elif sql.startswith("UPDATE extraction_runs"):
            status, error, raw_result, finished_at, run_id = params
            run = next(r for r in self._runs if r["id"] == run_id)
            run["status"] = status
            run["error"] = error
            if raw_result is not None:
                run["raw_result"] = raw_result.obj
            if finished_at is not None:
                run["finished_at"] = finished_at
        elif sql.startswith("SELECT raw_result FROM extraction_runs"):
            done_runs = [r for r in self._runs if r["paper_id"] == params[0] and r["status"] == "done"]
            self._result = (done_runs[-1]["raw_result"],) if done_runs else None
        elif sql.startswith("SELECT id FROM extraction_runs"):
            done_runs = [r for r in self._runs if r["paper_id"] == params[0] and r["status"] == "done"]
            self._result = (done_runs[-1]["id"],) if done_runs else None
        elif sql.startswith("INSERT INTO tags"):
            tag = dict(params)
            if tag.get("extra_data") is not None and hasattr(tag["extra_data"], "obj"):
                tag["extra_data"] = tag["extra_data"].obj
            tag["id"] = self._next_tag_id
            self._next_tag_id += 1
            tag.setdefault("review_status", "pending")
            tag.setdefault("edited_value", None)
            tag.setdefault("added_manually", False)
            self._tags.append(tag)
        elif sql.startswith("SELECT id, tag_type"):
            run_id = params[0]
            rows = [
                tuple(tag[col] for col in main._TAG_READ_COLUMNS)
                for tag in self._tags
                if tag["extraction_run_id"] == run_id
            ]
            self._result = rows

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    """Just enough of psycopg's Connection/cursor context-manager protocol for tests."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor


def main_() -> None:
    # get_papers() reports "pending" for papers with no extraction run yet.
    cursor = FakeCursor([
        {"id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True, "pdf_url": "http://x/a.pdf"},
    ])
    assert main.get_papers(cursor) == [{
        "id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True,
        "pdf_url": "http://x/a.pdf", "extraction_status": "pending", "extraction_error": None,
        "extraction_started_at": None,
    }]

    # labels_to_tags() flattens the service's {"labels": {section: [decision, ...]}}
    # shape into one tag row per decision, regardless of how many sections there are.
    label_result = {
        "paper_id": 1,
        "labels": {
            "Intro": [{
                "label": "governance", "verdict": "YES", "anchor_text": "the city council",
                "context": "As mandated by the city council in 2019...", "match_score": 0.92,
                "rationale": "Discusses local governance policy.",
            }],
            "Methods": [{
                "label": "empirical_urban_environment", "verdict": "UNVERIFIED", "anchor_text": "field measurements",
                "context": "Field measurements were taken across the site.", "match_score": 0.41,
                "rationale": None,
            }],
        },
    }
    tags = main.labels_to_tags(7, label_result)
    assert len(tags) == 2
    assert tags[0] == main._tag(
        extraction_run_id=7, tag_type="label", value="governance",
        anchor_text="the city council", context="As mandated by the city council in 2019...",
        match_score=0.92, rationale="Discusses local governance policy.", verified=True,
    )
    assert tags[1]["value"] == "empirical_urban_environment"
    assert tags[1]["verified"] is False  # UNVERIFIED decisions are not "YES"

    # entities_to_tags() flattens each entity's per-field {value, context, ...}
    # dicts into one tag row per field, clustered by a shared group_id — and
    # skips flat (non-dict) fields like "type".
    entity_result = {"entities": [
        {
            "type": "Building",
            "name": {"value": "Casa Verde", "context": "the Casa Verde building", "context_verified": True, "context_match_score": 0.95},
            "city": {"value": "Lisbon", "context": None, "context_verified": None, "context_match_score": None},
        },
        {"name": {"value": "Building B", "context": "Building B was studied", "context_verified": False, "context_match_score": 0.2}},
    ]}
    entity_tags = main.entities_to_tags(7, entity_result)
    assert len(entity_tags) == 3  # name+city for entity 0, name for entity 1 ("type" is skipped)
    assert entity_tags[0]["group_id"] == 0 and entity_tags[0]["field"] == "name" and entity_tags[0]["value"] == "Casa Verde"
    assert entity_tags[1]["group_id"] == 0 and entity_tags[1]["field"] == "city"
    assert entity_tags[2]["group_id"] == 1 and entity_tags[2]["value"] == "Building B"

    # design_strategies_to_tags() / ecosystem_services_to_tags() put
    # implementation_details/vocab_top_matches in extra_data, not flat columns.
    design_result = {"design_strategies": [{
        "name": "green roof", "anchor_text": "extensive green roof", "context": "an extensive green roof was installed",
        "anchor_verified": True, "anchor_match_score": 0.88,
        "implementation_details": ["30cm substrate depth"], "vocab_top_matches": [{"name": "green roof", "score": 0.99}],
    }]}
    design_tags = main.design_strategies_to_tags(7, design_result)
    assert design_tags[0]["tag_type"] == "design_strategy"
    assert design_tags[0]["extra_data"] == {
        "implementation_details": ["30cm substrate depth"],
        "vocab_top_matches": [{"name": "green roof", "score": 0.99}],
    }

    eco_result = {"ecosystem_services": [{
        "name": "stormwater retention", "category": "regulating", "anchor_text": "retains stormwater",
        "context": "the system retains stormwater during peak rainfall", "anchor_verified": True, "anchor_match_score": 0.81,
        "vocab_top_matches": [{"name": "stormwater retention", "score": 0.97, "category": "regulating"}],
    }]}
    eco_tags = main.ecosystem_services_to_tags(7, eco_result)
    assert eco_tags[0]["tag_type"] == "ecosystem_service" and eco_tags[0]["category"] == "regulating"

    # extraction_result_to_tags() combines all four mappers over one result dict.
    combined_result = {**label_result, **entity_result, **design_result, **eco_result}
    combined_tags = main.extraction_result_to_tags(7, combined_result)
    assert len(combined_tags) == len(tags) + len(entity_tags) + len(design_tags) + len(eco_tags)

    # Happy path: download succeeds, extraction service accepts it -> "done",
    # its result JSON is persisted to the (fake) database, and every extractor's
    # output (labels, entities, design strategies, ecosystem services) lands in
    # the tags table too, not just as raw_result JSON.
    cursor = FakeCursor([{"id": 1, "title": "A", "authors": [], "query": "q", "open_access": True, "pdf_url": "http://x/a.pdf"}])
    run_id = main.create_run(cursor, 1, "downloading")
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)), \
         patch("main.download_pdf", lambda url: b"%PDF-1.4 fake"), \
         patch("main.notify_extraction", lambda paper_id, pdf_bytes: combined_result):
        main.run_extraction(run_id, 1)
    run = cursor._latest_run(1)
    assert run["status"] == "done"
    assert run["raw_result"] == combined_result
    assert run["finished_at"] is not None
    assert [{k: t[k] for k in main._TAG_FIELDS} for t in cursor._tags] == main.extraction_result_to_tags(run_id, combined_result)

    # get_latest_done_run_id() + get_tags() are what GET /papers/{id}/results now
    # reads from — every inserted tag should come back with its DB-side defaults.
    found_run_id = main.get_latest_done_run_id(cursor, 1)
    assert found_run_id == run_id
    read_tags = main.get_tags(cursor, found_run_id)
    assert len(read_tags) == len(combined_tags)
    assert {t["tag_type"] for t in read_tags} == {"label", "entity", "design_strategy", "ecosystem_service"}
    assert all(t["review_status"] == "pending" and t["added_manually"] is False for t in read_tags)

    # No pdf_url on record -> "failed" without ever calling the network.
    cursor = FakeCursor([{"id": 2, "title": "B", "authors": [], "query": "q", "open_access": False, "pdf_url": None}])
    run_id = main.create_run(cursor, 2, "downloading")
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)):
        main.run_extraction(run_id, 2)
    assert cursor._latest_run(2)["status"] == "failed"

    # Download succeeds but knowledge-extraction is unreachable -> "failed", not "extracting" forever.
    def _unreachable(paper_id, pdf_bytes):
        raise ConnectionRefusedError("no such service")

    cursor = FakeCursor([{"id": 3, "title": "C", "authors": [], "query": "q", "open_access": True, "pdf_url": "http://x/c.pdf"}])
    run_id = main.create_run(cursor, 3, "downloading")
    with patch("main.get_db_connection", lambda: FakeConnection(cursor)), \
         patch("main.download_pdf", lambda url: b"%PDF-1.4 fake"), \
         patch("main.notify_extraction", _unreachable):
        main.run_extraction(run_id, 3)
    run = cursor._latest_run(3)
    assert run["status"] == "failed"
    assert "unreachable" in run["error"]

    print("backend papers/extraction self-check passed")


if __name__ == "__main__":
    main_()
