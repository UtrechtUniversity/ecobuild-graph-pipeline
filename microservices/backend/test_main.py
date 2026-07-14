"""Assert-based self-check for the papers/extraction logic in main.py.

Run directly: `poetry run python -m test_main`
"""
import os
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

    def _search_filter(self, like_params):
        papers = self._papers
        if like_params:
            needle = like_params[0].strip("%").lower()
            def matches(p):
                fields = [p.get("title"), p.get("abstract"), p.get("query"), *p.get("authors", [])]
                return any(needle in (f or "").lower() for f in fields)
            papers = [p for p in papers if matches(p)]
        return papers

    def execute(self, sql, params=()):
        sql = " ".join(sql.split())
        if sql.startswith("SELECT COUNT(*) FROM papers"):
            self._result = (len(self._search_filter(list(params))),)
        elif sql.startswith("SELECT p.id, p.title"):
            *like_params, limit, offset = params
            papers = self._search_filter(like_params)
            rows = []
            for p in papers[offset:offset + limit]:
                run = self._latest_run(p["id"])
                rows.append((
                    p["id"], p["title"], p["authors"], p["query"], p["open_access"], p["pdf_url"],
                    p.get("source"), p.get("external_id"), p.get("url"), p.get("doi"),
                    p.get("venue"), p.get("citation_count"), p.get("abstract"),
                    p.get("relevance_checked"), p.get("relevant"), p.get("created_at"),
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
                "error": None, "raw_result": None, "raw_text": None,
                "started_at": datetime.now(), "finished_at": None,
            })
            self._result = (run_id,)
        elif sql.startswith("UPDATE extraction_runs"):
            status, error, raw_result, raw_text, finished_at, run_id = params
            run = next(r for r in self._runs if r["id"] == run_id)
            run["status"] = status
            run["error"] = error
            if raw_result is not None:
                run["raw_result"] = raw_result.obj
            if raw_text is not None:
                run["raw_text"] = raw_text
            if finished_at is not None:
                run["finished_at"] = finished_at
        elif sql.startswith("SELECT 1 FROM extraction_runs"):
            in_progress = [
                r for r in self._runs
                if r["paper_id"] == params[0] and r["status"] in ("downloading", "extracting")
            ]
            self._result = (1,) if in_progress else None
        elif sql.startswith("SELECT id FROM extraction_runs"):
            done_runs = [r for r in self._runs if r["paper_id"] == params[0] and r["status"] == "done"]
            self._result = (done_runs[-1]["id"],) if done_runs else None
        elif sql.startswith("SELECT raw_text FROM extraction_runs"):
            run = next((r for r in self._runs if r["id"] == params[0]), None)
            self._result = (run["raw_text"],) if run else None
        elif sql.startswith("INSERT INTO tags"):
            tag = dict(params)
            if tag.get("extra_data") is not None and hasattr(tag["extra_data"], "obj"):
                tag["extra_data"] = tag["extra_data"].obj
            if tag.get("bbox") is not None and hasattr(tag["bbox"], "obj"):
                tag["bbox"] = tag["bbox"].obj
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
        elif sql.startswith("UPDATE tags SET review_status"):
            status, tag_id = params
            tag = next((t for t in self._tags if t["id"] == tag_id), None)
            if tag is not None:
                tag["review_status"] = status
            self._result = (tag_id,) if tag is not None else None
        elif sql.startswith("UPDATE tags SET edited_value"):
            edited_value, tag_id = params
            tag = next((t for t in self._tags if t["id"] == tag_id), None)
            if tag is not None:
                tag["edited_value"] = edited_value
                tag["review_status"] = "edited"
            self._result = (tag_id,) if tag is not None else None

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
        {
            "id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True, "pdf_url": "http://x/a.pdf",
            "source": "semantic_scholar", "external_id": "ss1", "url": "http://semanticscholar.org/a", "doi": "10.1/a",
            "venue": "Journal of X", "citation_count": 5, "abstract": "An abstract.",
            "relevance_checked": True, "relevant": True, "created_at": None,
        },
    ])
    assert main.get_papers(cursor) == [{
        "id": 1, "title": "A", "authors": ["X"], "query": "q", "open_access": True,
        "pdf_url": "http://x/a.pdf", "source": "semantic_scholar", "external_id": "ss1",
        "url": "http://semanticscholar.org/a", "doi": "10.1/a", "venue": "Journal of X", "citation_count": 5,
        "abstract": "An abstract.", "relevance_checked": True, "relevant": True,
        "created_at": None, "extraction_status": "pending", "extraction_error": None,
        "extraction_started_at": None,
    }]

    # get_papers() pages through results with limit/offset, ordered by id.
    paged_cursor = FakeCursor([{"id": i, "title": f"P{i}", "authors": [], "query": None,
                                 "open_access": None, "pdf_url": None, "created_at": None} for i in range(1, 6)])
    assert [p["id"] for p in main.get_papers(paged_cursor, limit=2, offset=0)] == [1, 2]
    assert [p["id"] for p in main.get_papers(paged_cursor, limit=2, offset=4)] == [5]
    assert main.get_papers(paged_cursor, limit=2, offset=10) == []

    # get_papers() search (q) matches title, abstract, matched-query, or any author,
    # case-insensitively, regardless of which field it hits.
    search_cursor = FakeCursor([
        {"id": 1, "title": "Urban Heat Islands", "authors": ["Jane Doe"], "query": None,
         "abstract": "cooling strategies", "open_access": None, "pdf_url": None, "created_at": None},
        {"id": 2, "title": "Unrelated Paper", "authors": ["Bob Smith"], "query": '"heat pump"',
         "abstract": None, "open_access": None, "pdf_url": None, "created_at": None},
        {"id": 3, "title": "Also Unrelated", "authors": ["Ann Heatly"], "query": None,
         "abstract": None, "open_access": None, "pdf_url": None, "created_at": None},
        {"id": 4, "title": "No Match Here", "authors": ["Someone Else"], "query": None,
         "abstract": None, "open_access": None, "pdf_url": None, "created_at": None},
    ])
    assert [p["id"] for p in main.get_papers(search_cursor, limit=50, offset=0, q="heat")] == [1, 2, 3]

    # count_papers() reports the total match count regardless of limit/offset.
    assert main.count_papers(paged_cursor) == 5
    assert main.count_papers(search_cursor, q="heat") == 3

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
    # page_number/bbox (from service.py's PDF location capture) pass through too.
    design_result = {"design_strategies": [{
        "name": "green roof", "anchor_text": "extensive green roof", "context": "an extensive green roof was installed",
        "anchor_verified": True, "anchor_match_score": 0.88,
        "implementation_details": ["30cm substrate depth"], "vocab_top_matches": [{"name": "green roof", "score": 0.99}],
        "page_number": 3, "bbox": {"x0": 10.0, "y0": 20.0, "x1": 100.0, "y1": 40.0},
    }]}
    design_tags = main.design_strategies_to_tags(7, design_result)
    assert design_tags[0]["tag_type"] == "design_strategy"
    assert design_tags[0]["extra_data"] == {
        "implementation_details": ["30cm substrate depth"],
        "vocab_top_matches": [{"name": "green roof", "score": 0.99}],
    }
    assert design_tags[0]["page_number"] == 3
    assert design_tags[0]["bbox"] == {"x0": 10.0, "y0": 20.0, "x1": 100.0, "y1": 40.0}

    eco_result = {"ecosystem_services": [{
        "name": "stormwater retention", "category": "regulating", "anchor_text": "retains stormwater",
        "context": "the system retains stormwater during peak rainfall", "anchor_verified": True, "anchor_match_score": 0.81,
        "vocab_top_matches": [{"name": "stormwater retention", "score": 0.97, "category": "regulating"}],
    }]}
    eco_tags = main.ecosystem_services_to_tags(7, eco_result)
    assert eco_tags[0]["tag_type"] == "ecosystem_service" and eco_tags[0]["category"] == "regulating"

    # extraction_result_to_tags() combines all four mappers over one result dict.
    combined_result = {
        **label_result, **entity_result, **design_result, **eco_result,
        "raw_text": "Full paper text. As mandated by the city council in 2019, the roof was retrofitted.",
    }
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
    assert main.get_raw_text(cursor, run_id) == combined_result["raw_text"]
    inserted_keys = main._TAG_FIELDS + ["review_status", "added_manually"]
    assert [{k: t[k] for k in inserted_keys} for t in cursor._tags] == main.extraction_result_to_tags(run_id, combined_result)

    # get_latest_done_run_id() + get_tags() are what GET /papers/{id}/results now
    # reads from — every inserted tag should come back with its DB-side defaults.
    found_run_id = main.get_latest_done_run_id(cursor, 1)
    assert found_run_id == run_id
    read_tags = main.get_tags(cursor, found_run_id)
    assert len(read_tags) == len(combined_tags)
    assert {t["tag_type"] for t in read_tags} == {"label", "entity", "design_strategy", "ecosystem_service"}
    assert all(t["review_status"] == "pending" and t["added_manually"] is False for t in read_tags)
    design_read = next(t for t in read_tags if t["tag_type"] == "design_strategy")
    assert design_read["page_number"] == 3
    assert design_read["bbox"] == {"x0": 10.0, "y0": 20.0, "x1": 100.0, "y1": 40.0}

    # update_tag_review_status() sets review_status without touching anything
    # else, and never deletes the row (accept/reject are both soft).
    some_tag_id = read_tags[0]["id"]
    assert main.update_tag_review_status(cursor, some_tag_id, "rejected") is True
    rejected = next(t for t in cursor._tags if t["id"] == some_tag_id)
    assert rejected["review_status"] == "rejected"
    assert len(cursor._tags) == len(combined_tags)  # nothing was deleted
    assert main.update_tag_review_status(cursor, 999999, "accepted") is False  # unknown id -> not found

    # update_tag_value() records a correction as edited_value, leaving the
    # original extractor "value" untouched, and marks review_status "edited".
    another_tag_id = read_tags[1]["id"]
    original_value = next(t for t in cursor._tags if t["id"] == another_tag_id)["value"]
    assert main.update_tag_value(cursor, another_tag_id, "Corrected Name") is True
    edited = next(t for t in cursor._tags if t["id"] == another_tag_id)
    assert edited["edited_value"] == "Corrected Name"
    assert edited["value"] == original_value  # original is never overwritten
    assert edited["review_status"] == "edited"
    assert main.update_tag_value(cursor, 999999, "x") is False  # unknown id -> not found

    # add_manual_tag() creates a tag with added_manually=True and
    # review_status "accepted" straight away — a human just typed it in,
    # there's nothing left to review.
    before_count = len(cursor._tags)
    manual_tag = main.add_manual_tag(cursor, run_id, "design_strategy", "rainwater cistern")
    assert manual_tag["added_manually"] is True and manual_tag["review_status"] == "accepted"
    assert len(cursor._tags) == before_count + 1
    stored_manual = next(t for t in cursor._tags if t["value"] == "rainwater cistern")
    assert stored_manual["tag_type"] == "design_strategy" and stored_manual["added_manually"] is True

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

    # Manual-upload watcher: a "{paper_id}.pdf" dropped in manual_uploads/ runs
    # through the normal pipeline and ends up in manual_uploads/extracted/,
    # while unrelated files and in-progress papers are left alone.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        manual_dir = os.path.join(tmp, "manual_uploads")
        extracted_dir = os.path.join(manual_dir, "extracted")
        papers_dir = os.path.join(tmp, "downloaded_papers")
        os.makedirs(extracted_dir)
        os.makedirs(papers_dir)
        with open(os.path.join(manual_dir, "42.pdf"), "wb") as f:
            f.write(b"%PDF-1.4 fake")
        with open(os.path.join(manual_dir, "notes.txt"), "w") as f:
            f.write("not a paper")  # should be left untouched, not a "{paper_id}.pdf"

        cursor = FakeCursor([{"id": 42, "title": "D", "authors": [], "query": "q", "open_access": True, "pdf_url": None}])
        with patch("main.MANUAL_DIR", manual_dir), \
             patch("main.MANUAL_EXTRACTED_DIR", extracted_dir), \
             patch("main.PAPERS_DIR", papers_dir), \
             patch("main.get_db_connection", lambda: FakeConnection(cursor)), \
             patch("main.notify_extraction", lambda paper_id, pdf_bytes: combined_result):
            main._process_manual_uploads()

        assert not os.path.exists(os.path.join(manual_dir, "42.pdf"))  # picked up...
        assert os.path.exists(os.path.join(extracted_dir, "42.pdf"))  # ...and filed away
        assert os.path.exists(os.path.join(manual_dir, "notes.txt"))  # untouched
        assert not os.path.exists(os.path.join(papers_dir, "42.pdf"))  # staging copy cleaned up
        assert cursor._latest_run(42)["status"] == "done"

        # A paper already mid-extraction is left for next tick, not double-processed.
        with open(os.path.join(manual_dir, "43.pdf"), "wb") as f:
            f.write(b"%PDF-1.4 fake")
        cursor = FakeCursor(
            [{"id": 43, "title": "E", "authors": [], "query": "q", "open_access": True, "pdf_url": None}],
            runs=[{"id": 1, "paper_id": 43, "status": "extracting", "error": None, "raw_result": None,
                   "raw_text": None, "started_at": datetime.now(), "finished_at": None}],
        )
        with patch("main.MANUAL_DIR", manual_dir), \
             patch("main.MANUAL_EXTRACTED_DIR", extracted_dir), \
             patch("main.PAPERS_DIR", papers_dir), \
             patch("main.get_db_connection", lambda: FakeConnection(cursor)):
            main._process_manual_uploads()
        assert os.path.exists(os.path.join(manual_dir, "43.pdf"))  # still there, not touched this tick

    print("backend papers/extraction self-check passed")


if __name__ == "__main__":
    main_()
