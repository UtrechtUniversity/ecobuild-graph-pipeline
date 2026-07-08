CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    ss_id TEXT NOT NULL UNIQUE,                              /*Semantic scholar internal ID*/
    title TEXT NOT NULL,                              /*Title of the paper*/
    authors TEXT[],                                   /*Names of the authors*/
    url TEXT UNIQUE,                                  /*url to the article within semantic scholar*/
    doi TEXT UNIQUE,                                  /*doi to the article*/
    abstract TEXT,                                    /*abstract of the article*/
    pdf_url TEXT,                                     /*url of the pdf*/
    open_access BOOL,                                 /*whether the pdf is openly accessible*/
    query TEXT,                                       /*the query that found this article*/
    relevance_checked BOOL,                           /*whether this has been checked for relevance*/
    relevant BOOL,                                    /*whether it is relevant*/
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP    /*timestamp*/
);

CREATE TABLE extraction_runs (
    id SERIAL PRIMARY KEY,
    paper_id INT NOT NULL REFERENCES papers(id),
    status TEXT NOT NULL,                              /*pending|downloading|downloaded|extracting|done|failed*/
    error TEXT,                                        /*error message, set when status = failed*/
    raw_result JSONB,                                  /*knowledge-extraction's result JSON, set when status = done*/
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP                              /*set once status reaches done or failed*/
);

CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    extraction_run_id INT NOT NULL REFERENCES extraction_runs(id),
    tag_type TEXT NOT NULL,                            /*'label' | 'entity' | 'design_strategy' | 'ecosystem_service' | ... (open set, new extractors just use a new value)*/
    group_id INT,                                      /*clusters rows belonging to one extracted item, e.g. a single building's separately-verified fields*/
    field TEXT,                                        /*for per-field tag types (e.g. entity): which field this row is ('name', 'city', ...)*/
    value TEXT,                                        /*the extracted value, e.g. 'governance', 'Amsterdam', 'green roof'*/
    category TEXT,                                     /*e.g. ecosystem service category; null where not applicable*/
    anchor_text TEXT,                                  /*raw LLM-picked phrase*/
    context TEXT,                                       /*resolved source passage*/
    match_score FLOAT,                                 /*anchor-resolution / verification quality, 0.0-1.0*/
    rationale TEXT,                                     /*LLM's justification, where the extractor produces one*/
    verified BOOLEAN,                                  /*the extractor's own verdict (YES/verified vs UNVERIFIED/unverified)*/
    extra_data JSONB,                                   /*structured extras that don't fit flat columns, e.g. vocab_top_matches/implementation_details*/
    review_status TEXT NOT NULL DEFAULT 'pending',      /*pending|accepted|rejected|edited*/
    edited_value TEXT,                                  /*reviewer's correction, set when review_status = 'edited'*/
    added_manually BOOLEAN NOT NULL DEFAULT FALSE,      /*true for tags a reviewer added, not the extractor*/
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE search_queries (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL UNIQUE,                        /*the search query the crawler runs*/
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- seed with the queries that used to be hardcoded in crawler/config.py
INSERT INTO search_queries (query) VALUES
    ('Green roof effect on evaporation'),
    ('rainwater harvesting effectiveness morocco');

