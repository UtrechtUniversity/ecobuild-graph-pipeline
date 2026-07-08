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

CREATE TABLE search_queries (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL UNIQUE,                        /*the search query the crawler runs*/
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- seed with the queries that used to be hardcoded in crawler/config.py
INSERT INTO search_queries (query) VALUES
    ('Green roof effect on evaporation'),
    ('rainwater harvesting effectiveness morocco');

