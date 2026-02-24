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

