# Scopus Crawler

Populates the document database with papers crawled from the [Scopus Search API](https://dev.elsevier.com/documentation/SCOPUSSearchAPI.wadl), mirroring `paper-crawler`'s shape as a second, independent crawler source (`source = 'scopus'` in `papers`/`search_queries`).

## Getting started

Make sure an instance of the document database is running. Create a `.env` file based on `.env.template` — you need an Elsevier API key (`SCOPUS_API_KEY`), and possibly an institutional token (`SCOPUS_INSTTOKEN`) for abstract/author access outside your institution's IP range.

Then build the image using:

```bash
docker build -t scopus-crawler .
```

Then run the container using:

```bash
docker run -p 8000:8000 scopus-crawler
```

## Notes

- Scopus's Search API only returns a title snippet and first author; this crawler makes a second Abstract Retrieval API call per paper for the full abstract and author list.
- Scopus has no full-text/PDF hosting, so `pdf_url` is always `None` — the existing manual-PDF-upload fallback in `api-backend` covers these papers.
- Scopus's quota is weekly (not per-second like Semantic Scholar), so this crawler uses `rate_limiter.EvenSpacing` instead of `FixedDelay`.
