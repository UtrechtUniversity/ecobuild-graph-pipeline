# Paper Crawler

This repo contains a crawler that populates the [Document Database](https://git.science.uu.nl/auto-kg-lit/document-database) with papers crawled from the [Semantic Scholar API](https://api.semanticscholar.org/)

## Getting started 

Make sure you have an instance of the [Document Database](https://git.science.uu.nl/auto-kg-lit/document-database) running. Create a `.env` file based on `.env.template` (you need to request a semantic scholar API key).

Then build the image using: 

```bash
docker build -t paper-crawler . 
```

Then run the container using 
```bash
docker run -p 8000:8000 paper-crawler  
```

## Progress
To do:  
- find out what information to filter and query on 
- handle integration with document database (ensure database is running before crawling)
- allow to write to database
- allow for periodic reruns to find research published after last run

Done: 
- write main crawling logic
- initialized repo
- set up containerization
- handled API key usage

