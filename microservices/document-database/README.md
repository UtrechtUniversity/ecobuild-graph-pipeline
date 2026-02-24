# Document Database

This repo contains the files to spin up a docker container for a PostgreSQL database that will store information about crawled papers. 

For every paper we store:
- Title
- Authors
- url within semantic scholar
- abstract
- url of the pdf
- whether the paper is publicly accessible
- the query used to obtain this paper

# Getting started

Spin up the database by using the instructions in the [orchestration repo](https://git.science.uu.nl/auto-kg-lit/orchestration)

# Populating the database

Populating this database with actual papers happens by running the [Paper Crawler microservice](https://git.science.uu.nl/auto-kg-lit/paper-crawler]
