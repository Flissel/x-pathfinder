# :mag: X-Pathfinder

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Evolutionary X/Twitter account and email discovery with backer scoring.**

X-Pathfinder uses genetic algorithms to evolve search strategies for discovering relevant X/Twitter accounts, email addresses, and crowdfunding backers. Search strategies improve over time through persistent knowledge accumulation across sessions.

## Features

- **Genetic algorithm search evolution** -- strategies mutate and compete to find the most relevant results
- **Account fitness scoring** (0-100) across 4 dimensions: relevance, engagement, authority, activity
- **Backer score** (0-100) for evaluating crowdfunding campaign potential -- NEW
- **Email pattern evolution** with MX record and SMTP verification
- **Persistent knowledge base** -- learns and improves across sessions
- **Live dashboard** on `localhost:8420`

## Installation

```bash
git clone https://github.com/Flissel/x-pathfinder.git
cd x-pathfinder
pip install -r requirements.txt
```

## Usage

### Discover backers for a topic

```bash
python -m x_pathfinder backers ai
```

### Find email addresses

```bash
python -m x_pathfinder emails ai
```

### Discover X/Twitter accounts

```bash
python -m x_pathfinder accounts "machine learning"
```

### Launch the dashboard

```bash
python -m x_pathfinder dashboard
# Open http://localhost:8420
```

## How It Works

1. **Initialize** -- seed population of search strategies is created
2. **Search** -- each strategy queries X/Twitter and scores results
3. **Evaluate** -- accounts are scored on fitness dimensions
4. **Evolve** -- top strategies reproduce, mutate, and replace weak ones
5. **Persist** -- knowledge and winning strategies are saved for next run

## Project Structure

```
x-pathfinder/
  x_pathfinder/
    __main__.py
    evolution/
      genetic.py
      fitness.py
      backer_score.py
    discovery/
      accounts.py
      emails.py
    verification/
      mx_check.py
      smtp_check.py
    knowledge/
      store.py
    dashboard/
      app.py
  requirements.txt
```

## License

MIT -- Felix Baumann ([@Flissel](https://github.com/Flissel))
