# AI-Powered Sports Analytics Agent

## Overview

This project implements a dual-agent system using LangChain and OpenAI's GPT-4o to provide intelligent responses for **DraftKings research** and **soccer analytics** across Europe's top 5 leagues (Premier League, Bundesliga, LaLiga, Serie A, Ligue 1).

Key features:
- **Dynamic routing** between specialized agents based on query content
- **Live web scraping** from [one-versus-one.com](https://one-versus-one.com) for 1vs1 Index, xG, goals, assists, attacking threat, and more
- **Comprehensive test suite** with HTML snapshots and error path coverage
- **Production-ready** with proper error handling and structured JSON outputs

## Architecture

main.py (Router + Agents)
├── DraftKings Agent (search + wiki tools)
├── Soccer Agent (60+ 1vs1 scrapers + news tools)
├── Players: All 5 leagues × 8 metrics (1vs1 Index, defensive/offensive, goals, assists, xG, etc.)
├── Teams: All 5 leagues × 8 metrics
└── Predictions: All 5 leagues
└── test_tools.py (32 tests)

text

## Quick Start

### 1. Setup Environment
Use included venv (recommended)
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt

text

### 2. Add OpenAI Key
Create `.env`:
OPENAI_API_KEY=sk-...

text

### 3. Run
.\venv\Scripts\python.exe -u .\main.py

text
undefined
Ask about DraftKings or soccer: top xG players in LaLiga

Routes to soccer agent → calls laliga_players_xg_total_tool → returns structured JSON

text

## Core Functionality

### Soccer Agent Capabilities
| Query Type | Example | Tool Called |
|------------|---------|-------------|
| League rankings | "top xG LaLiga" | `laliga_players_xg_total_tool` |
| Cross-league | "best 1vs1 index overall" | `overall_players_1vs1_index_tool` |
| Team stats | "PL teams attacking threat" | `pl_teams_attacking_threat_tool` |
| Predictions | "Ligue 1 predictions" | `ligue_1_predictions_tool` |

### Technical Highlights
- **Generic URL builders** handle all league/metric combinations dynamically
- **Robust scraping** with error handling for missing tables/network failures
- **Unit tests** cover live data, HTML snapshots (5 pages), and all error paths
- **Rate-limit safe** with shared HTTP session and structured fallbacks

## Testing

Comprehensive test suite (25+ tests):
.\venv\Scripts\python.exe -m pip install pytest
.\venv\Scripts\python.exe -m pytest -q # All tests pass

text

Coverage includes:
- URL builder correctness (players/teams/predictions)
- Live scraper smoke tests (no network flakiness)
- 5× HTML snapshot parsing tests
- Error paths (no table, network failures)

## Example Queries

**Soccer Analytics:**
"Top 5 xG players in LaLiga" → laliga_players_xg_total_tool
"Best defensive teams Bundesliga" → bundesliga_teams_1vs1_index_defensive_tool
"Premier League goals leaders" → pl_players_goals_tool

text

**DraftKings:**
"DraftKings daily fantasy rules" → DraftKings agent + web search
"DraftKings leadership team" → Wikipedia + search

text

## Files Structure

├── main.py # Router + dual agents
├── tools.py # 60+ 1vs1 scrapers + utilities
├── tests/
│ ├── test_tools.py # 25+ comprehensive tests
│ └── data/ # 5× HTML snapshots for testing
├── requirements.txt # Dependencies
├── .env.example # API key template
└── README.md

text

## Deployment Notes

- **Scalable**: Add new leagues/metrics by extending `PLAYER_METRICS`
- **Maintainable**: Generic builders + comprehensive tests
- **Rate-limit safe**: Shared session + error JSON fallbacks
- **Extensible**: Easy to add new agents/tools to `sports_tools` list

## Future Work

1. **Integration tests** for `main.py` routing accuracy
2. **Caching layer** for 1vs1 data (reduce API calls)
3. **Dashboard** to visualize scraped rankings
4. **More leagues** (MLS, Brasileirão via same URL pattern)
5. **More sports?**

---
*Implemented by Minh for Directed Studies - December 2025*