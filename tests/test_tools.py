"""
Comprehensive tests for tools.py.

These tests verify:
- URL builders produce correct patterns for players, teams, and predictions
- Public fetch_* functions return valid JSON with expected structure
- LangChain tools are callable and return parseable JSON
- HTML snapshot parsing (overall 1vs1 index, Serie A attacking threat, LaLiga goals, 
  Bundesliga teams, Ligue 1 predictions) works without crashing
- Error paths (no table, network failures) return structured error JSON

Future expansion:
- Tighter assertions on snapshot data (specific headers, row counts)
- More comprehensive snapshot coverage across leagues/metrics
- Integration tests for main.py routing
"""


import json
import re

import pytest

import tools


# -------------------------
# URL builder tests
# -------------------------

def test_build_player_url_overall_xg_total():
    """
    Ensure overall/top-five Europe player xG URL is built correctly.
    """
    url = tools._build_player_url("overall", "xg_total")
    assert url == "https://one-versus-one.com/en/rankings/players/xg-total"


def test_build_player_url_laliga_1vs1_index():
    """
    Ensure LaLiga player 1vs1 index URL is built correctly.
    """
    url = tools._build_player_url("laliga", "1vs1_index")
    assert url == "https://one-versus-one.com/en/rankings/laliga/players/1vs1-index"


def test_build_team_url_premier_league_1vs1_index():
    """
    Ensure Premier League team 1vs1 index URL is built correctly.
    """
    url = tools._build_team_url("premier_league", "1vs1_index")
    assert url == "https://one-versus-one.com/en/rankings/premier-league/teams/1vs1-index"


def test_build_predictions_url_overall_and_league():
    """
    Ensure predictions URLs are built correctly for overall and a league.
    """
    overall_url = tools._build_predictions_url("overall")
    pl_url = tools._build_predictions_url("premier_league")

    assert overall_url == "https://one-versus-one.com/en/predictions"
    assert pl_url == "https://one-versus-one.com/en/predictions/premier-league"


# -------------------------
# Fetch function tests (smoke)
# -------------------------

@pytest.mark.parametrize(
    "league_key,metric_key",
    [
        ("overall", "1vs1_index"),
        ("premier_league", "goals"),
        ("laliga", "xg_total"),
    ],
)
def test_fetch_player_rankings_smoke(league_key, metric_key):
    """
    Smoke test: fetch_player_rankings returns valid JSON with required keys.

    This calls the real site; later you can mock HTTP responses and
    assert more detailed structure.
    """
    out = tools.fetch_player_rankings(league_key, metric_key)
    data = json.loads(out)

    assert "source" in data
    assert "url" in data
    assert "headers" in data
    assert "rows" in data
    assert isinstance(data["headers"], list)
    assert isinstance(data["rows"], list)


@pytest.mark.parametrize(
    "league_key,metric_key",
    [
        ("overall", "1vs1_index"),
        ("premier_league", "1vs1_index"),
    ],
)
def test_fetch_team_rankings_smoke(league_key, metric_key):
    """
    Smoke test: fetch_team_rankings returns valid JSON with required keys.
    """
    out = tools.fetch_team_rankings(league_key, metric_key)
    data = json.loads(out)

    assert "source" in data
    assert "url" in data
    assert "headers" in data
    assert "rows" in data
    assert isinstance(data["headers"], list)
    assert isinstance(data["rows"], list)


@pytest.mark.parametrize(
    "league_key",
    ["overall", "premier_league", "laliga"],
)
def test_fetch_league_predictions_smoke(league_key):
    """
    Smoke test: fetch_league_predictions returns JSON with basic meta fields.
    """
    out = tools.fetch_league_predictions(league_key)
    data = json.loads(out)

    assert "source" in data
    assert "url" in data
    assert "page_title" in data
    assert "raw_html" in data
    assert isinstance(data["raw_html"], str)
    # Very loose check that the HTML looks like HTML
    assert "<html" in data["raw_html"].lower()


# -------------------------
# LangChain Tool smoke tests
# -------------------------

@pytest.mark.parametrize(
    "tool_obj",
    [
        # Overall players
        tools.overall_players_1vs1_index_tool,
        tools.overall_players_goals_tool,
        tools.overall_players_xg_total_tool,
        # Premier League players
        tools.pl_players_1vs1_index_tool,
        tools.pl_players_goals_tool,
        tools.pl_players_xg_total_tool,
        # LaLiga players
        tools.laliga_players_1vs1_index_tool,
        tools.laliga_players_xg_total_tool,
        # Overall teams
        tools.overall_teams_1vs1_index_tool,
        # Premier League teams
        tools.pl_teams_1vs1_index_tool,
        # Predictions
        tools.overall_predictions_tool,
        tools.pl_predictions_tool,
    ],
)
def test_langchain_tools_return_json(tool_obj):
    """
    Smoke test: each LangChain Tool is callable and returns parseable JSON.

    For now this only verifies that:
    - The tool can be called with a dummy string.
    - The output parses as JSON.
    - The JSON has at least 'source' and 'url' keys.
    """
    out = tool_obj.run("")
    data = json.loads(out)

    assert "source" in data
    assert "url" in data

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

DATA_DIR = Path(__file__).parent / "data"


def _load_html(name: str) -> str:
    """
    Helper to load static HTML snapshots from tests/data.
    """
    return (DATA_DIR / name).read_text(encoding="utf-8")


@patch("tools.session.get")
def test_scrape_overall_players_1vs1_index_from_snapshot(mock_get):
    """
    Smoke test: _scrape_1vs1_table must handle the saved overall 1vs1 index
    HTML snapshot without crashing and return valid JSON.

    The view-source snapshot may not contain a usable <table>, so this test
    only asserts that the function returns valid JSON with a URL, and either
    an 'error' field or empty headers/rows. It does NOT require data rows.
    """
    html = _load_html(
        "view-source_https___one-versus-one.com_en_rankings_players_1vs1-index.html"
    )

    # Mock session.get to return the snapshot HTML
    mock_get.return_value = SimpleNamespace(
        status_code=200,
        text=html,
        raise_for_status=lambda: None,
    )

    url = tools._build_player_url("overall", "1vs1_index")
    out = tools._scrape_1vs1_table(
        url=url,
        source_tag="overall_players_1vs1_index_snapshot",
        max_rows=20,
    )
    data = json.loads(out)

    # Always valid JSON with a URL
    assert data.get("url") == url

    # Either an error (no table) or a basic headers/rows structure
    if "error" in data:
        assert "No <table> found" in data["error"] or "Failed to fetch" in data["error"]
    else:
        assert "headers" in data
        assert "rows" in data
        assert isinstance(data["headers"], list)
        assert isinstance(data["rows"], list)

@patch("tools.session.get")
def test_snapshot_serie_a_players_attacking_threat(mock_get):
    """
    Smoke test: Serie A players attacking-threat snapshot.
    """
    html = _load_html(
        "view-source_https___one-versus-one.com_en_rankings_serie-a_players_attacking-threat.html"
    )

    mock_get.return_value = SimpleNamespace(
        status_code=200,
        text=html,
        raise_for_status=lambda: None,
    )

    url = tools._build_player_url("serie_a", "attacking_threat")
    out = tools._scrape_1vs1_table(
        url=url,
        source_tag="serie_a_players_attacking_threat_snapshot",
        max_rows=20,
    )
    data = json.loads(out)

    assert data.get("url") == url
    if "error" in data:
        assert "No <table>" in data["error"] or "Failed to fetch" in data["error"]
    else:
        assert "headers" in data
        assert "rows" in data
        assert isinstance(data["headers"], list)
        assert isinstance(data["rows"], list)


@patch("tools.session.get")
def test_snapshot_laliga_players_goals(mock_get):
    """
    Smoke test: LaLiga players goals snapshot.
    """
    html = _load_html(
        "view-source_https___one-versus-one.com_en_rankings_laliga_players_goals.html"
    )

    mock_get.return_value = SimpleNamespace(
        status_code=200,
        text=html,
        raise_for_status=lambda: None,
    )

    url = tools._build_player_url("laliga", "goals")
    out = tools._scrape_1vs1_table(
        url=url,
        source_tag="laliga_players_goals_snapshot",
        max_rows=20,
    )
    data = json.loads(out)

    assert data.get("url") == url
    if "error" in data:
        assert "No <table>" in data["error"] or "Failed to fetch" in data["error"]
    else:
        assert "headers" in data
        assert "rows" in data
        assert isinstance(data["headers"], list)
        assert isinstance(data["rows"], list)


@patch("tools.session.get")
def test_snapshot_bundesliga_teams_offensive_interventions(mock_get):
    """
    Smoke test: Bundesliga teams offensive-interventions snapshot.
    """
    html = _load_html(
        "view-source_https___one-versus-one.com_en_rankings_bundesliga_teams_offensive-interventions.html"
    )

    mock_get.return_value = SimpleNamespace(
        status_code=200,
        text=html,
        raise_for_status=lambda: None,
    )

    url = tools._build_team_url("bundesliga", "offensive_interventions")
    out = tools._scrape_1vs1_table(
        url=url,
        source_tag="bundesliga_teams_offensive_interventions_snapshot",
        max_rows=20,
    )
    data = json.loads(out)

    assert data.get("url") == url
    if "error" in data:
        assert "No <table>" in data["error"] or "Failed to fetch" in data["error"]
    else:
        assert "headers" in data
        assert "rows" in data
        assert isinstance(data["headers"], list)
        assert isinstance(data["rows"], list)


@patch("tools.session.get")
def test_snapshot_ligue_1_predictions(mock_get):
    """
    Smoke test: Ligue 1 predictions snapshot.
    """
    html = _load_html(
        "view-source_https___one-versus-one.com_en_predictions_ligue-1.html"
    )

    mock_get.return_value = SimpleNamespace(
        status_code=200,
        text=html,
        raise_for_status=lambda: None,
    )

    url = tools._build_predictions_url("ligue_1")
    out = tools._scrape_1vs1_predictions(
        url=url,
        source_tag="ligue_1_predictions_snapshot",
    )
    data = json.loads(out)

    assert data.get("url") == url
    assert "page_title" in data
    assert "raw_html" in data
    assert isinstance(data["raw_html"], str)

@patch("tools.session.get")
def test_scrape_1vs1_table_no_table(mock_get):
    """
    When no <table> exists, _scrape_1vs1_table should return an error JSON.
    """
    mock_get.return_value = SimpleNamespace(
        status_code=200,
        text="<html><body><p>No table here</p></body></html>",
        raise_for_status=lambda: None,
    )

    out = tools._scrape_1vs1_table("https://example.com", "test")
    data = json.loads(out)

    assert "error" in data
    assert "No <table> found" in data["error"]


import requests

@patch("tools.session.get")
def test_scrape_1vs1_table_network_error(mock_get):
    """
    If the HTTP request fails (e.g. timeout/connection), _scrape_1vs1_table
    should return an error JSON instead of raising.
    """
    mock_get.side_effect = requests.RequestException("boom")

    out = tools._scrape_1vs1_table("https://example.com", "test")
    data = json.loads(out)

    assert "error" in data
    assert "Failed to fetch 1vs1 page" in data["error"]


@patch("tools.session.get")
def test_scrape_1vs1_predictions_network_error(mock_get):
    """
    If the HTTP request fails, _scrape_1vs1_predictions should return error JSON.
    """
    mock_get.side_effect = requests.RequestException("boom")

    out = tools._scrape_1vs1_predictions("https://example.com/predictions", "test_predictions")
    data = json.loads(out)

    assert "error" in data
    assert "Failed to fetch 1vs1 predictions page" in data["error"]
