from datetime import datetime
import json
from typing import Optional, Callable

import requests
from bs4 import BeautifulSoup

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool

# -------------------------
# Shared HTTP session
# -------------------------

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
})

# -------------------------
# Utility tools
# -------------------------

def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """
    Append a timestamped block of text to a local .txt file.

    Args:
        data: Arbitrary text to be written.
        filename: Name of the output file (default 'research_output.txt').

    Returns:
        A short status message confirming the write.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

    return f"Data successfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

# Generic web search
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information.",
)

# Wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Sports news search
sports_search = DuckDuckGoSearchRun()
sports_news_tool = Tool(
    name="sports_news_search",
    func=sports_search.run,
    description=(
        "Use this to get recent news, scores, and information about sports teams, "
        "players, leagues, and events."
    ),
)

# -------------------------
# 1vs1 URL config
# -------------------------

ONE_VS_ONE_BASE_RANKINGS = "https://one-versus-one.com/en/rankings"
ONE_VS_ONE_BASE_PREDICTIONS = "https://one-versus-one.com/en/predictions"

# League slugs for rankings and predictions
LEAGUE_SLUGS = {
    "overall": "",             
    "premier_league": "premier-league",
    "bundesliga": "bundesliga",
    "laliga": "laliga",
    "serie_a": "serie-a",
    "ligue_1": "ligue-1",
}

# Player metrics
PLAYER_METRICS = {
    "1vs1_index": "1vs1-index",
    "1vs1_index_defensive": "1vs1-index-defensive",
    "1vs1_index_offensive": "1vs1-index-offensive",
    "offensive_interventions": "offensive-interventions",
    "attacking_threat": "attacking-threat",
    "goals": "goals",
    "assists": "assists",
    "xg_total": "xg-total",
}

# Teams use the same metric slugs as players
TEAM_METRICS = PLAYER_METRICS

# -------------------------
# Low-level scrapers
# -------------------------

def _scrape_1vs1_table(url: str, source_tag: str, max_rows: Optional[int] = 50) -> str:
    """
    Internal helper to scrape a generic 1vs1 rankings table from a given URL.

    The function assumes that the page contains a main <table> element
    representing the ranking. It extracts all column headers from <th> tags
    and then reads up to max_rows data rows from <td> cells.

    Args:
        url: Full 1vs1 URL for a rankings table.
        source_tag: Short string identifying the type of ranking (e.g. 'pl_players_1vs1_index').
        max_rows: Optional maximum number of rows to return (default 50).

    Returns:
        A JSON string with keys:
            - source: short source tag
            - url: the URL that was scraped
            - headers: list of column names
            - row_count: number of rows returned
            - rows: list of row dicts mapping header -> cell text
    """
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return json.dumps({
            "error": f"Failed to fetch 1vs1 page: {e}",
            "url": url,
        })

    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table")
    if table is None:
        return json.dumps({
            "error": "No <table> found on the 1vs1 rankings page.",
            "url": url,
        })

    # Extract headers
    header_cells = table.find_all("th")
    headers = [h.get_text(strip=True) for h in header_cells]

    # Prefer tbody rows; otherwise use all <tr> except header
    if table.find("tbody"):
        rows = table.find("tbody").find_all("tr")
    else:
        all_rows = table.find_all("tr")
        rows = all_rows[1:] if len(all_rows) > 1 else []

    data = []
    for row in rows:
        cells = [c.get_text(strip=True) for c in row.find_all("td")]
        if not cells:
            continue

        # Truncate or pad cells to match headers length
        if len(cells) > len(headers):
            cells = cells[:len(headers)]
        elif len(cells) < len(headers):
            cells = cells + [""] * (len(headers) - len(cells))

        record = dict(zip(headers, cells))
        data.append(record)

        if max_rows is not None and len(data) >= max_rows:
            break

    return json.dumps({
        "source": f"one-versus-one_{source_tag}",
        "url": url,
        "headers": headers,
        "row_count": len(data),
        "rows": data,
    })


def _scrape_1vs1_predictions(url: str, source_tag: str) -> str:
    """
    Internal helper to scrape a 1vs1 predictions page.

    The predictions pages are less tabular and more content-oriented, so this
    function currently returns:
        - page_title: <title> tag text
        - h1: first <h1> text (if any)
        - raw_html: full HTML (caller can parse further)
        - url: scraped URL

    Args:
        url: Full 1vs1 predictions URL.
        source_tag: Short string identifying the predictions source.

    Returns:
        JSON string with meta information and raw HTML.
    """
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return json.dumps({
            "error": f"Failed to fetch 1vs1 predictions page: {e}",
            "url": url,
        })

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""

    return json.dumps({
        "source": f"one-versus-one_{source_tag}",
        "url": url,
        "page_title": title,
        "h1": h1,
        "raw_html": resp.text,
    })

# -------------------------
# URL builders
# -------------------------

def _build_player_url(league_key: str, metric_key: str) -> str:
    """
    Build a 1vs1 player rankings URL for a given league and metric.

    Args:
        league_key: One of LEAGUE_SLUGS keys ('overall', 'premier_league', etc.).
        metric_key: One of PLAYER_METRICS keys ('1vs1_index', 'goals', etc.).

    Returns:
        Full URL to the corresponding 1vs1 player rankings page.
    """
    league_slug = LEAGUE_SLUGS[league_key]
    metric_slug = PLAYER_METRICS[metric_key]

    # Overall/top-five-Europe has no league segment in rankings URL
    if league_key == "overall":
        return f"{ONE_VS_ONE_BASE_RANKINGS}/players/{metric_slug}"
    else:
        return f"{ONE_VS_ONE_BASE_RANKINGS}/{league_slug}/players/{metric_slug}"


def _build_team_url(league_key: str, metric_key: str) -> str:
    """
    Build a 1vs1 team rankings URL for a given league and metric.

    Args:
        league_key: One of LEAGUE_SLUGS keys ('overall', 'premier_league', etc.).
        metric_key: One of TEAM_METRICS keys ('1vs1_index', etc.).

    Returns:
        Full URL to the corresponding 1vs1 team rankings page.
    """
    league_slug = LEAGUE_SLUGS[league_key]
    metric_slug = TEAM_METRICS[metric_key]

    # Overall/top-five-Europe has no league segment
    if league_key == "overall":
        return f"{ONE_VS_ONE_BASE_RANKINGS}/teams/{metric_slug}"
    else:
        return f"{ONE_VS_ONE_BASE_RANKINGS}/{league_slug}/teams/{metric_slug}"


def _build_predictions_url(league_key: str) -> str:
    """
    Build a 1vs1 predictions URL for a given league.

    For 'overall', this points to the generic predictions overview.
    For specific leagues, it uses /predictions/<league-slug>.

    Args:
        league_key: One of LEAGUE_SLUGS keys.

    Returns:
        Full URL to the corresponding 1vs1 predictions page.
    """
    league_slug = LEAGUE_SLUGS[league_key]
    if league_key == "overall":
        return f"{ONE_VS_ONE_BASE_PREDICTIONS}"
    else:
        return f"{ONE_VS_ONE_BASE_PREDICTIONS}/{league_slug}"

# -------------------------
# Generic public fetchers
# -------------------------

def fetch_player_rankings(league_key: str, metric_key: str, max_rows: int = 50) -> str:
    """
    Generic fetcher for 1vs1 player rankings across leagues and metrics.

    Args:
        league_key: League identifier (e.g. 'overall', 'premier_league', 'bundesliga').
        metric_key: Metric identifier (e.g. '1vs1_index', 'goals', 'assists', 'xg_total').
        max_rows: Maximum number of table rows to extract.

    Returns:
        JSON string with headers and rows for the requested league/metric.
    """
    url = _build_player_url(league_key, metric_key)
    source_tag = f"{league_key}_players_{metric_key}"
    return _scrape_1vs1_table(url=url, source_tag=source_tag, max_rows=max_rows)


def fetch_team_rankings(league_key: str, metric_key: str, max_rows: int = 50) -> str:
    """
    Generic fetcher for 1vs1 team rankings across leagues and metrics.

    Args:
        league_key: League identifier (e.g. 'overall', 'premier_league', 'laliga').
        metric_key: Team metric identifier (e.g. '1vs1_index').

    Returns:
        JSON string with headers and rows for the requested league/metric.
    """
    url = _build_team_url(league_key, metric_key)
    source_tag = f"{league_key}_teams_{metric_key}"
    return _scrape_1vs1_table(url=url, source_tag=source_tag, max_rows=max_rows)


def fetch_league_predictions(league_key: str) -> str:
    """
    Fetch 1vs1 AI predictions page for a given league.

    Args:
        league_key: League identifier (e.g. 'overall', 'premier_league', 'bundesliga').

    Returns:
        JSON string containing basic page meta and raw HTML of the predictions page.
    """
    url = _build_predictions_url(league_key)
    source_tag = f"{league_key}_predictions"
    return _scrape_1vs1_predictions(url=url, source_tag=source_tag)

# -------------------------
# LangChain tools (examples)
# -------------------------
# You can add or remove tools here depending on what you want the agent to call.

def _make_player_tool(
    league_key: str,
    metric_key: str,
    tool_name: str,
    description: str,
) -> Tool:
    """
    Helper to create a LangChain Tool for a specific league + player metric.

    Args:
        league_key: League identifier.
        metric_key: Player metric identifier.
        tool_name: Name visible to the agent.
        description: Human-readable description of what the tool returns.

    Returns:
        A configured Tool instance.
    """
    def _func(_: str = "") -> str:
        return fetch_player_rankings(league_key, metric_key)

    return Tool(
        name=tool_name,
        func=_func,
        description=description,
    )


def _make_team_tool(
    league_key: str,
    metric_key: str,
    tool_name: str,
    description: str,
) -> Tool:
    """
    Helper to create a LangChain Tool for a specific league + team metric.

    Args:
        league_key: League identifier.
        metric_key: Team metric identifier.
        tool_name: Name visible to the agent.
        description: Human-readable description.

    Returns:
        A configured Tool instance.
    """
    def _func(_: str = "") -> str:
        return fetch_team_rankings(league_key, metric_key)

    return Tool(
        name=tool_name,
        func=_func,
        description=description,
    )


def _make_predictions_tool(
    league_key: str,
    tool_name: str,
    description: str,
) -> Tool:
    """
    Helper to create a LangChain Tool for a specific league predictions page.

    Args:
        league_key: League identifier.
        tool_name: Name visible to the agent.
        description: Human-readable description.

    Returns:
        A configured Tool instance.
    """
    def _func(_: str = "") -> str:
        return fetch_league_predictions(league_key)

    return Tool(
        name=tool_name,
        func=_func,
        description=description,
    )

# -------------------------
# Concrete tools you can import in main.py
# -------------------------

# Overall (Top Five Europe) players – full metric set

overall_players_1vs1_index_tool = _make_player_tool(
    league_key="overall",
    metric_key="1vs1_index",
    tool_name="overall_players_1vs1_index",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

overall_players_1vs1_index_defensive_tool = _make_player_tool(
    league_key="overall",
    metric_key="1vs1_index_defensive",
    tool_name="overall_players_1vs1_index_defensive",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

overall_players_1vs1_index_offensive_tool = _make_player_tool(
    league_key="overall",
    metric_key="1vs1_index_offensive",
    tool_name="overall_players_1vs1_index_offensive",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

overall_players_offensive_interventions_tool = _make_player_tool(
    league_key="overall",
    metric_key="offensive_interventions",
    tool_name="overall_players_offensive_interventions",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of players."
    ),
)

overall_players_attacking_threat_tool = _make_player_tool(
    league_key="overall",
    metric_key="attacking_threat",
    tool_name="overall_players_attacking_threat",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of players."
    ),
)

overall_players_goals_tool = _make_player_tool(
    league_key="overall",
    metric_key="goals",
    tool_name="overall_players_goals",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of players."
    ),
)

overall_players_assists_tool = _make_player_tool(
    league_key="overall",
    metric_key="assists",
    tool_name="overall_players_assists",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of players."
    ),
)

overall_players_xg_total_tool = _make_player_tool(
    league_key="overall",
    metric_key="xg_total",
    tool_name="overall_players_xg_total",
    description=(
        "Fetch Top Five Europe player rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of players."
    ),
)


# Premier League players – full metric set

pl_players_1vs1_index_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="1vs1_index",
    tool_name="pl_players_1vs1_index",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

pl_players_1vs1_index_defensive_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="1vs1_index_defensive",
    tool_name="pl_players_1vs1_index_defensive",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

pl_players_1vs1_index_offensive_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="1vs1_index_offensive",
    tool_name="pl_players_1vs1_index_offensive",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

pl_players_offensive_interventions_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="offensive_interventions",
    tool_name="pl_players_offensive_interventions",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of players."
    ),
)

pl_players_attacking_threat_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="attacking_threat",
    tool_name="pl_players_attacking_threat",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of players."
    ),
)

pl_players_goals_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="goals",
    tool_name="pl_players_goals",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of players."
    ),
)

pl_players_assists_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="assists",
    tool_name="pl_players_assists",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of players."
    ),
)

pl_players_xg_total_tool = _make_player_tool(
    league_key="premier_league",
    metric_key="xg_total",
    tool_name="pl_players_xg_total",
    description=(
        "Fetch Premier League player rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of players."
    ),
)

# Bundesliga players – full metric set

bundesliga_players_1vs1_index_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="1vs1_index",
    tool_name="bundesliga_players_1vs1_index",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

bundesliga_players_1vs1_index_defensive_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="1vs1_index_defensive",
    tool_name="bundesliga_players_1vs1_index_defensive",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

bundesliga_players_1vs1_index_offensive_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="1vs1_index_offensive",
    tool_name="bundesliga_players_1vs1_index_offensive",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

bundesliga_players_offensive_interventions_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="offensive_interventions",
    tool_name="bundesliga_players_offensive_interventions",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of players."
    ),
)

bundesliga_players_attacking_threat_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="attacking_threat",
    tool_name="bundesliga_players_attacking_threat",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of players."
    ),
)

bundesliga_players_goals_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="goals",
    tool_name="bundesliga_players_goals",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of players."
    ),
)

bundesliga_players_assists_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="assists",
    tool_name="bundesliga_players_assists",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of players."
    ),
)

bundesliga_players_xg_total_tool = _make_player_tool(
    league_key="bundesliga",
    metric_key="xg_total",
    tool_name="bundesliga_players_xg_total",
    description=(
        "Fetch Bundesliga player rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of players."
    ),
)

# LaLiga players – full metric set

laliga_players_1vs1_index_tool = _make_player_tool(
    league_key="laliga",
    metric_key="1vs1_index",
    tool_name="laliga_players_1vs1_index",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

laliga_players_1vs1_index_defensive_tool = _make_player_tool(
    league_key="laliga",
    metric_key="1vs1_index_defensive",
    tool_name="laliga_players_1vs1_index_defensive",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

laliga_players_1vs1_index_offensive_tool = _make_player_tool(
    league_key="laliga",
    metric_key="1vs1_index_offensive",
    tool_name="laliga_players_1vs1_index_offensive",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

laliga_players_offensive_interventions_tool = _make_player_tool(
    league_key="laliga",
    metric_key="offensive_interventions",
    tool_name="laliga_players_offensive_interventions",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of players."
    ),
)

laliga_players_attacking_threat_tool = _make_player_tool(
    league_key="laliga",
    metric_key="attacking_threat",
    tool_name="laliga_players_attacking_threat",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of players."
    ),
)

laliga_players_goals_tool = _make_player_tool(
    league_key="laliga",
    metric_key="goals",
    tool_name="laliga_players_goals",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of players."
    ),
)

laliga_players_assists_tool = _make_player_tool(
    league_key="laliga",
    metric_key="assists",
    tool_name="laliga_players_assists",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of players."
    ),
)

laliga_players_xg_total_tool = _make_player_tool(
    league_key="laliga",
    metric_key="xg_total",
    tool_name="laliga_players_xg_total",
    description=(
        "Fetch LaLiga player rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of players."
    ),
)

# Serie A players – full metric set

serie_a_players_1vs1_index_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="1vs1_index",
    tool_name="serie_a_players_1vs1_index",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

serie_a_players_1vs1_index_defensive_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="1vs1_index_defensive",
    tool_name="serie_a_players_1vs1_index_defensive",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

serie_a_players_1vs1_index_offensive_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="1vs1_index_offensive",
    tool_name="serie_a_players_1vs1_index_offensive",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

serie_a_players_offensive_interventions_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="offensive_interventions",
    tool_name="serie_a_players_offensive_interventions",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of players."
    ),
)

serie_a_players_attacking_threat_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="attacking_threat",
    tool_name="serie_a_players_attacking_threat",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of players."
    ),
)

serie_a_players_goals_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="goals",
    tool_name="serie_a_players_goals",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of players."
    ),
)

serie_a_players_assists_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="assists",
    tool_name="serie_a_players_assists",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of players."
    ),
)

serie_a_players_xg_total_tool = _make_player_tool(
    league_key="serie_a",
    metric_key="xg_total",
    tool_name="serie_a_players_xg_total",
    description=(
        "Fetch Serie A player rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of players."
    ),
)

# Ligue 1 players – full metric set

ligue_1_players_1vs1_index_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="1vs1_index",
    tool_name="ligue_1_players_1vs1_index",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

ligue_1_players_1vs1_index_defensive_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="1vs1_index_defensive",
    tool_name="ligue_1_players_1vs1_index_defensive",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

ligue_1_players_1vs1_index_offensive_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="1vs1_index_offensive",
    tool_name="ligue_1_players_1vs1_index_offensive",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of players."
    ),
)

ligue_1_players_offensive_interventions_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="offensive_interventions",
    tool_name="ligue_1_players_offensive_interventions",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of players."
    ),
)

ligue_1_players_attacking_threat_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="attacking_threat",
    tool_name="ligue_1_players_attacking_threat",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of players."
    ),
)

ligue_1_players_goals_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="goals",
    tool_name="ligue_1_players_goals",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of players."
    ),
)

ligue_1_players_assists_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="assists",
    tool_name="ligue_1_players_assists",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of players."
    ),
)

ligue_1_players_xg_total_tool = _make_player_tool(
    league_key="ligue_1",
    metric_key="xg_total",
    tool_name="ligue_1_players_xg_total",
    description=(
        "Fetch Ligue 1 player rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of players."
    ),
)


# Overall (Top Five Europe) teams – full metric set

overall_teams_1vs1_index_tool = _make_team_tool(
    league_key="overall",
    metric_key="1vs1_index",
    tool_name="overall_teams_1vs1_index",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

overall_teams_1vs1_index_defensive_tool = _make_team_tool(
    league_key="overall",
    metric_key="1vs1_index_defensive",
    tool_name="overall_teams_1vs1_index_defensive",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

overall_teams_1vs1_index_offensive_tool = _make_team_tool(
    league_key="overall",
    metric_key="1vs1_index_offensive",
    tool_name="overall_teams_1vs1_index_offensive",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

overall_teams_offensive_interventions_tool = _make_team_tool(
    league_key="overall",
    metric_key="offensive_interventions",
    tool_name="overall_teams_offensive_interventions",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of teams."
    ),
)

overall_teams_attacking_threat_tool = _make_team_tool(
    league_key="overall",
    metric_key="attacking_threat",
    tool_name="overall_teams_attacking_threat",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of teams."
    ),
)

overall_teams_goals_tool = _make_team_tool(
    league_key="overall",
    metric_key="goals",
    tool_name="overall_teams_goals",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of teams."
    ),
)

overall_teams_assists_tool = _make_team_tool(
    league_key="overall",
    metric_key="assists",
    tool_name="overall_teams_assists",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of teams."
    ),
)

overall_teams_xg_total_tool = _make_team_tool(
    league_key="overall",
    metric_key="xg_total",
    tool_name="overall_teams_xg_total",
    description=(
        "Fetch Top Five Europe team rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of teams."
    ),
)

# Premier League teams – full metric set

pl_teams_1vs1_index_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="1vs1_index",
    tool_name="pl_teams_1vs1_index",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

pl_teams_1vs1_index_defensive_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="1vs1_index_defensive",
    tool_name="pl_teams_1vs1_index_defensive",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

pl_teams_1vs1_index_offensive_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="1vs1_index_offensive",
    tool_name="pl_teams_1vs1_index_offensive",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

pl_teams_offensive_interventions_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="offensive_interventions",
    tool_name="pl_teams_offensive_interventions",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of teams."
    ),
)

pl_teams_attacking_threat_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="attacking_threat",
    tool_name="pl_teams_attacking_threat",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of teams."
    ),
)

pl_teams_goals_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="goals",
    tool_name="pl_teams_goals",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of teams."
    ),
)

pl_teams_assists_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="assists",
    tool_name="pl_teams_assists",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of teams."
    ),
)

pl_teams_xg_total_tool = _make_team_tool(
    league_key="premier_league",
    metric_key="xg_total",
    tool_name="pl_teams_xg_total",
    description=(
        "Fetch Premier League team rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of teams."
    ),
)

# Bundesliga teams – full metric set

bundesliga_teams_1vs1_index_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="1vs1_index",
    tool_name="bundesliga_teams_1vs1_index",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

bundesliga_teams_1vs1_index_defensive_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="1vs1_index_defensive",
    tool_name="bundesliga_teams_1vs1_index_defensive",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

bundesliga_teams_1vs1_index_offensive_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="1vs1_index_offensive",
    tool_name="bundesliga_teams_1vs1_index_offensive",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

bundesliga_teams_offensive_interventions_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="offensive_interventions",
    tool_name="bundesliga_teams_offensive_interventions",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of teams."
    ),
)

bundesliga_teams_attacking_threat_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="attacking_threat",
    tool_name="bundesliga_teams_attacking_threat",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of teams."
    ),
)

bundesliga_teams_goals_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="goals",
    tool_name="bundesliga_teams_goals",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of teams."
    ),
)

bundesliga_teams_assists_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="assists",
    tool_name="bundesliga_teams_assists",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of teams."
    ),
)

bundesliga_teams_xg_total_tool = _make_team_tool(
    league_key="bundesliga",
    metric_key="xg_total",
    tool_name="bundesliga_teams_xg_total",
    description=(
        "Fetch Bundesliga team rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of teams."
    ),
)

# LaLiga teams – full metric set

laliga_teams_1vs1_index_tool = _make_team_tool(
    league_key="laliga",
    metric_key="1vs1_index",
    tool_name="laliga_teams_1vs1_index",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

laliga_teams_1vs1_index_defensive_tool = _make_team_tool(
    league_key="laliga",
    metric_key="1vs1_index_defensive",
    tool_name="laliga_teams_1vs1_index_defensive",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

laliga_teams_1vs1_index_offensive_tool = _make_team_tool(
    league_key="laliga",
    metric_key="1vs1_index_offensive",
    tool_name="laliga_teams_1vs1_index_offensive",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

laliga_teams_offensive_interventions_tool = _make_team_tool(
    league_key="laliga",
    metric_key="offensive_interventions",
    tool_name="laliga_teams_offensive_interventions",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of teams."
    ),
)

laliga_teams_attacking_threat_tool = _make_team_tool(
    league_key="laliga",
    metric_key="attacking_threat",
    tool_name="laliga_teams_attacking_threat",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of teams."
    ),
)

laliga_teams_goals_tool = _make_team_tool(
    league_key="laliga",
    metric_key="goals",
    tool_name="laliga_teams_goals",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of teams."
    ),
)

laliga_teams_assists_tool = _make_team_tool(
    league_key="laliga",
    metric_key="assists",
    tool_name="laliga_teams_assists",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of teams."
    ),
)

laliga_teams_xg_total_tool = _make_team_tool(
    league_key="laliga",
    metric_key="xg_total",
    tool_name="laliga_teams_xg_total",
    description=(
        "Fetch LaLiga team rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of teams."
    ),
)

# Serie A teams – full metric set

serie_a_teams_1vs1_index_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="1vs1_index",
    tool_name="serie_a_teams_1vs1_index",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

serie_a_teams_1vs1_index_defensive_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="1vs1_index_defensive",
    tool_name="serie_a_teams_1vs1_index_defensive",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

serie_a_teams_1vs1_index_offensive_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="1vs1_index_offensive",
    tool_name="serie_a_teams_1vs1_index_offensive",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

serie_a_teams_offensive_interventions_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="offensive_interventions",
    tool_name="serie_a_teams_offensive_interventions",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of teams."
    ),
)

serie_a_teams_attacking_threat_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="attacking_threat",
    tool_name="serie_a_teams_attacking_threat",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of teams."
    ),
)

serie_a_teams_goals_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="goals",
    tool_name="serie_a_teams_goals",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of teams."
    ),
)

serie_a_teams_assists_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="assists",
    tool_name="serie_a_teams_assists",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of teams."
    ),
)

serie_a_teams_xg_total_tool = _make_team_tool(
    league_key="serie_a",
    metric_key="xg_total",
    tool_name="serie_a_teams_xg_total",
    description=(
        "Fetch Serie A team rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of teams."
    ),
)

# Ligue 1 teams – full metric set

ligue_1_teams_1vs1_index_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="1vs1_index",
    tool_name="ligue_1_teams_1vs1_index",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for the "
        "overall 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

ligue_1_teams_1vs1_index_defensive_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="1vs1_index_defensive",
    tool_name="ligue_1_teams_1vs1_index_defensive",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for the "
        "defensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

ligue_1_teams_1vs1_index_offensive_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="1vs1_index_offensive",
    tool_name="ligue_1_teams_1vs1_index_offensive",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for the "
        "offensive 1vs1 Index. Returns JSON with headers and rows of teams."
    ),
)

ligue_1_teams_offensive_interventions_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="offensive_interventions",
    tool_name="ligue_1_teams_offensive_interventions",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for "
        "offensive interventions. Returns JSON with headers and rows of teams."
    ),
)

ligue_1_teams_attacking_threat_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="attacking_threat",
    tool_name="ligue_1_teams_attacking_threat",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for "
        "attacking threat. Returns JSON with headers and rows of teams."
    ),
)

ligue_1_teams_goals_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="goals",
    tool_name="ligue_1_teams_goals",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for "
        "total goals. Returns JSON with headers and rows of teams."
    ),
)

ligue_1_teams_assists_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="assists",
    tool_name="ligue_1_teams_assists",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for "
        "total assists. Returns JSON with headers and rows of teams."
    ),
)

ligue_1_teams_xg_total_tool = _make_team_tool(
    league_key="ligue_1",
    metric_key="xg_total",
    tool_name="ligue_1_teams_xg_total",
    description=(
        "Fetch Ligue 1 team rankings from one-versus-one.com for "
        "total expected goals (xG). Returns JSON with headers and rows of teams."
    ),
)


# Predictions – overall + individual leagues
overall_predictions_tool = _make_predictions_tool(
    league_key="overall",
    tool_name="overall_predictions",
    description=(
        "Fetch the overall 1vs1 predictions overview page. Returns basic meta "
        "info and raw HTML."
    ),
)

pl_predictions_tool = _make_predictions_tool(
    league_key="premier_league",
    tool_name="pl_predictions",
    description=(
        "Fetch Premier League predictions from one-versus-one.com. Returns "
        "basic meta info and raw HTML for match predictions."
    ),
)

bundesliga_predictions_tool = _make_predictions_tool(
    league_key="bundesliga",
    tool_name="bundesliga_predictions",
    description=(
        "Fetch Bundesliga predictions from one-versus-one.com. Returns basic "
        "meta info and raw HTML for match predictions."
    ),
)

laliga_predictions_tool = _make_predictions_tool(
    league_key="laliga",
    tool_name="laliga_predictions",
    description=(
        "Fetch LaLiga predictions from one-versus-one.com. Returns basic meta "
        "info and raw HTML for match predictions."
    ),
)

serie_a_predictions_tool = _make_predictions_tool(
    league_key="serie_a",
    tool_name="serie_a_predictions",
    description=(
        "Fetch Serie A predictions from one-versus-one.com. Returns basic "
        "meta info and raw HTML for match predictions."
    ),
)

ligue_1_predictions_tool = _make_predictions_tool(
    league_key="ligue_1",
    tool_name="ligue_1_predictions",
    description=(
        "Fetch Ligue 1 predictions from one-versus-one.com. Returns basic "
        "meta info and raw HTML for match predictions."
    ),
)
