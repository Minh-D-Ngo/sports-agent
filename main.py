from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from tools import (
    # Generic utilities
    search_tool,
    wiki_tool,
    save_tool,
    sports_news_tool,

    # -----------------
    # Overall (Top Five Europe) – players
    # -----------------
    overall_players_1vs1_index_tool,
    overall_players_1vs1_index_defensive_tool,
    overall_players_1vs1_index_offensive_tool,
    overall_players_offensive_interventions_tool,
    overall_players_attacking_threat_tool,
    overall_players_goals_tool,
    overall_players_assists_tool,
    overall_players_xg_total_tool,

    # -----------------
    # Premier League – players
    # -----------------
    pl_players_1vs1_index_tool,
    pl_players_1vs1_index_defensive_tool,
    pl_players_1vs1_index_offensive_tool,
    pl_players_offensive_interventions_tool,
    pl_players_attacking_threat_tool,
    pl_players_goals_tool,
    pl_players_assists_tool,
    pl_players_xg_total_tool,

    # -----------------
    # Bundesliga – players
    # -----------------
    bundesliga_players_1vs1_index_tool,
    bundesliga_players_1vs1_index_defensive_tool,
    bundesliga_players_1vs1_index_offensive_tool,
    bundesliga_players_offensive_interventions_tool,
    bundesliga_players_attacking_threat_tool,
    bundesliga_players_goals_tool,
    bundesliga_players_assists_tool,
    bundesliga_players_xg_total_tool,

    # -----------------
    # LaLiga – players
    # -----------------
    laliga_players_1vs1_index_tool,
    laliga_players_1vs1_index_defensive_tool,
    laliga_players_1vs1_index_offensive_tool,
    laliga_players_offensive_interventions_tool,
    laliga_players_attacking_threat_tool,
    laliga_players_goals_tool,
    laliga_players_assists_tool,
    laliga_players_xg_total_tool,

    # -----------------
    # Serie A – players
    # -----------------
    serie_a_players_1vs1_index_tool,
    serie_a_players_1vs1_index_defensive_tool,
    serie_a_players_1vs1_index_offensive_tool,
    serie_a_players_offensive_interventions_tool,
    serie_a_players_attacking_threat_tool,
    serie_a_players_goals_tool,
    serie_a_players_assists_tool,
    serie_a_players_xg_total_tool,

    # -----------------
    # Ligue 1 – players
    # -----------------
    ligue_1_players_1vs1_index_tool,
    ligue_1_players_1vs1_index_defensive_tool,
    ligue_1_players_1vs1_index_offensive_tool,
    ligue_1_players_offensive_interventions_tool,
    ligue_1_players_attacking_threat_tool,
    ligue_1_players_goals_tool,
    ligue_1_players_assists_tool,
    ligue_1_players_xg_total_tool,

    # -----------------
    # Overall – teams
    # -----------------
    overall_teams_1vs1_index_tool,
    overall_teams_1vs1_index_defensive_tool,
    overall_teams_1vs1_index_offensive_tool,
    overall_teams_offensive_interventions_tool,
    overall_teams_attacking_threat_tool,
    overall_teams_goals_tool,
    overall_teams_assists_tool,
    overall_teams_xg_total_tool,

    # -----------------
    # Premier League – teams
    # -----------------
    pl_teams_1vs1_index_tool,
    pl_teams_1vs1_index_defensive_tool,
    pl_teams_1vs1_index_offensive_tool,
    pl_teams_offensive_interventions_tool,
    pl_teams_attacking_threat_tool,
    pl_teams_goals_tool,
    pl_teams_assists_tool,
    pl_teams_xg_total_tool,

    # -----------------
    # Bundesliga – teams
    # -----------------
    bundesliga_teams_1vs1_index_tool,
    bundesliga_teams_1vs1_index_defensive_tool,
    bundesliga_teams_1vs1_index_offensive_tool,
    bundesliga_teams_offensive_interventions_tool,
    bundesliga_teams_attacking_threat_tool,
    bundesliga_teams_goals_tool,
    bundesliga_teams_assists_tool,
    bundesliga_teams_xg_total_tool,

    # -----------------
    # LaLiga – teams
    # -----------------
    laliga_teams_1vs1_index_tool,
    laliga_teams_1vs1_index_defensive_tool,
    laliga_teams_1vs1_index_offensive_tool,
    laliga_teams_offensive_interventions_tool,
    laliga_teams_attacking_threat_tool,
    laliga_teams_goals_tool,
    laliga_teams_assists_tool,
    laliga_teams_xg_total_tool,

    # -----------------
    # Serie A – teams
    # -----------------
    serie_a_teams_1vs1_index_tool,
    serie_a_teams_1vs1_index_defensive_tool,
    serie_a_teams_1vs1_index_offensive_tool,
    serie_a_teams_offensive_interventions_tool,
    serie_a_teams_attacking_threat_tool,
    serie_a_teams_goals_tool,
    serie_a_teams_assists_tool,
    serie_a_teams_xg_total_tool,

    # -----------------
    # Ligue 1 – teams
    # -----------------
    ligue_1_teams_1vs1_index_tool,
    ligue_1_teams_1vs1_index_defensive_tool,
    ligue_1_teams_1vs1_index_offensive_tool,
    ligue_1_teams_offensive_interventions_tool,
    ligue_1_teams_attacking_threat_tool,
    ligue_1_teams_goals_tool,
    ligue_1_teams_assists_tool,
    ligue_1_teams_xg_total_tool,

    # -----------------
    # Predictions
    # -----------------
    overall_predictions_tool,
    pl_predictions_tool,
    bundesliga_predictions_tool,
    laliga_predictions_tool,
    serie_a_predictions_tool,
    ligue_1_predictions_tool,
)



load_dotenv()


class ResearchResponse(BaseModel):
    """
    Standard structured response type for all agents.

    Attributes:
        topic: Short title describing the main subject.
        summary: Natural language explanation / answer.
        sources: List of data or URL sources referenced.
        tools_used: List of tool names the agent invoked.
    """
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# -------------------------
# LLM and parser
# -------------------------

# OpenAI-only LLM used by all agents
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)

# Pydantic parser enforcing JSON output structure
parser = PydanticOutputParser(pydantic_object=ResearchResponse)


# -------------------------
# DraftKings-focused agent
# -------------------------

# Tools available to the DraftKings agent
draftkings_tools = [search_tool, wiki_tool, save_tool]

# Prompt instructing the DraftKings agent
draftkings_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant specialized in DraftKings.

            Your job is to:
            - Answer questions about DraftKings as a company (business model, products, history, leadership, financials, etc.).
            - Help users navigate and understand the DraftKings website (e.g., where to find certain pages or features).
            - Use web search and Wikipedia tools when needed to retrieve up-to-date or detailed information.
            - If a question is not about DraftKings or its website or closely related topics, say that another agent in the system may handle it.

            Wrap your final answer in this exact JSON structure and provide no other text:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# DraftKings agent and executor
draftkings_agent = create_tool_calling_agent(
    llm=llm,
    prompt=draftkings_prompt,
    tools=draftkings_tools,
)

draftkings_executor = AgentExecutor(
    agent=draftkings_agent,
    tools=draftkings_tools,
    verbose=True,
)


# -------------------------
# Soccer / stats agent (1vs1 + news)
# -------------------------

# Tools available to the soccer agent:
# - sports_news_tool & search_tool: general news/info from the web
# - 1vs1 tools: structured Premier League player rankings for different stats
sports_tools = [
    # Generic info tools
    sports_news_tool,
    search_tool,

    # -----------------
    # Overall (Top Five Europe) – players
    # -----------------
    overall_players_1vs1_index_tool,
    overall_players_1vs1_index_defensive_tool,
    overall_players_1vs1_index_offensive_tool,
    overall_players_offensive_interventions_tool,
    overall_players_attacking_threat_tool,
    overall_players_goals_tool,
    overall_players_assists_tool,
    overall_players_xg_total_tool,

    # -----------------
    # Premier League – players
    # -----------------
    pl_players_1vs1_index_tool,
    pl_players_1vs1_index_defensive_tool,
    pl_players_1vs1_index_offensive_tool,
    pl_players_offensive_interventions_tool,
    pl_players_attacking_threat_tool,
    pl_players_goals_tool,
    pl_players_assists_tool,
    pl_players_xg_total_tool,

    # -----------------
    # Bundesliga – players
    # -----------------
    bundesliga_players_1vs1_index_tool,
    bundesliga_players_1vs1_index_defensive_tool,
    bundesliga_players_1vs1_index_offensive_tool,
    bundesliga_players_offensive_interventions_tool,
    bundesliga_players_attacking_threat_tool,
    bundesliga_players_goals_tool,
    bundesliga_players_assists_tool,
    bundesliga_players_xg_total_tool,

    # -----------------
    # LaLiga – players
    # -----------------
    laliga_players_1vs1_index_tool,
    laliga_players_1vs1_index_defensive_tool,
    laliga_players_1vs1_index_offensive_tool,
    laliga_players_offensive_interventions_tool,
    laliga_players_attacking_threat_tool,
    laliga_players_goals_tool,
    laliga_players_assists_tool,
    laliga_players_xg_total_tool,

    # -----------------
    # Serie A – players
    # -----------------
    serie_a_players_1vs1_index_tool,
    serie_a_players_1vs1_index_defensive_tool,
    serie_a_players_1vs1_index_offensive_tool,
    serie_a_players_offensive_interventions_tool,
    serie_a_players_attacking_threat_tool,
    serie_a_players_goals_tool,
    serie_a_players_assists_tool,
    serie_a_players_xg_total_tool,

    # -----------------
    # Ligue 1 – players
    # -----------------
    ligue_1_players_1vs1_index_tool,
    ligue_1_players_1vs1_index_defensive_tool,
    ligue_1_players_1vs1_index_offensive_tool,
    ligue_1_players_offensive_interventions_tool,
    ligue_1_players_attacking_threat_tool,
    ligue_1_players_goals_tool,
    ligue_1_players_assists_tool,
    ligue_1_players_xg_total_tool,

    # -----------------
    # Overall – teams
    # -----------------
    overall_teams_1vs1_index_tool,
    overall_teams_1vs1_index_defensive_tool,
    overall_teams_1vs1_index_offensive_tool,
    overall_teams_offensive_interventions_tool,
    overall_teams_attacking_threat_tool,
    overall_teams_goals_tool,
    overall_teams_assists_tool,
    overall_teams_xg_total_tool,

    # -----------------
    # Premier League – teams
    # -----------------
    pl_teams_1vs1_index_tool,
    pl_teams_1vs1_index_defensive_tool,
    pl_teams_1vs1_index_offensive_tool,
    pl_teams_offensive_interventions_tool,
    pl_teams_attacking_threat_tool,
    pl_teams_goals_tool,
    pl_teams_assists_tool,
    pl_teams_xg_total_tool,

    # -----------------
    # Bundesliga – teams
    # -----------------
    bundesliga_teams_1vs1_index_tool,
    bundesliga_teams_1vs1_index_defensive_tool,
    bundesliga_teams_1vs1_index_offensive_tool,
    bundesliga_teams_offensive_interventions_tool,
    bundesliga_teams_attacking_threat_tool,
    bundesliga_teams_goals_tool,
    bundesliga_teams_assists_tool,
    bundesliga_teams_xg_total_tool,

    # -----------------
    # LaLiga – teams
    # -----------------
    laliga_teams_1vs1_index_tool,
    laliga_teams_1vs1_index_defensive_tool,
    laliga_teams_1vs1_index_offensive_tool,
    laliga_teams_offensive_interventions_tool,
    laliga_teams_attacking_threat_tool,
    laliga_teams_goals_tool,
    laliga_teams_assists_tool,
    laliga_teams_xg_total_tool,

    # -----------------
    # Serie A – teams
    # -----------------
    serie_a_teams_1vs1_index_tool,
    serie_a_teams_1vs1_index_defensive_tool,
    serie_a_teams_1vs1_index_offensive_tool,
    serie_a_teams_offensive_interventions_tool,
    serie_a_teams_attacking_threat_tool,
    serie_a_teams_goals_tool,
    serie_a_teams_assists_tool,
    serie_a_teams_xg_total_tool,

    # -----------------
    # Ligue 1 – teams
    # -----------------
    ligue_1_teams_1vs1_index_tool,
    ligue_1_teams_1vs1_index_defensive_tool,
    ligue_1_teams_1vs1_index_offensive_tool,
    ligue_1_teams_offensive_interventions_tool,
    ligue_1_teams_attacking_threat_tool,
    ligue_1_teams_goals_tool,
    ligue_1_teams_assists_tool,
    ligue_1_teams_xg_total_tool,

    # -----------------
    # Predictions
    # -----------------
    overall_predictions_tool,
    pl_predictions_tool,
    bundesliga_predictions_tool,
    laliga_predictions_tool,
    serie_a_predictions_tool,
    ligue_1_predictions_tool,
]



# Prompt instructing the soccer/stats agent
sports_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a soccer (football) news and insights assistant.

            Your job is to:
            - Provide latest news, scores, standings, and basic analysis for soccer.
            - Use the sports news search tool and general web search for news and match info.
            - Use the 1vs1 Premier League ranking tools when the user asks about top players,
              player rankings, or who is strongest in specific metrics (overall index, defensive,
              offensive, goals, assists, xG, etc.).
            - Read the JSON output from these tools (headers and rows) and use it to support
              your analysis and comparisons.
            - Use stats only to provide information-based insights and context. Do NOT guarantee
              betting outcomes or provide sure-fire picks. Always remind the user that betting is risky.

            If a question is specifically about DraftKings as a company or its website,
            say that another agent in the system should handle it.

            Wrap your final answer in this exact JSON structure and provide no other text:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Soccer agent and executor
sports_agent = create_tool_calling_agent(
    llm=llm,
    prompt=sports_prompt,
    tools=sports_tools,
)

sports_executor = AgentExecutor(
    agent=sports_agent,
    tools=sports_tools,
    verbose=True,
)


# -------------------------
# Simple router + merger
# -------------------------

def route_query(user_query: str) -> list[tuple[str, ResearchResponse]]:
    """
    Route an incoming user query to one or both agents and parse their outputs.

    Routing rules (simple keyword-based):
        - If the query clearly asks for rankings / top players, use the soccer agent.
        - If it mentions DraftKings-related terms, use the DraftKings agent.
        - If it is soccer-related but not DraftKings-specific, use the soccer agent.
        - If nothing matches, fallback to the DraftKings agent.

    Args:
        user_query: Raw user text input.

    Returns:
        A list of (agent_name, ResearchResponse) tuples. One or more responses
        may be returned depending on the routing logic.
    """
    q_lower = user_query.lower()

    dk_keywords = ["draftkings", "sportsbook", "dfs", "daily fantasy", "casino"]
    sports_keywords = [
        "score", "odds", "spread", "line", "game", "match", "league", "team",
        "player", "premier league", "champions league", "la liga", "serie a",
    ]
    rankings_keywords = [
    "top players",
    "best players",
    "player rankings",
    "player ranking",
    "rankings",
    "who is the best",
    "who are the best",
    "1vs1 index",
    "1vs1-index",
    "top scorer",
    "top scorers",
    "most goals",
    "most assists",
    "xg leaders",
    "predictions",
    "prediction",
    "win probability",]


    use_dk = any(k in q_lower for k in dk_keywords)
    use_sports = any(k in q_lower for k in sports_keywords)
    use_rankings = any(k in q_lower for k in rankings_keywords)

    results: list[tuple[str, ResearchResponse]] = []

    # 1) If explicitly rankings / top players, prefer the soccer agent
    if use_rankings:
        sports_raw = sports_executor.invoke({"query": user_query})
        try:
            sports_output_str = sports_raw.get("output")
            sports_structured = parser.parse(sports_output_str)
            results.append(("sports", sports_structured))
        except Exception as e:
            print("Sports agent parse error (rankings):", e, sports_raw)
        return results

    # 2) DraftKings agent if DK-related
    if use_dk:
        dk_raw = draftkings_executor.invoke({"query": user_query})
        try:
            dk_output_str = dk_raw.get("output")
            dk_structured = parser.parse(dk_output_str)
            results.append(("draftkings", dk_structured))
        except Exception as e:
            print("DraftKings agent parse error:", e, dk_raw)

    # 3) Sports agent for soccer-related queries (but not DK-specific)
    if use_sports and not use_dk:
        sports_raw = sports_executor.invoke({"query": user_query})
        try:
            sports_output_str = sports_raw.get("output")
            sports_structured = parser.parse(sports_output_str)
            results.append(("sports", sports_structured))
        except Exception as e:
            print("Sports agent parse error:", e, sports_raw)

    # 4) Fallback: if nothing matched, default to DraftKings agent
    if not results:
        dk_raw = draftkings_executor.invoke({"query": user_query})
        try:
            dk_output_str = dk_raw.get("output")
            dk_structured = parser.parse(dk_output_str)
            results.append(("draftkings", dk_structured))
        except Exception as e:
            print("DraftKings agent (fallback) parse error:", e, dk_raw)

    return results


if __name__ == "__main__":
    """
    Entry point for running the chatbot in a simple CLI loop.

    - Reads a single query from stdin.
    - Routes it to the appropriate agent(s).
    - If only one agent responds, prints that response.
    - If multiple agents respond, merges their ResearchResponse objects into
      a single combined response.
    """
    query = input("Ask about DraftKings or soccer: ")
    responses = route_query(query)

    if not responses:
        print("No agent produced a response.")
    elif len(responses) == 1:
        agent_name, resp = responses[0]
        print(f"\n=== COMBINED RESPONSE (from {agent_name}) ===")
        print(resp)
    else:
        # Merge multiple ResearchResponse objects into one combined response
        topics = {r.topic for _, r in responses}
        summaries = [r.summary for _, r in responses]
        sources = {s for _, r in responses for s in r.sources}
        tools_used = {t for _, r in responses for t in r.tools_used}

        combined = ResearchResponse(
            topic=" + ".join(topics),
            summary=" ".join(summaries),
            sources=list(sources),
            tools_used=list(tools_used),
        )

        print("\n=== COMBINED RESPONSE (merged agents) ===")
        print(combined)
