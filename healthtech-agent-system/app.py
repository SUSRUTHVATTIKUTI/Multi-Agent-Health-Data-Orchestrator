import asyncio
import logging
import httpx

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from autogen_ext.models.openai import OpenAIChatCompletionClient


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.WARNING, # <-- Now it will only print errors/warnings
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------------
# Tool
# ----------------------------
async def fetch_health_data(disease_query: str) -> str:
    """
    Fetch live epidemiological statistics.
    """

    logger.info(f"Fetching data for: {disease_query}")

    url = "https://disease.sh/v3/covid-19/all"

    headers = {
        "User-Agent": "HealthTech-Agent-System"
    }

    try:

        async with httpx.AsyncClient(timeout=10.0) as client:

            response = await client.get(url, headers=headers)

            response.raise_for_status()

            data = response.json()

        logger.info("Successfully fetched live data.")

        return f"""
Requested Disease:
{disease_query}

Global Viral Statistics

Today's Cases: {data.get("todayCases")}

Today's Deaths: {data.get("todayDeaths")}

Today's Recovered: {data.get("todayRecovered")}

Active Cases: {data.get("active")}

Critical Cases: {data.get("critical")}

Total Cases: {data.get("cases")}

Total Deaths: {data.get("deaths")}

Recovered: {data.get("recovered")}
"""

    except Exception as e:

        logger.error(f"API Error: {e}")

        return f"""
Unable to retrieve live data.

Reason:
{str(e)}

Continue using general medical knowledge.
"""


# ----------------------------
# Main
# ----------------------------
async def main():

    logger.info("Initializing model...")

    model_client = OpenAIChatCompletionClient(

        model="qwen2.5:3b",

        base_url="http://localhost:11434/v1",

        api_key="ollama",

        temperature=0.2,

        model_info={

            "vision": False,

            "function_calling": True,

            "json_output": True,

            "structured_output": False,

            "family": "unknown",
        },
    )

    # ----------------------------
    # Agent 1
    # ----------------------------

    researcher = AssistantAgent(

        name="Medical_Researcher",

        model_client=model_client,

        tools=[fetch_health_data],

        system_message="""
You are a Medical Researcher.

Always call the fetch_health_data tool exactly once.

Never invent outbreak statistics.

Summarize the retrieved data clearly.

If the tool fails,
mention that live data was unavailable.
""",
    )

    # ----------------------------
    # Agent 2
    # ----------------------------

    analyst = AssistantAgent(

        name="Clinical_Analyst",

        model_client=model_client,

        system_message="""
You are a Clinical Data Analyst.

Using the researcher's findings:

1. Identify important outbreak trends.

2. Identify high-risk groups.

3. Suggest TWO HealthTech solutions.

4. Explain why each solution works.

Keep the answer concise.
""",
    )

    # ----------------------------
    # Agent 3
    # ----------------------------

    supervisor = AssistantAgent(

        name="Medical_Director",

        model_client=model_client,

        system_message="""
You are the Medical Director.

Review the analyst's work.

Produce:

Overall Assessment

Priority Recommendation

Finally output exactly:

TERMINATE
""",
    )

    termination = TextMentionTermination("TERMINATE")

    team = RoundRobinGroupChat(

        participants=[

            researcher,

            analyst,

            supervisor,

        ],

        termination_condition=termination,
    )

    logger.info("Running multi-agent workflow...")

    await Console(

        team.run_stream(

            task="""
Analyze the current global viral outbreak.

Use live data.

Recommend AI and HealthTech interventions.
"""
        )
    )

    await model_client.close()

    logger.info("Completed.")


if __name__ == "__main__":

    asyncio.run(main())