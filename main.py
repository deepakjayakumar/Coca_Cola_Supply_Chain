import os
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from tools import query_snowflake, assign_orders_to_driver

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=google_api_key)

# Combine the tools
tools = [query_snowflake]

# Define a more robust prompt for manual agent creation
SYSTEM_PROMPT = """
You are an expert supply chain and logistics planner. Your mission is to assign all pending orders to available drivers.

You must follow this procedure precisely:
1.  **FIRST**, use query_snowflake tool to get the most up-to-date data on all pending orders (order_status = 'Pending') from the `order_details` table. Do not worry about the date.Use WEIGHT_KG column for weight calculation. Use only these columns (ORDER_ID,STORE_ID,ORDER_DATE,QUANTITY,WEIGHT_KG,ORDER_STATUS)
2.  **NEXT**, get the details of all available drivers from the `driver_details` table. No Driver status is needed, get all driver details.
3.  **NEXT**, get the location data (column names - latitude, longitude) for all stores from the `store_details` table.
4.  **NEXT**, You are an expert supply chain and logistics planner. Your mission is to get the necessary data and create an optimal delivery plan. YOU MUST NOT USE any tools to execute the plan; you will simply provide a final list of assignments in a clean format.



"""

# Correct and simplified prompt for tool-calling agents
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    HumanMessage(content="{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Build the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# The user's input that kicks off the entire process
user_input = "Please assign all of today's pending orders to the available delivery drivers and display them like a table."
response = agent_executor.invoke({"input": user_input})

print("\n--- Final Agent Response ---")
print(response['output'])