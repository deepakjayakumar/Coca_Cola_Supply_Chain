import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import query_snowflake, assign_orders_to_driver # Assuming your tools are in tools.py
import os
from dotenv import load_dotenv

load_dotenv() # This line loads the variables from the .env file

# Now, your script can access the API key
google_api_key = os.getenv("GOOGLE_API_KEY")
# Step 1: Set up the LLM
# Ensure your GOOGLE_API_KEY environment variable is set
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=google_api_key)

# Step 2: Combine the tools
tools = [query_snowflake, assign_orders_to_driver]

# Step 3: Define the prompt
SYSTEM_PROMPT = """
You are an expert supply chain and logistics planner for a large beverage company.
Your goal is to create an optimal delivery plan for today's orders.
You have access to a Snowflake database to get the necessary data and record your final plan.

Follow these steps to complete your task:

1.  **Understand the Request:** Your task is to process today's orders.
2.  **Gather Data:** Use the 'query_snowflake' tool to get a full list of orders from the 'ORDER_DETAILS' table, driver details from the 'DRIVER_DETAILS' table, and store details from the 'STORE_DETAILS' table. You must perform separate queries for each table. Do not assume you already have the data.
3.  **Analyze and Plan:**
    * Carefully analyze the orders. Consider the 'weight_kg' and 'quantity' of each order.
    * Consider the available drivers, their 'vehicle_capacity_kg', and 'hours_available'.
    * Make logical assignments by grouping orders for each driver. Your primary objective is to **balance the workload evenly among drivers** and **respect the vehicle capacity constraint**.
    * If a driver's vehicle cannot hold all the orders for a particular route, you must split the orders across multiple drivers.
    * Your plan must include an estimated time for each delivery in minutes.
4.  **Execute the Plan:** Once you have a final, complete, and optimized plan, use the 'assign_orders_to_driver' tool to record the assignments in the `driver_assignments` table.
5.  **Final Answer:** Conclude your response with a summary of the assignments and a confirmation that the data has been recorded. Do not include any SQL code in your final answer.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Step 4: Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Step 5: Run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# The user's input that kicks off the entire process
user_input = "Please assign all of today's pending orders to the available delivery drivers."

# Correct way to invoke the agent
response = agent_executor.invoke({
    "input": user_input, 
    "chat_history": [] # This is the missing variable
})

print("\n--- Final Agent Response ---")
print(response['output'])