import os
from dotenv import load_dotenv
import gradio as gr
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from tools import query_snowflake, assign_orders_to_driver, calculate_distance

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7, google_api_key=google_api_key)

# Combine the tools
tools = [query_snowflake, calculate_distance]

# Define the prompt
# The updated SYSTEM_PROMPT in your app.py file
SYSTEM_PROMPT = """
You are an expert supply chain and logistics planner. Your mission is to get the necessary data and create an optimal delivery plan. All drivers in the `driver_details` table are considered available.

You must follow this procedure precisely:
1.  **FIRST**, use query_snowflake tool to get the most up-to-date data on all pending orders (order_status = 'Pending') from the `order_details` table. Do not worry about the date.Use WEIGHT_KG column for weight calculation. Use only these columns (ORDER_ID,STORE_ID,ORDER_DATE,QUANTITY,WEIGHT_KG,ORDER_STATUS)
2.  **NEXT**, get the details of all available drivers from the `driver_details` table. All drivers in this table are considered available for assignment.
3.  **NEXT**, get the location data for all stores from the `store_details` table.
4.  **NEXT**, get the location data for the warehouses from the `warehouse_details` table.
5.  **NEXT**, you **must** use the `calculate_distance` tool to determine the distance between each driver's warehouse and each store.
6.  **FINALLY**, analyze the data and provide a detailed, clean list of assignments in your final answer.
    * The assignments must be optimized based on the shortest distance from the driver's warehouse to the assigned stores.
    * Please structure the final output clearly, showing the driver,all the stores assigned to the driver, the total weight of the delivery, and the route the driver should follow.  (All response should only be in english)
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    HumanMessage(content="{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Build the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# The function that Gradio will use to run the agent
def run_agent_gradio(user_input):
    response = agent_executor.invoke({"input": user_input})
    return response['output']

# Gradio Interface setup
iface = gr.Interface(
    fn=run_agent_gradio,
    inputs=gr.Textbox(lines=5, label="Enter your request for the agent"),
    outputs=gr.Textbox(lines=10, label="Agent's Response"),
    title="Agentic AI Supply Chain PoC",
    description="Ask the agent to create an optimal delivery plan."
)

if __name__ == "__main__":
    iface.launch()