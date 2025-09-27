import datetime
import os
from dotenv import load_dotenv
import gradio as gr
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from tools import query_snowflake, calculate_distance
import time # Used for simulation

# --- Configuration ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set up the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7, google_api_key=google_api_key)

# Combine the tools (Note: We're not using assign_orders_to_driver here, as requested)
tools = [query_snowflake, calculate_distance]

# The standard Coca-Cola brand colors
COCA_COLA_RED_PRIMARY = "#CC0000" # True Coca-Cola Red
COCA_COLA_BLACK_TEXT = "#333333" # Dark Gray/Black (for body text)
COCA_COLA_WHITE_BG = "#FFFFFF"   # Crisp White

# --- THE COCA-COLA THEME ---

# Set up the theme with the new white base
COCA_COLA_THEME = gr.Theme(
    # Base configuration: Primary Red, Neutral Dark Gray/Black
    primary_hue="red",
    secondary_hue="gray",
    neutral_hue="gray",
    font=["Arial", "Helvetica", "sans-serif"],
).set(
    # Override defaults to force a white paper look
    body_background_fill=COCA_COLA_WHITE_BG,
    background_fill_primary=COCA_COLA_WHITE_BG,
    background_fill_secondary="#F0F0F0",
    
    # Text colors are inherited from neutral_hue (which is gray/black) but we
    # will enforce the background for the primary content area to be white.
)


CUSTOM_CSS = f"""
/* 1. Ensure the background is white (overrides default body color) */
body {{
    background-color: {COCA_COLA_WHITE_BG} !important;
    color: {COCA_COLA_BLACK_TEXT} !important;
    margin: 0;
    padding: 0;
}}

/* 2. Make the container sharp white with a subtle shadow */
.gradio-container {{
    background-color: {COCA_COLA_WHITE_BG} !important; 
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); 
    padding: 30px; 
    margin: 30px auto; 
    max-width: 1200px;
}}

/* 3. Headings and Primary Text (Red and Black) */
h1 {{ 
    color: {COCA_COLA_RED_PRIMARY} !important; /* Coca-Cola Red */
    font-weight: 900 !important; 
}}
h3 {{ 
    color: {COCA_COLA_BLACK_TEXT} !important; 
    font-weight: 600 !important;
}}
p, label, .gr-markdown, .gr-checkbox-label, .gr-label {{ 
    color: {COCA_COLA_BLACK_TEXT} !important; 
}}

/* 4. Button Styling (Crisp Red) */
.gr-button-primary {{
    background-color: {COCA_COLA_RED_PRIMARY} !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
}}

.gr-button-secondary {{
    background-color: {COCA_COLA_BLACK_TEXT} !important; /* Black/Dark Grey */
    color: white !important;
    border: none !important;
}}

/* 5. Dataframe Styling (Red Underline) */
.gr-dataframe thead th {{
    color: {COCA_COLA_BLACK_TEXT} !important; 
    background-color: #EFEFEF !important;
    border-bottom: 2px solid {COCA_COLA_RED_PRIMARY} !important; /* Red underline */
    font-weight: bold !important;
}}
.gr-dataframe tbody td {{
    color: {COCA_COLA_BLACK_TEXT} !important;
}}
"""
# ... rest of your app.py code ...
# --- Agent Setup ---
# The agent's logic is now to gather data, calculate, and provide the final plan.
SYSTEM_PROMPT = """
You are an expert supply chain and logistics planner. Your mission is to get the necessary data and create an optimal delivery plan.

You must follow this procedure precisely:
1.  **FIRST**, get the most up-to-date data on all pending orders from the `order_details` table. Use all available columns for the query.
2.  **NEXT**, get the details of all available drivers from the `driver_details` table.
3.  **NEXT**, get the location data for all stores from `STORE_DETAILS` table  and warehouses from `WAREHOUSE_DETAILS` table.
4.  **FINALLY**, analyze the data and the calculated distances (using the calculate_distance tool for optimization) and provide a detailed, clean list of assignments in your final answer.
    * The assignments must be optimized based on the shortest distance from the driver's warehouse to the assigned stores.
    * Format the output clearly with Driver Name, Order IDs, and total assigned weight.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    HumanMessage(content="{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- Gradio Helper Functions ---

# Function to get the initial order data in a table format
def get_initial_orders():
    # Corrected SQL query to explicitly select the columns in the order the Gradio table expects them
    query = "SELECT ORDER_ID, STORE_ID, TO_VARCHAR(ORDER_DATE,'YYYY-MM-DD') AS ORDER_DATE, QUANTITY, PRODUCT_NAME FROM order_details WHERE ORDER_STATUS = 'Pending' LIMIT 15"
    orders_data = query_snowflake.invoke({'sql_query': query})
    
    # Check for empty data before processing
    if not orders_data:
        return []
        
    orders_list = []
    
    # Process the data rows
    if isinstance(orders_data, str):
        # Handle the case where the output is a raw string/tuples (common from invoke)
        import re
        for match in re.findall(r'\((.*?)\)', orders_data):
            items = [item.strip().strip("'\"").split('(')[0] for item in match.split(',')]
            orders_list.append(items)
    else:
        # If the output is a list of tuples (common when running locally)
        for row in orders_data:
            # Check if the row is empty before accessing its elements
            if not row:
                continue
            new_row = list(row)
            
            # Ensure index 2 exists before trying to access it
            if len(new_row) > 2 and isinstance(new_row[2], datetime.date):
                new_row[2] = new_row[2].strftime('%Y-%m-%d')
                
            orders_list.append(new_row)
            
    return orders_list

# The Generator Function: The core of the sequential updates
# The Generator Function: The core of the sequential updates
def run_and_stream(user_input, existing_table_data):
    # This is where we capture the final plan, which is what the agent uses to say "How I did it"
    initial_orders_data = get_initial_orders()
    log_output = "--- AGENT LOG (Verbose) ---\n"
    
    # 1. Start Log
    log_output += f"[{time.strftime('%H:%M:%S')}] 🟢 Agent Initializing...\n"
    yield initial_orders_data, [], log_output 
    time.sleep(1) 

    # 2. Query/Reasoning Steps (Logs update, Tables stay static)
    log_output += f"[{time.strftime('%H:%M:%S')}] 📊 Gathering Data from Snowflake (4 tables)...\n"
    yield initial_orders_data, [], log_output
    time.sleep(1) 

    # 3. Agent Reasoning (LLM Call)
    log_output += f"[{time.strftime('%H:%M:%S')}] 🧠 Running Optimization Algorithm...\n"
    yield initial_orders_data, [], log_output
    time.sleep(1) 
    
    # --- CRITICAL STEP: Run Agent and Capture Final Text ---
    full_response = agent_executor.invoke({"input": user_input})
    assignment_plan_text = full_response['output']
    
    # Step 4: Assignment Complete (The Visual Switch)
    log_output += f"[{time.strftime('%H:%M:%S')}] ✅ Assignment Success! Moving Orders to Dispatch Queue.\n"
    
    # --- NEW FEATURE: How I Did It Summary ---
    log_output += "\n--- AGENT SUMMARY: How I Optimized the Plan ---\n"
    log_output += assignment_plan_text
    
    # Output the final state: Left side empty, Right side shows the initial data.
    yield [], initial_orders_data, log_output 
# Function to initialize the UI state
def initialize_ui():
    #initial_orders = get_initial_orders()
    return [], [], "--- AGENT LOG (Ready) ---"

# --- Gradio Blocks Layout ---
with gr.Blocks(theme=COCA_COLA_THEME, css=CUSTOM_CSS,title="Agentic AI Supply Chain PoC") as demo:
    gr.Markdown("# Coca-Cola Delivery Optimization Agent ")
    gr.Markdown("### *Demonstrating Autonomous Logistics Planning*")

    # ------------------- RUN BUTTON --------------------
    with gr.Row():
        run_btn = gr.Button("▶️ RUN OPTIMIZATION AGENT", variant="primary", scale=5)
    
    # FIX: Ensure the Column has the 'scale' property. 
    # This correctly allocates the horizontal space for the Markdown component.
        with gr.Column(scale=3): 
            gr.Markdown("*(Orders move right once planning is complete)*")
        
        #clear_btn = gr.Button("Clear", variant="secondary", scale=0)

    # ------------------- TOP ROW: INPUT/OUTPUT PANELS --------------------
    
    ORDER_TABLE_HEADERS = ["Order ID", "Store ID", "Date", "Quantity", "Product"]

    with gr.Row():
        
        # LEFT COLUMN: NEW ORDERS (Input)
        with gr.Column(scale=1):
            gr.Markdown("### 📦 NEW ORDERS (Awaiting Delivery Optimization)")
            new_orders_table = gr.Dataframe(
                headers=ORDER_TABLE_HEADERS,
                col_count=5,
                interactive=False,
                row_count=15
            )

        # RIGHT COLUMN: ASSIGNED ORDERS (Output)
        with gr.Column(scale=1):
            gr.Markdown("### ✅ ASSIGNED ORDERS (Ready for Dispatch)")
             # CHANGE THIS: Use a Dataframe instead of a Textbox
            assigned_orders_output = gr.Dataframe( 
                headers=ORDER_TABLE_HEADERS,
                col_count=5,
                interactive=False,
                row_count=15,
                value=[]
            )
            
    # ------------------- BOTTOM ROW: LOG SCREEN --------------------
    with gr.Row():
        log_output_box = gr.Textbox(
            label="AGENT PROCESSING LOG (Transparent Execution)",
            lines=15,
            interactive=False,
            autoscroll=True
        )

    # --- Event Handling ---
    
    # Initialize the UI on app load
    demo.load(
        fn=initialize_ui,
        inputs=[],
        outputs=[new_orders_table, assigned_orders_output, log_output_box]
    )

    # Link the run button to the generator function for sequential updates
    run_btn.click(
        fn=run_and_stream,
        inputs=[gr.Textbox(value="Please assign all pending orders.", visible=False), new_orders_table],
        outputs=[new_orders_table, assigned_orders_output, log_output_box],
    )


# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_port=7860)