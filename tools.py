import snowflake.connector
from langchain.tools import tool

@tool
def query_snowflake(sql_query: str) -> str:
    """
    This tool executes a read-only SQL query on the Snowflake database and returns the results.
    Use this tool to get information about orders, drivers, and stores.
    The input is the SQL query as a string.
    """
    try:
        # Replace with your actual Snowflake connection details
        conn = snowflake.connector.connect(
            user='DKUMARJAYAKUMAR',
            password='Admin@1122334455',
            account='CIXVEMB-MHB25679',
            warehouse='AGENTIC_WH',
            database='COCA_COLA_SUPPLY_CHAIN',
            schema='AGENT'
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Simple formatting of the results
        formatted_results = "\n".join([str(row) for row in rows])
        
        return formatted_results
    except Exception as e:
        return f"An error occurred: {e}"
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# Example of how you would call the tool in a script (for testing)
if __name__ == '__main__':
    test_query = "SELECT order_id, quantity, store_id FROM order_details LIMIT 5;"
    results = query_snowflake.invoke(test_query)
    print("Query Results:\n", results)


import snowflake.connector
import datetime
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List

# 1. Define a Pydantic model for a single assignment
class Assignment(BaseModel):
    """Details of a single driver-to-order assignment."""
    order_id: int = Field(description="The unique identifier of the order.")
    driver_id: int = Field(description="The unique identifier of the driver.")
    estimated_time_minutes: int = Field(description="The estimated time in minutes for the delivery.")

# 2. Update the tool function to accept a list of this model
@tool
def assign_orders_to_driver(assignments: List[Assignment]) -> str:
    """
    Assigns a list of orders to drivers in the driver_assignments table.
    The input is a list of Pydantic Assignment models.
    """
    if not assignments:
        return "No assignments to process."
    
    try:
        conn = snowflake.connector.connect(
            user='DKUMARJAYAKUMAR',
            password='Admin@1122334455',
            account='CIXVEMB-MHB25679',
            warehouse='AGENTIC_WH',
            database='COCA_COLA_SUPPLY_CHAIN',
            schema='AGENT'
        )
        cursor = conn.cursor()
        
        insert_sql = """
            INSERT INTO COCA_COLA_SUPPLY_CHAIN.AGENT.DRIVER_ASSIGNMENTS (
                order_id,
                driver_id,
                assigned_date,
                assigned_timestamp,
                status,
                estimated_time_minutes
            ) VALUES (%s, %s, %s, %s, %s, %s);
        """
        
        # Prepare the data for insertion from the Pydantic models
        records_to_insert = [
            (
                a.order_id,
                a.driver_id,
                datetime.date.today(),
                datetime.datetime.now(),
                'Assigned',
                a.estimated_time_minutes
            ) for a in assignments
        ]
        
        cursor.executemany(insert_sql, records_to_insert)
        conn.commit()
        
        return f"Successfully assigned {len(assignments)} orders to drivers."
    except Exception as e:
        conn.rollback()
        return f"An error occurred during assignment: {e}"
    finally:
        if 'conn' in locals() and conn:
            conn.close()

# Example of how you would correctly call the tool
if __name__ == '__main__':
    dummy_assignments = [
        {"order_id": 1001, "driver_id": 1, "estimated_time_minutes": 45},
        {"order_id": 1004, "driver_id": 1, "estimated_time_minutes": 60}
    ]
    
    # Correct way to invoke the tool with a dictionary mapping
    result = assign_orders_to_driver.invoke({"assignments": dummy_assignments})
    print(result)


# In your tools.py file, add the following code:
import math

@tool
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the distance in kilometers between two sets of geographical coordinates (latitude and longitude).
    
    Args:
        lat1: The latitude of the first point.
        lon1: The longitude of the first point.
        lat2: The latitude of the second point.
        lon2: The longitude of the second point.
        
    Returns:
        The distance in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dLon / 2) * math.sin(dLon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance