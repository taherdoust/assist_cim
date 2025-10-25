from agent_cim_assist import query_agent, query_agent_with_thinking, execute_sql

# Natural language queries
# query_agent("How many buildings are there?")
# query_agent("How many projects are there?")
query_agent("How many census zones are there?")


# query_agent("Find all the buildings that their centroid is within 1 meters of the centroid of the building with building_id '259f59e2-20c4-45d4-88b9-298022fd9c7f'")


# Debug mode (see thinking)
# query_agent_with_thinking("Find all the buildings that their centroid is within 100 meters of the centroid of the building with building_id '259f59e2-20c4-45d4-88b9-298022fd9c7f'")

# Direct SQL
# execute_sql("SELECT COUNT(*) FROM cim_wizard_building;")