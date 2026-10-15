# [MODULE DOC]
# Example script: text-to-SQL agent. Creates an in-memory SQLite "receipts" table (receipt_id, customer_name,
#   price, tip) with four rows, prints its column description, defines the sql_engine tool and asks a
#   CodeAgent (Llama-3.1-8B-Instruct via InferenceClientModel) which customer had the most expensive receipt.
# Inputs: none; depends on sqlalchemy and smolagents, HF inference credentials.
# Outputs: none (script); module globals engine, receipts, table_description, agent.
# Exceptions: model/network errors propagate.
# Side effects: at import, creates the in-memory database, prints, network calls.
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    inspect,
    text,
)


engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# create city SQL table
table_name = "receipts"
receipts = Table(
    table_name,
    metadata_obj,
    Column("receipt_id", Integer, primary_key=True),
    Column("customer_name", String(16), primary_key=True),
    Column("price", Float),
    Column("tip", Float),
)
metadata_obj.create_all(engine)

rows = [
    {"receipt_id": 1, "customer_name": "Alan Payne", "price": 12.06, "tip": 1.20},
    {"receipt_id": 2, "customer_name": "Alex Mason", "price": 23.86, "tip": 0.24},
    {"receipt_id": 3, "customer_name": "Woodrow Wilson", "price": 53.43, "tip": 5.43},
    {"receipt_id": 4, "customer_name": "Margaret James", "price": 21.11, "tip": 1.00},
]
for row in rows:
    stmt = insert(receipts).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

inspector = inspect(engine)
columns_info = [(col["name"], col["type"]) for col in inspector.get_columns("receipts")]

table_description = "Columns:\n" + "\n".join([f"  - {name}: {col_type}" for name, col_type in columns_info])
print(table_description)

from smolagents import tool


@tool
def sql_engine(query: str) -> str:
    # [FUNCTION DOC]
    # Tool: executes an arbitrary SQL string against the module-level engine and returns the rows as text.
    # Inputs: query (str, raw SQL; executed without sanitization).
    # Outputs: concatenation of a newline plus the string form of each row (empty string when there are no
    #   rows).
    # Exceptions: SQLAlchemy errors (syntax errors, non-row statements) propagate.
    # Side effects: reads (or could modify) the in-memory database; statements are not committed.
    """
    Allows you to perform SQL queries on the table. Returns a string representation of the result.
    The table is named 'receipts'. Its description is as follows:
        Columns:
        - receipt_id: INTEGER
        - customer_name: VARCHAR(16)
        - price: FLOAT
        - tip: FLOAT

    Args:
        query: The query to perform. This should be correct SQL.
    """
    output = ""
    with engine.connect() as con:
        rows = con.execute(text(query))
        for row in rows:
            output += "\n" + str(row)
    return output


from smolagents import CodeAgent, InferenceClientModel


agent = CodeAgent(
    tools=[sql_engine],
    model=InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"),
)
agent.run("Can you give me the name of the client who got the most expensive receipt?")
