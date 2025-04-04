from typing import List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain.load import dumps, loads
import pickle
import os

# Setting states
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    outcome: Literal["vectorstore", "reject"] = Field(
        ...,
        description="Given a user question choose to route it to a vectorstore or reject.",
    )

def setEnvs(path: str):
    """
    Load environment variables from a file.
    Each non-empty line in the file should be in the format: KEY=value
    """
    with open(path) as f:
        for line in f:
            if line.strip():
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
