# [MODULE DOC]
# Example script: agentic RAG with lexical retrieval. Loads the m-ric/huggingface_doc dataset, keeps only
#   transformers docs, wraps them as LangChain Documents, splits them into ~500-char chunks (overlap 50),
#   indexes them with BM25 inside RetrieverTool and lets a CodeAgent (Qwen3-Next-80B-A3B-Thinking, max 4
#   steps, streaming) answer a question about forward vs backward pass speed; prints the final output.
# Inputs: none; depends on datasets, langchain, langchain_community (BM25, needs rank_bm25) and smolagents.
# Outputs: none (script); defines RetrieverTool and module globals knowledge_base, source_docs,
#   docs_processed, agent.
# Exceptions: dataset/network/model errors propagate.
# Side effects: at import, downloads the dataset, network calls to the LLM, console output.
# from huggingface_hub import login

# login()
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever


knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}) for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

from smolagents import Tool


class RetrieverTool(Tool):
    # [CLASS DOC]
    # Tool named "retriever" that does BM25 lexical search over a fixed list of LangChain documents.
    # Inputs: constructor docs (list of LangChain Document), extra kwargs forwarded to Tool. Class attributes
    #   define the tool name, description, one string input "query" and string output.
    # Key attributes: retriever (BM25Retriever returning the top 10 documents).
    # Exceptions: see methods.
    # Side effects: none beyond building the index.
    name = "retriever"
    description = "Uses lexical search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be lexically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        # [FUNCTION DOC]
        # Initializes the Tool base and builds a BM25Retriever over docs with k equal to 10.
        # Inputs: docs (list of Document), kwargs forwarded to Tool.
        # Outputs: none.
        # Exceptions: propagates retriever construction errors (e.g. missing rank_bm25).
        # Side effects: sets self.retriever.
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        # [FUNCTION DOC]
        # Runs the BM25 retrieval for the query and formats the hits.
        # Inputs: query (str).
        # Outputs: string starting with "Retrieved documents:" followed by each document's page_content under
        #   a "===== Document i =====" header (i from 0).
        # Exceptions: AssertionError if query is not a str.
        # Side effects: none.
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )


from smolagents import CodeAgent, InferenceClientModel


retriever_tool = RetrieverTool(docs_processed)
agent = CodeAgent(
    tools=[retriever_tool],
    model=InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking"),
    max_steps=4,
    verbosity_level=2,
    stream_outputs=True,
)

agent_output = agent.run("For a transformers model training, which is slower, the forward or the backward pass?")

print("Final output:")
print(agent_output)
