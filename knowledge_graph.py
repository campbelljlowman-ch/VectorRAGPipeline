import os
import asyncio

from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "asdfasdf"

graph = Neo4jGraph(refresh_schema=False)
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
llm_transformer = LLMGraphTransformer(llm=llm)

async def run_graph_transform():
    # text = """
    # Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
    # She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
    # Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
    # She was, in 1906, the first woman to become a professor at the University of Paris.
    # """
    loader = PyPDFLoader("/Users/clowman/Documents/Code/VectorRAGPipeline/data/A Dale Harris_9760914.zip.d/A Dale Harris Cv - converted.pdf", mode="single")
    documents = loader.load()
    # print(f"docuemnt: {documents}")
    # documents = [Document(page_content=text)]
    graph_documents = await llm_transformer.aconvert_to_graph_documents(documents)
    print(f"Nodes:{graph_documents[0].nodes}")
    print(f"Relationships:{graph_documents[0].relationships}")
    graph.add_graph_documents(graph_documents)
    # graph.add_graph_documents(graph_documents, baseEntityLabel=True)



if __name__ == "__main__":
    asyncio.run(run_graph_transform())
# if __name__ == "__main__":
#     run_graph_transform()