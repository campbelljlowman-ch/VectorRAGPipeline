import os
from data_loaders.s3_data_loader import S3BatchDownloader
from langchain import hub
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict



if not os.environ.get("GOOGLE_API_KEY"):
  raise ValueError("GOOGLE_API_KEY environment variable not set.")


llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql://postgres:postgres@localhost:5432/hcm",
)

S3_BUCKET = "pinpointmigration2"
data_loader = S3BatchDownloader(bucket=S3_BUCKET, dest_dir="data/", prefix=None, chunk_size=5, state_path=".s3_progress.json")


def run_pipeline():
    while not data_loader.done():
        pdfs_pulled = data_loader.get_next_chunk()
        # print(f"Downloaded {len(pdfs_pulled)} files from S3 bucket '{S3_BUCKET}' to 'data/' directory.")

        for pdf_pulled in pdfs_pulled:
            pdf_path = pdf_pulled[0]
            # s3url = pdf_pulled[1]

            loader = PyPDFLoader(pdf_path, mode="single")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
            all_splits = text_splitter.split_documents(docs)
            vector_store.add_documents(documents=all_splits)
            resumes_processed += 1
        print(f"Processed {resumes_processed} resumes so far.")



# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
# prompt = hub.pull("rlm/rag-prompt")
prompt = PromptTemplate.from_template("""You are an assistant for searching a database of resumes. Use the following snippets of resumes to answer the question. If you don't know the answer, just say that you don't know. Be concise but make sure to include the names of all relevant people.
Question: {question} 
Context: {context} 
Answer:""")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=20)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def run_chat():
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    while True:
        print("Running the Chat RAG Pipeline...")
        question = input("Enter your question: ")

        response = graph.invoke({"question": question})
        print(response["answer"])

if __name__ == "__main__":
    run_chat()
    # run_pipeline()