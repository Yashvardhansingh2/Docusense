import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=settings.gemini_api_key,
    temperature=0.1,
)

prompt = ChatPromptTemplate.from_template("""
You are a precise document assistant. Answer ONLY using the context below.
If the answer is not found in the context, say exactly:
"I couldn't find this information in the document."
Always mention which section you used.

Context from document:
{context}

Question: {question}

Answer:""")

chain = prompt | llm | StrOutputParser()


async def generate_answer(question: str, chunks: list[dict]) -> dict:
    context = "\n\n---\n\n".join(
        [f"[Section {c['chunk_index']}]\n{c['content']}" for c in chunks]
    )
    start = time.time()
    answer = await chain.ainvoke({"context": context, "question": question})
    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "answer": answer,
        "sources": [c["chunk_index"] for c in chunks],
        "top_score": chunks[0]["score"] if chunks else 0,
        "response_time_ms": elapsed_ms,
    }
