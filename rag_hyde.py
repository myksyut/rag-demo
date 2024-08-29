import argparse
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain.retrievers import RePhraseQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever

# OpenAI embedding model
LLM_MODEL = "gpt-4o"

# Retriever settings
TOP_K = 5

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', help='Query with RRF search')

load_dotenv()

# プロンプトテンプレート
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# HyDEプロンプトテンプレート
hyde_prompt_template = """ \
以下の質問の回答を書いてください。
質問: {question}
回答: """

# Documentsを整形する関数
def doc_to_str(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

# RAGチャットボットを実行
def chat_with_bot(question: str):

    # LLM
    chat_model = AzureChatOpenAI(
                api_version="2024-02-01",
                temperature=0,
                azure_deployment=LLM_MODEL
            )

    # Vector Retriever
    retriever = AzureAISearchRetriever(
        content_key="chunk",
        top_k=TOP_K,
    )

    # HyDE Prompt
    hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template)

    # HyDE retriever
    rephrase_retriever = RePhraseQueryRetriever.from_llm(
        retriever = retriever,
        llm = chat_model,
        prompt = hyde_prompt,
    )

    # RAG Chain
    rag_chain = (
        {"context": rephrase_retriever | doc_to_str, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # プロンプトテンプレートに基づいて応答を生成
    response = rag_chain.invoke(question)

    return response


if __name__ == "__main__":

    if os.environ.get("AZURE_OPENAI_API_KEY") == "":
        print("`OPENAI_API_KEY` is not set", file=sys.stderr)
        sys.exit(1)

    # args
    args = parser.parse_args()
    
    if args.query:
        response = chat_with_bot(args.query)
    else:
        sys.exit(0)

    # print answer
    print('---\nAnswer:')
    print(response)
    

    # チャットセッションの開始
    