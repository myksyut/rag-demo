import argparse
import os
import sys
from operator import itemgetter

from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import AzureChatOpenAI
# LLM model
LLM_MODEL = "gpt-4o"

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', help='Query with RRF search')


# Retriever options
TOP_K = 3
MAX_DOCS_FOR_CONTEXT = 3
# .env
load_dotenv()

# Template
my_template_jp = """Please answer the [question] using only the following [information] in Japanese. If there is no [information] available to answer the question, do not force an answer.

Information: {context}

Question: {question}
Final answer:"""



def reciprocal_rank_fusion(results: list[list], k=60):

    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # for TEST (print reranked documentsand scores)
    print("Reranked documents: ", len(reranked_results))
    for doc in reranked_results:
        print('---')
        print('Docs: ', ' '.join(doc[0].page_content[:300].split()))
        print('RRF score: ', doc[1])

    # return only documents
    return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]



def query_generator(original_query: dict) -> list[str]:
    # original query
    query = original_query.get("query")

    # prompt for query generator
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("user", "Generate multiple search queries related to:  {original_query}. When creating queries, please refine or add closely related contextual information in Japanese, without significantly altering the original query's meaning"),
        ("user", "OUTPUT (3 queries):")
    ])

    # LLM model
    model = AzureChatOpenAI(
                api_version="2024-02-01",
                temperature=0,
                azure_deployment=LLM_MODEL
            )

    # query generator chain
    query_generator_chain = (
        prompt | model | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # gererate queries
    queries = query_generator_chain.invoke({"original_query": query})

    # add original query
    queries.insert(0, "0. " + query)

    # for TEST
    print('Generated queries:\n', '\n'.join(queries))

    return queries


def rrf_retriever(query: str) -> list[Document]:

    # Retriever
    retriever = AzureAISearchRetriever(
        content_key="chunk",
        top_k=TOP_K,
    )
    # RRF chain
    chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(query_generator)
        | retriever.map()
        | reciprocal_rank_fusion
    )

    # invoke
    result = chain.invoke({"query": query})

    return result



def query(query: str, retriever: BaseRetriever):
    # model
    model = AzureChatOpenAI(
        api_version="2024-02-01",
        temperature=0,
        model_name=LLM_MODEL,
        )

    # prompt
    prompt = PromptTemplate(
        template=my_template_jp,
        input_variables=["context", "question"],
    )

    # Query chain
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question")
        }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt | model | StrOutputParser(),
            "context": itemgetter("context"),
        }
    )

    # execute chain
    result = chain.invoke({"question": query})

    return result


# main
def main():
    # OpenAI API KEY
    if os.environ.get("AZURE_OPENAI_API_KEY") == "":
        print("`OPENAI_API_KEY` is not set", file=sys.stderr)
        sys.exit(1)

    # args
    args = parser.parse_args()

    # retriever
    retriever = RunnableLambda(rrf_retriever)

    # query
    if args.query:
        retriever = RunnableLambda(rrf_retriever)
        result = query(args.query, retriever)
    else:
        sys.exit(0)

    # print answer
    print('---\nAnswer:')
    print(result['response'])


if __name__ == '__main__':
    main()