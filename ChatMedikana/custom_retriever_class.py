from langchain.schema import BaseRetriever, Document
from typing import List, Callable

class CountryFilteredRetriever(BaseRetriever):
    """
    A retriever that filters documents based on the country extracted from the query.
    """
    base_retriever: BaseRetriever
    extract_country_fn: Callable[[str], str]

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Extract country from query
        countries = self.extract_country_fn(query)

        # Retrieve a large pool
        docs = self.base_retriever.get_relevant_documents(query)

        # If country detected, filter manually
        if countries:
            country_norms = [c.lower() for c in countries]
            filtered = [
                d for d in docs
                if any(c.lower() in country_norms for c in d.metadata.get("countries", []))
            ]
            if filtered:  # fallback if something matches
                return filtered

        return docs  # fallback to unfiltered

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
