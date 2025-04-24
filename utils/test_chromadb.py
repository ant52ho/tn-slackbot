# file for loading data into the chroma db
import asyncio
from similarity_search.dataloader import ChatDataLoader
from config import CHAT_DATA_PATH, CHROMA_HOST, CHROMA_PORT, OPENAI_API_KEY
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
import tiktoken
from tqdm import tqdm

class ChromaDBLoader:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.collection = None
        self.chroma_client = None
        self.embedding_function = None
        self.cdl = ChatDataLoader()
    
    async def initialize(self):
        """Initialize the ChromaDB client and collection"""
        self.chroma_client = await chromadb.AsyncHttpClient(host='localhost', port=CHROMA_PORT)
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-ada-002"
        )
        self.collection = await self.chroma_client.get_or_create_collection(
            self.collection_name,
            embedding_function=self.embedding_function
        )

    def split_by_tokens(self, text, model="text-embedding-ada-002", max_tokens=4000):
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [enc.decode(chunk) for chunk in chunks]

    async def load_data(self, chat_data_path):
        # load data
        df = self.cdl.preprocess_training_data(file_path=chat_data_path)
        print(df)   
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Uploading Chunks"):
            for i, chunk in enumerate(self.split_by_tokens(row['conversation'])):
                await self.collection.add(
                    ids=[f"{row['Channel ID']}:{row['Thread ID']}:{i}"],
                    documents=[chunk],
                    metadatas=[{
                        "channel_id": row['Channel ID'],
                        "thread_id": row['Thread ID'],
                        "chunk_index": i,
                    }]
                )
    async def delete_collection(self):
        await self.chroma_client.delete_collection(self.collection_name)
        collections = await self.chroma_client.list_collections()
        print("Remaining collections:", len(collections))

async def main():
    # data path for chat data
    chat_data_path = CHAT_DATA_PATH

    # collections desginate where you want to load the data
    collection_name = "messages3"


    # load to collection with collection name
    cl = ChromaDBLoader(collection_name)
    await cl.initialize()
    # await cl.load_data(chat_data_path)


    collections = await cl.chroma_client.list_collections()
    print("Collections:", collections)

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002"
    )

    collection = await cl.chroma_client.get_collection(
        collection_name,
        embedding_function=embedding_function
    )
    print("COLLECTION", collection)
    print(await collection.count())
    results = await collection.query(
            query_texts=["Leah: help me"],
            n_results=3
        )
    print("Results:", results)
    # await cl.delete_collection()


if __name__ == "__main__":
    asyncio.run(main())
