import chromadb
chroma_client = chromadb.HttpClient(host='localhost', port=9000)
chroma_client.heartbeat()