# from haystack.document_stores import PineconeDocumentStore
# from haystack.nodes import PreProcessor, DensePassageRetriever
# from haystack import Document

# document_store = PineconeDocumentStore(
#     api_key='8b1f9cc1-4214-458e-8f54-2ddacba4f5c9',
#     index='semasenti',
#     environment='us-east1-gcp',
#     similarity="dot_product",
#     embedding_dim=768
# )

# processor = PreProcessor(
#     clean_empty_lines=True,
#     clean_whitespace=True,
#     clean_header_footer=True,
#     split_by="word",
#     split_length=200,
#     split_respect_sentence_boundary=True,
#     split_overlap=3
# )

# retriever = DensePassageRetriever(
#     document_store=document_store,
#     query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
#     passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
#     use_gpu=True
# )