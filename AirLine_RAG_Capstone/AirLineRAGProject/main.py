from src.chroma_utils import add_documents, load_pdfs_from_directory, query_collections
from src.Structured_Output_RAG import StructuredOutputRAG
from src.Agentic_RAG import AgenticRAG
from src.Multi_document_RAG import MultiDocumentRAG

import os

def main():
    print("Main Program")
    input_pdf_source = r"./data/handbook.pdf"
    multi_doc_directory = r"./data"
        
    k_val=3
    queries = [
                    #"Effects of Battle of Plassey",                    
                    # "on what basis divorce will be granted",
                    #"What are the effect on the benefits I receive if my probation is extended?",
                    #"There has been a demise in my family last night, and I need to attend the last rites.How should I inform the office, and will I be granted leave?",
                    #"What should I do if I notice suspected harassment with my female colleague?"
                    #"What are the effect on the benefits I receive if my probation is extended also There has been a demise in my family last night, and I need to attend the last rites.How should I inform the office, and will I be granted leave?"
                    #"Who was more powerful than French fleet and they helped in what?",
                    "What was the effect of international event"
              ]

    #TestSinglePDF(input_pdf_source)
    
    #Test_Simple_RAG(input_pdf_source,query,k,True,0,500)
    
    #Test_Conversational_RAG(input_pdf_source)

    #Test_Multi_Query_RAG(input_pdf_source,queries,fine_tune=False)

    Test_Multi_Document_RAG(multi_doc_directory,queries,fine_tune=True,k_val=k_val)

    #Test_Hierarchical_RAG(multi_doc_directory,queries,True)

    #Test_Structured_Output_RAG(multi_doc_directory,queries,True)

    #Test_Agentic_RAG(multi_doc_directory,queries,True)

def Test_Multi_Document_RAG(directory,queries, fine_tune=False,k_val=3):
    try:
        directory = os.path.normpath(directory)
        documents, ids, collections = load_pdfs_from_directory(directory)
        print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        if collections:
            rag = MultiDocumentRAG(max_history=5)
           
            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.multi_document_rag(query, collections, top_k=k_val, fine_tune=fine_tune)
                print(answer)
        else:
            print(f"No valid PDFs processed in {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")


def Test_Structured_Output_RAG(directory,queries, fine_tune=False):
    try:
        directory = os.path.normpath(directory)
        documents, ids, collections = load_pdfs_from_directory(directory)
        print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        if collections:
            rag = StructuredOutputRAG(max_history=5)
            
            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.structured_output_rag(query, collections, top_k=3, fine_tune=fine_tune)
                print(answer)
                print("-"*250)
        else:
            print(f"No valid PDFs processed in {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")



def Test_Agentic_RAG(directory, queries, fine_tune=False):
    try:
        directory = os.path.normpath(directory)
        documents, ids, collections = load_pdfs_from_directory(directory)
        print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        if collections:
            rag = AgenticRAG(max_history=5)
           
            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.agentic_rag(query, collections, top_k=3, fine_tune=fine_tune)
                print(answer)
                print("-"*200)
        else:
            print(f"No valid PDFs processed in {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")


if __name__ == "__main__":
    main()
