import os
import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    TextStreamer
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
import gradio as gr

# Initialize the model and components
def initialize_components():
    # 1. Initialize the HuggingFace model with quantization config
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # You may change to another model, Example [mistralai/Mistral-7B-v0.1 , deepseek-ai/DeepSeek-R1-Distill-Qwen-7B]
    model_id = "pankajmathur/orca_mini_3b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

    # Initialize the HuggingFace embedding model.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Example documents with metadata.
    documents = [
        Document(
            page_content=(
                "Once upon a time, in a magical kingdom filled with wonders, the people lived in harmony with nature. "
                "They were guided by a wise sage whose ancient wisdom shaped their traditions."
            ),
            metadata={"statement": "Statement 1", "chapter": "Chapter 1", "page": "Page 3"}
        ),
        Document(
            page_content=(
                "Legends spoke of hidden treasures and timeless traditions that celebrated the kingdom's resilient spirit. "
                "Festivals and folklore preserved these memories for generations."
            ),
            metadata={"statement": "Statement 2", "chapter": "Chapter 2", "page": "Page 2"}
        ),
    ]

    # Create a Chroma vector store from the documents.
    vectorstore = Chroma.from_documents(documents, embeddings)

    # Initialize the streamer to print tokens as they are generated.
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Create the HuggingFace pipeline for text generation.
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,  # Use sampling for diverse outputs
        streamer=streamer  # Streamer will print tokens automatically
    )

    # Wrap the HF pipeline in LangChain's HuggingFacePipeline LLM wrapper.
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
    return llm, vectorstore

# Initialize components once
llm, vectorstore = initialize_components()

def generate_response(user_query):
    # Retrieve the top 2 most similar documents from the vector store.
    retrieved_docs = vectorstore.similarity_search(user_query, k=2)
    # Combine the retrieved documents' content to form the context.
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Construct the prompt
    prompt = (
        f"Answer the following question using the provided context. Refine your output. "
        f"If the question is outside the context, simply say 'I don't know'.\n\n"
        f"Question: {user_query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    # Generate the response
    response_text = llm(prompt)

    # Prepare sources information
    sources = set()
    for doc in retrieved_docs:
        source_info = f"{doc.metadata.get('statement')}, {doc.metadata.get('chapter')}, {doc.metadata.get('page')}"
        sources.add(source_info)
    
    sources_text = "\n".join(sources) if sources else "No sources found"
    
    # Format the output
    output = f"""
**Response:**
{response_text}

**Sources:**
{sources_text}

**Context Used:**
{context}
"""
    return output

# Create Gradio interface
with gr.Blocks(title="LLM with Document Retrieval") as app:
    gr.Markdown("# LLM with Document Retrieval")
    gr.Markdown("Ask questions based on the provided documents")
    
    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(label="Enter your question", placeholder="Type your question here...")
            submit_btn = gr.Button("Submit")
        
        with gr.Column():
            output = gr.Textbox(label="Response", interactive=False)
    
    submit_btn.click(
        fn=generate_response,
        inputs=user_input,
        outputs=output
    )
    
    examples = gr.Examples(
        examples=[
            "What did the legends speak about?",
            "Who guided the people in the magical kingdom?",
            "What preserved the memories of the kingdom?"
        ],
        inputs=user_input
    )

if __name__ == "__main__":
    app.launch(share=False)
