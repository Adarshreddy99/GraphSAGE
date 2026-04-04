import gradio as gr
import requests
import pandas as pd
import json
from loguru import logger

# Configuration
# Inside Docker, 127.0.0.1 is the local container loopback
API_URL = "http://127.0.0.1:8000/recommend"

def get_recommendations(query, k, mmr_lambda):
    """
    Calls the FastAPI backend to get GraphSAGE recommendations.
    """
    if not query.strip():
        return pd.DataFrame(columns=["Title", "Year", "Subject", "Similarity"])

    payload = {
        "query": query,
        "k": int(k),
        "lambda_param": float(mmr_lambda)
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        results = response.json()

        # Format into a DataFrame
        data = []
        for r in results:
            # Ensure Title starts with a Capital Letter (User Request)
            title = str(r['title']).capitalize()
            # If the title is all caps or weirdly formatted, capitalize will make it just first letter cap.
            # If it's already mixed case, we can use title() but capitalize() was specifically requested.
            
            data.append({
                "Title": title,
                "Year": r['year'],
                "Subject": r['subject'],
                "Similarity": f"{r['score']:.4f}"
            })
        
        return pd.DataFrame(data)

    except Exception as e:
        logger.error(f"UI Error: Could not connect to API at {API_URL}. Error: {str(e)}")
        return pd.DataFrame([{"Error": "Is the FastAPI server running on port 8000?"}])

# Build the Gradio Interface
with gr.Blocks(title="Research Paper Recommender") as demo:
    gr.Markdown("# 🎓 AI-Powered Research Paper Recommender")
    gr.Markdown("Discover **Related Research Papers** using a **3-hop Inductive Graph Neural Network.**")
    
    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Research Topic / Abstract Extract",
                placeholder="e.g., Deep learning for graph neural networks and node classification...",
                lines=3
            )
        with gr.Column(scale=1):
            k_slider = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Find Top-K Papers")
            mmr_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.7, step=0.1, 
                label="MMR"
            )
            search_button = gr.Button("🔍 Find Related Papers", variant="primary")

    output_table = gr.Dataframe(
        headers=["Title", "Year", "Subject", "Similarity"],
        datatype=["str", "number", "str", "str"],
        label="Recommended Citations"
    )

    # Event Handlers
    search_button.click(
        fn=get_recommendations,
        inputs=[query_input, k_slider, mmr_slider],
        outputs=output_table
    )
    
    # Allow Enter to trigger search
    query_input.submit(
        fn=get_recommendations,
        inputs=[query_input, k_slider, mmr_slider],
        outputs=output_table
    )

    gr.Markdown("---")
    gr.Markdown("Built with **PyTorch Geometric**, **FAISS (HNSW)**, and **FastAPI**.")

if __name__ == "__main__":
    # Inside Docker, 0.0.0.0 is MANDATORY to be accessible from the host
    demo.launch(
        server_name="0.0.0.0", 
        server_port=8080, 
        share=False
    )
