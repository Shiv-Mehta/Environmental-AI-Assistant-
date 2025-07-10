import gradio as gr
from transformers import pipeline
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
import os
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

# Authenticate Hugging Face token for image generation
login(token=os.environ.get("HF_TOKEN"))

# Load SpaCy model
spacy.load("en_core_web_sm")

# Initialize models
classifier = None
image_pipe = None
ner_pipe = None
mask_filler = None

def download_models():
    global classifier, image_pipe, ner_pipe, mask_filler

    print("Downloading sentence classification model...")
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    print("Downloading image generation model...")
    image_pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        use_auth_token=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    print("Downloading NER model...")
    ner_pipe = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        grouped_entities=True
    )

    print("Downloading mask filling model...")
    mask_filler = pipeline(
        "fill-mask",
        model="bert-large-uncased"
    )
    return "All models downloaded successfully!"

def classify_text(text):
    environment_categories = {
        'POSITIVE': 'Environmentally Positive',
        'NEGATIVE': 'Environmentally Negative',
        'NEUTRAL': 'Environmentally Neutral',
        'POLICY': 'Environmental Policy',
        'SCIENCE': 'Environmental Science'
    }

    result = classifier(text)[0]
    label = result['label']
    score = result['score']

    if label in ['LABEL_0', 'NEGATIVE']:
        return f"Category: {environment_categories['NEGATIVE']}\nConfidence: {score:.2f}"
    elif label in ['LABEL_1', 'POSITIVE']:
        return f"Category: {environment_categories['POSITIVE']}\nConfidence: {score:.2f}"
    else:
        return f"Category: {environment_categories['NEUTRAL']}\nConfidence: {score:.2f}"

def generate_environment_image(prompt):
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
    image = image_pipe(
        f"environmental {prompt}, high quality, detailed, nature, realistic",
        generator=generator,
        num_inference_steps=50
    ).images[0]
    return image

def create_ner_graph(text):
    ner_results = ner_pipe(text)
    G = nx.Graph()

    for entity in ner_results:
        entity_text = entity['word']
        entity_type = entity['entity_group']
        G.add_node(entity_text, type=entity_type)

    for i in range(len(ner_results)-1):
        node1 = ner_results[i]['word']
        node2 = ner_results[i+1]['word']
        G.add_edge(node1, node2)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    colors = []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'PER':
            colors.append('red')
        elif G.nodes[node]['type'] == 'ORG':
            colors.append('blue')
        elif G.nodes[node]['type'] == 'LOC':
            colors.append('green')
        else:
            colors.append('yellow')

    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=2000, font_size=12)
    plt.title("Environmental NER Graph")
    plt.axis("off")
    plt.savefig("ner_graph.png")
    plt.close()
    return "ner_graph.png"

def fill_environment_mask(text_with_mask):
    results = mask_filler(text_with_mask)
    top_result = results[0]
    return {
        "filled_text": text_with_mask.replace("[MASK]", top_result['token_str']),
        "options": [r['token_str'] for r in results]
    }

# Gradio Interface
with gr.Blocks(title="Environmental AI Assistant") as demo:
    gr.Markdown("# 🌍 Environmental AI Assistant")
    gr.Markdown("Analyze and generate environmental content using AI models")

    with gr.Tab("Sentence Classification"):
        gr.Markdown("### Classify text into environmental categories")
        text_input = gr.Textbox(label="Enter environmental text")
        classify_btn = gr.Button("Classify")
        classification_output = gr.Textbox(label="Classification Result")
        classify_btn.click(classify_text, inputs=text_input, outputs=classification_output)

    with gr.Tab("Image Generation"):
        gr.Markdown("### Generate environmental images from text")
        prompt_input = gr.Textbox(label="Describe the environmental scene")
        generate_btn = gr.Button("Generate Image")
        image_output = gr.Image(label="Generated Image")
        generate_btn.click(generate_environment_image, inputs=prompt_input, outputs=image_output)

    with gr.Tab("NER Graph"):
        gr.Markdown("### Extract entities and visualize relationships")
        ner_text_input = gr.Textbox(label="Input text with environmental content")
        ner_btn = gr.Button("Extract Entities")
        ner_graph_output = gr.Image(label="Entity Relationship Graph")
        ner_btn.click(create_ner_graph, inputs=ner_text_input, outputs=ner_graph_output)

    with gr.Tab("Fill Mask"):
        gr.Markdown("### Complete environmental sentences with AI")
        mask_examples = [
            "Reducing [MASK] emissions is critical for climate change.",
            "The [MASK] is home to many endangered species.",
            "We should recycle more [MASK] to help the environment."
        ]
        mask_input = gr.Textbox(label="Enter sentence with [MASK]", value=mask_examples[0])
        fill_btn = gr.Button("Fill Mask")
        filled_output = gr.Textbox(label="Completed Sentence")
        options_output = gr.HighlightedText(label="Alternative Options")
        fill_btn.click(lambda x: fill_environment_mask(x), inputs=mask_input, outputs=[filled_output, options_output])
        gr.Examples(examples=mask_examples, inputs=mask_input)

download_status = download_models()
print(download_status)

demo.launch()
