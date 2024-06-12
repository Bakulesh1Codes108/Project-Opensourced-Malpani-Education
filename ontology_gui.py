import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import language_tool_python
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().reshape(1, -1)

def evaluate_response(user_response, reference_texts):
    user_embedding = get_embedding(user_response)
    reference_embeddings = np.vstack([get_embedding(text) for text in reference_texts])
    similarities = cosine_similarity(user_embedding, reference_embeddings)
    return similarities

def track_ontology_coverage(user_responses, subgraph):
    covered_nodes = set()
    for response in user_responses:
        for node in subgraph.nodes(data=True):
            if node[1]['label'].lower() in response.lower():
                covered_nodes.add(node[0])
    return covered_nodes

def calculate_metrics(user_responses, reference_texts, subgraph):
    accuracy_scores = []
    for response in user_responses:
        similarities = evaluate_response(response, reference_texts)
        accuracy_scores.append(similarities.max())
    
    accuracy_percentage = np.mean(accuracy_scores) * 100
    covered_nodes = track_ontology_coverage(user_responses, subgraph)
    coverage_percentage = (len(covered_nodes) / len(subgraph.nodes)) * 100
    
    return accuracy_percentage, coverage_percentage

def identify_mistakes(user_response, reference_texts):
    mistakes = []
    user_words = set(user_response.lower().split())
    for ref in reference_texts:
        ref_words = set(ref.lower().split())
        missing_words = ref_words - user_words
        if missing_words:
            mistakes.append(f"Missing concepts: {', '.join(missing_words)}")
    return mistakes

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Grammar checker
tool = language_tool_python.LanguageTool('en-US')

def check_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    mistakes = [match.ruleIssueType for match in matches]
    return corrected_text, mistakes

def evaluate_pronouns(user_response):
    correct_pronouns = ["he", "she", "they", "it"]
    pronouns_used = [word for word in user_response.lower().split() if word in correct_pronouns]
    pronoun_accuracy = len(pronouns_used) / len(user_response.split()) * 100
    return pronoun_accuracy, pronouns_used

def draw_graph(graph, title):
    pos = nx.kamada_kawai_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=False, node_color='skyblue', edge_color='black', node_size=300, font_size=10, alpha=0.6, width=0.5)
    for key, value in pos.items():
        x, y = value[0], value[1]
        plt.text(x, y, s=graph.nodes[key]['label'], bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'), horizontalalignment='center', fontsize=8)
    plt.title(title)
    return plt

class OntologyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ontology-Based Knowledge Evaluation")
        self.root.geometry("1200x800")

        self.question_label = tk.Label(root, text="Enter your question:", font=("Helvetica", 14))
        self.question_label.pack(pady=10)

        self.question_textbox = tk.Entry(root, width=100, font=("Helvetica", 12))
        self.question_textbox.pack(pady=10)

        self.response_label = tk.Label(root, text="Enter user's response:", font=("Helvetica", 14))
        self.response_label.pack(pady=10)

        self.response_textbox = tk.Entry(root, width=100, font=("Helvetica", 12))
        self.response_textbox.pack(pady=10)

        self.submit_button = tk.Button(root, text="Submit", command=self.handle_submit, font=("Helvetica", 14), bg="blue", fg="white")
        self.submit_button.pack(pady=10)

        self.output_area = scrolledtext.ScrolledText(root, width=100, height=10, font=("Helvetica", 12))
        self.output_area.pack(pady=20)

        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(pady=20)

        self.save_button = tk.Button(root, text="Save Session", command=self.save_session, font=("Helvetica", 14), bg="green", fg="white")
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.load_button = tk.Button(root, text="Load Session", command=self.load_session, font=("Helvetica", 14), bg="orange", fg="white")
        self.load_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(root, text="Clear Session", command=self.clear_session, font=("Helvetica", 14), bg="red", fg="white")
        self.clear_button.pack(side=tk.LEFT, padx=10)

        self.questions = []
        self.responses = []
        self.scores = []

    def handle_submit(self):
        question = self.question_textbox.get()
        user_response = self.response_textbox.get()

        self.questions.append(question)
        self.responses.append(user_response)

        # Evaluate the response
        similarities = evaluate_response(user_response, reference_texts)
        accuracy = similarities.max()
        covered_nodes = track_ontology_coverage([user_response], subgraph)
        coverage_percentage = (len(covered_nodes) / len(subgraph.nodes)) * 100

        if accuracy >= 0.9:
            feedback = f"User response accuracy: {accuracy * 100:.2f}% - Excellent (above 90%)"
        elif accuracy >= 0.8:
            feedback = f"User response accuracy: {accuracy * 100:.2f}% - Good (above 80%)"
        else:
            feedback = f"User response accuracy: {accuracy * 100:.2f}% - Needs Improvement (below 80%)"

        mistakes = identify_mistakes(user_response, reference_texts)
        sentiment_result = sentiment_analyzer(user_response)[0]
        sentiment = f"Sentiment: {sentiment_result['label']} with score {sentiment_result['score']:.2f}"
        sentiment_score = sentiment_result['score'] * (1 if sentiment_result['label'] == 'POSITIVE' else -1)

        corrected_text, grammar_mistakes = check_grammar(user_response)
        grammar_feedback = f"Grammar Check: {'; '.join(grammar_mistakes)}"
        grammar_score = (len(user_response.split()) - len(grammar_mistakes)) / len(user_response.split()) * 100

        pronoun_accuracy, pronouns_used = evaluate_pronouns(user_response)
        pronoun_feedback = f"Pronouns Used: {', '.join(pronouns_used)} (Accuracy: {pronoun_accuracy:.2f}%)"

        total_score = (accuracy * 0.5) + (sentiment_score * 0.2) + (grammar_score * 0.2) + (pronoun_accuracy * 0.1)
        self.scores.append(total_score)

        result = f"Question: {question}\n"
        result += f"User Response: {user_response}\n"
        result += f"Corrected Response: {corrected_text}\n"
        result += feedback + '\n'
        result += sentiment + '\n'
        result += grammar_feedback + '\n'
        result += pronoun_feedback + '\n'
        result += f"Current ontology coverage: {coverage_percentage:.2f}%\n"
        result += f"Total Score: {total_score:.2f}%\n"
        if total_score > 50:
            result += "Outcome: The response is on the right track (above 50%).\n"
        else:
            result += "Outcome: The response needs improvement (below 50%).\n"
        if mistakes:
            result += "Mistakes:\n" + "\n".join(mistakes) + '\n'
        result += '\n'

        self.output_area.insert(tk.END, result)

        # Draw the ontology graph
        fig = draw_graph(G, "Ontology Graph")
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def save_session(self):
        session_data = {
            "questions": self.questions,
            "responses": self.responses,
            "scores": self.scores
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.DataFrame(session_data)
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Save Session", "Session saved successfully!")

    def load_session(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)
            self.questions = df["questions"].tolist()
            self.responses = df["responses"].tolist()
            self.scores = df["scores"].tolist()
            self.output_area.insert(tk.END, "Session loaded successfully!\n")

    def clear_session(self):
        self.questions = []
        self.responses = []
        self.scores = []
        self.output_area.delete(1.0, tk.END)
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        messagebox.showinfo("Clear Session", "Session cleared successfully!")

# Load ontology data
df_data = pd.read_csv('ontology_data.csv')
df_relationships = pd.read_csv('ontology_relationships.csv')

# Create the ontology graph
G = nx.Graph()
for index, row in df_data.iterrows():
    G.add_node(row['id'], label=row['label'], type=row['type'])
for index, row in df_relationships.iterrows():
    G.add_edge(row['source'], row['target'], relationship=row['relationship'])

# Define subgraph for a specific chapter
chapter_id = 1  # ID for "Microbes in Human Welfare"
subgraph = G.subgraph(nx.descendants(G, chapter_id) | {chapter_id})

# Reference texts for evaluation
reference_texts = [
    "Microbes are used in the production of antibiotics.",
    "Microbes are important in biotechnology.",
    "Microbes help in sewage treatment."
]

# Create and run the app
root = tk.Tk()
app = OntologyApp(root)
root.mainloop()
