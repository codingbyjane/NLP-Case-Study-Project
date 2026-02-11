# Import the Natural Language Toolkit library
import nltk 

# Download the 'punkt' tokenizer models from NLTK 
nltk.download("punkt")

# nltk.download("punkt_tab") -> does not exist, so we only download "punkt" which is used for sentence tokenization

# Import the sentence tokenizer from NLTK 
from nltk.tokenize import sent_tokenize

# Import the pipeline and set_seed functions from the Huggingface transformers library
from transformers import pipeline, set_seed

# Import PlainTextParser from the sumy package
from sumy.parsers.plaintext import PlaintextParser  

# Import Tokenizer from the sumy package
from sumy.nlp.tokenizers import Tokenizer 

# Import TextRankSummarizer from the sumy package 
from sumy.summarizers.text_rank import TextRankSummarizer  

# Import functions to load datasets from the datasets library
# Note: the load_metric function is deprecated, so we use the evaluate library instead

from datasets import load_dataset 
import evaluate

# Import the pandas library to work with tables
import pandas as pd


# Load XSum (another summarization dataset)
dataset = load_dataset('xsum')

# Display the first example from the training set
sample_set = dataset['train'][0]['document'][:2000] # “Take the first 2000 characters of the text of the first article in the training set and store it in sample_text.”

# Display the keys of the first example in the training set to understand the structure of the dataset
print(dataset["train"][0].keys())

# Initialize an empty dictionary to store summaries
summaries = {}  


# Baseline summary
# Define a function to generate a three-sentence summary
def three_sentence_summary(text):
    # Return the first three sentences of the text
    return "\n".join(sent_tokenize(text)[:3])

# Generate a baseline summary and store it in the dictionary
summaries["baseline"] = three_sentence_summary(sample_set)

print(f"Baseline summary: {summaries["baseline"]}\n") # Display the three-sentence baseline summary to verify that it is working correctly



# Sumy TextRank summary
# Use PlainTextParser to parse the sample text
parser = PlaintextParser.from_string(sample_set, Tokenizer("english"))

# Initialize the TextRank summarizer  
summarizer = TextRankSummarizer()  

# Initialize an empty list to store summary sentences
summary_sentences = []

for sentence in summarizer(parser.document, 5):
    # Append each summary sentence to the list
    summary_sentences.append(str(sentence))

summaries["sumy"] = "\n".join(summary_sentences)

print(f"Sumy TextRank summary: {summaries["sumy"]}\n") # Display the TextRank summary to verify that it is working correctly



# GPT2 summary
# Set the random seed for reproducibility
set_seed(42)

# Initialize a text generation pipeline with the GPT-2 XL model model and its tokenizer
gpt2_pipe = pipeline("text-generation", model="gpt2-xl")

# Create a query for the GPT-2 model 
query = sample_set + "\nTl;DR:\n"

# Generate a summary using the GPT-2 model
pipe_out = gpt2_pipe(query, max_new_tokens=300) # removed cleanup_tokenization_spaces=True because it is not a valid argument for the text-generation pipeline

generated_text = pipe_out[0]['generated_text'][len(query):].strip()  # Extract the generated summary text by removing the original query from the output
generated_text = " ".join(generated_text.split())  # Clean up extra whitespace in the generated text

summaries["gpt2"] = "\n".join(sent_tokenize(generated_text))

print(f"GPT-2 summary: {summaries["gpt2"]}\n") # Display the GPT-2 summary to verify that it is working correctly



# DeepSeek summary
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Set the random seed for reproducibility
set_seed(42)

deepseek_pipe = pipeline('text-generation', model=model_name, trust_remote_code=True, truncation=True) # trust_remote_code is required for some models that have custom code on the Huggingface Hub
query = sample_set + "\nTl;DR:\n"

pipe_out = deepseek_pipe(query, max_new_tokens=300) # removed cleanup_tokenization_spaces=True because it is not a valid argument for the text-generation pipeline

generated_text = pipe_out[0]['generated_text'][len(query):].strip()  # Extract the generated summary text by removing the original query from the output
generated_text = " ".join(generated_text.split())  # Clean up extra whitespace in the generated text

summaries["deepseek"] = "\n".join(sent_tokenize(generated_text))

print(f"DeepSeek summary: {summaries["deepseek"]}\n") # Display the DeepSeek summary to verify that it is working correctly



# BART summary
bart_pipe = pipeline("summarization", model="facebook/bart-large-cnn") # fine-tuned on CNN/DailyMail summarization
pipe_out = bart_pipe(sample_set, max_length=150, min_length=40, do_sample=False)

summaries["bart"] = '\n'.join(sent_tokenize(pipe_out[0]["summary_text"]))

print(f"BART summary: {summaries["bart"]}\n") # Display the BART summary to verify that it is working correctly



# Results evaluation with ROUGE metric
# Load the ROUGE metric from the evaluate library
rouge_metric = evaluate.load("rouge") # the load_metric method is deprecated, so we use evaluate.load instead

reference = dataset["train"][0]["summary"]  # The reference summary for the first article in the training set

records = []  # Initialize an empty list to store evaluation records
model_names = [] # List to store model names for indexing the DataFrame

rouge_versions = ["rouge1", "rouge2", "rougeL", "rougeLsum"]  # List of ROUGE metric versions to evaluate

for model_name in summaries: # Iterate over the generated summaries

    score = rouge_metric.compute(predictions=[summaries[model_name]], references=[reference])  # Compute the ROUGE scores for the current summary against the reference summary
    
    # Extract the mid f-measure scores for each ROUGE version and store them in a dictionary

    rouge_dict = {rn: score[rn] for rn in rouge_versions}  # Create a dictionary of ROUGE scores for the current model

    records.append(rouge_dict)  # Append the scores to the records list
    model_names.append(model_name)  # Append the model name to the list for indexing

ROUGE_df = pd.DataFrame.from_records(records, index=model_names)  # Create a DataFrame from the records for better visualization

print(ROUGE_df)

ROUGE_df.to_csv("rouge_results.csv") # Save the ROUGE evaluation results to a CSV file