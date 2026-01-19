# Import the Natural Language Toolkit library
import nltk 

# Download the 'punkt' tokenizer models from NLTK 
nltk.download("punkt") 

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

# Initialize an empty dictionary to store summaries
summaries = {}  

# Define a function to generate a three-sentence summary
def three_sentence_summary(text):
    # Return the first three sentences of the text
    return "\n".join(sent_tokenize(text)[:3])

# Generate a baseline summary and store it in the dictionary
summaries["baseline"] = three_sentence_summary(sample_set)


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

# GPT2 summary
# Set the random seed for reproducibility
set_seed(42)

# Initialize a text generation pipeline with the GPT-2 XL model
pipe = pipeline("text-generation", model="gpt2-xl")

# Create a query for the GPT-2 model 
query = sample_set + "\nTl;DR:\n"

# Generate a summary using the GPT-2 model
pipe_out = pipe(query, max_new_tokens=1000, cleanup_tokenization_spaces=True)

summaries["gpt2"] = "\n".join(sent_tokenize(pipe_out[0]['generated_text'][len(query):]))