# Import the Natural Language Toolkit library
import nltk 

# Download the 'punkt' tokenizer models from NLTK 
nltk.download("punkt")
nltk.download("punkt_tab")

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

# Baseline summary
# Define a function to generate a three-sentence summary
def three_sentence_summary(text):
    # Return the first three sentences of the text
    return "\n".join(sent_tokenize(text)[:3])

# Generate a baseline summary and store it in the dictionary
summaries["baseline"] = three_sentence_summary(sample_set)
#print(summaries["baseline"])


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

#print(summaries["gpt2"])


# DeepSeek summary
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Set the random seed for reproducibility
set_seed(42)

pipe = pipeline('text-generation', model=model_name)
query = sample_set + "\nTl;DR:\n"

pipe_out = pipe(query, max_bew_tokens=1000, cleanup_tokenization_spaces=True)

summaries["deepseek"] = "\n".join(sent_tokenize(pipe_out[0]['generated_text'][len(query):]))


# BART summary
pipe = pipeline("summarization", model="facebook/bart-large-cnn") # fine-tuned on CNN/DailyMail summarization
pipe_out = pipe(sample_set)

summaries["bart"] = '\n'.join(sent_tokenize(pipe_out[0["summary_text"]]))


# Results evaluation with ROUGE metric
# Load the ROUGE metric from the evaluate library
rouge_metric = evaluate.load("rouge") # the load_metric method is deprecated, so we use evaluate.load instead

reference = dataset["train"][0]["highlight"]  # The reference summary for the first article in the training set

records = []  # Initialize an empty list to store evaluation records
rouge_versions = ["rouge1", "rouge2", "rougeL", "rougeLsum"]  # List of ROUGE metric versions to evaluate

for model_name in summaries: # Iterate over the generated summaries
    rouge_metric.add(prediction=summaries[model_name], reference=reference)  # Add the generated summary and reference summary to the metric
    score = rouge_metric.compute()  # Compute the ROUGE scores

    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_versions)  # Extract the mid f-measure scores for each ROUGE version
    records.append(rouge_dict)  # Append the scores to the records list

ROUGE_df = pd.DataFrame.from_records(records, index=summaries.keys())  # Create a DataFrame from the records for better visualization
# print(ROUGE_df)

ROUGE_df.to_csv("rouge_results.csv")