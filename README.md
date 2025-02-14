# RAG-MODEL
An end-to-end guide for extracting, embedding, and querying text from online sources like Wikipedia using Gemini 1.5 for answers.
# Requirements
Requests
This library allows us to make HTTP requests in Python, which is essential for retrieving online data, such as extracting text from websites (e.g., Wikipedia articles).

BeautifulSoup4
A powerful library for web scraping, BeautifulSoup4 is used here to parse and extract text from HTML, which is particularly helpful for structuring text from online sources.

Gemini
The genai library enables interaction with Gemini API for tasks like generating text and embeddings or performing language-based tasks using models such as Gemini 1.5 or Gemini 2.0.

pandas
pandas is a versatile data manipulation library that allows for structured data storage and management, making it easy to organize and export data (e.g., to Excel).

NumPy and SciPy
These libraries provide efficient mathematical functions. NumPy is used for numerical operations, while SciPy includes functions for calculating cosine similarity, which is helpful for comparing text embeddings.

spaCy
spaCy is a natural language processing (NLP) library that allows for keyword extraction, entity recognition, and other linguistic processing. Here, it helps us extract key phrases from chunks of text.

LangChain
This library supports the implementation of language model applications. It includes tools like RecursiveCharacterTextSplitter, which enables us to split text into manageable chunks while preserving context.

openpyxl
openpyxl is used to write data into Excel files, allowing us to save embeddings or other structured data for later use.

After installing these libraries, you are ready to set up your data processing pipeline for the RAG model.

```
pip3 install requests beautifulsoup4 google-generativeai pandas numpy scipy spacy langchain openpyxl
```
