# AI-Powered-MCQ-Generator-And-Test-System

[MCQs Generator.webm](https://github.com/user-attachments/assets/9a3e5ec4-02c6-41d1-94b7-6ead09054f33)

# 📚 AI-Powered MCQ Generator
# Project Description
This project is an AI-powered Multiple-Choice Question (MCQ) Generator built using Streamlit, Langchain, and Google Gemini API. It allows users to upload a document (TXT, PDF, or DOCX), specify the number of MCQs to generate from its content, and then take a test based on these generated questions. After completing the test, users receive a detailed summary of their performance and all the correct answers for review.

# Features
Document Upload: Supports uploading .txt, .pdf, and .docx files.

# AI-Powered MCQ Generation: Generates multiple-choice questions directly from the uploaded document's content using Google Gemini via Langchain.

Customizable MCQ Count: Users can specify the desired number of MCQs (1-20).

Interactive Test System: Presents generated MCQs without answers, allowing users to select their choices.

Test Submission & Validation: Ensures all questions are answered before submission.

Performance Summary: Provides a table summarizing correct/incorrect answers after test submission.

Correct Answers Reveal: Displays all correct answers for review after the test.

Professional GUI: Intuitive and clean user interface built with Streamlit.

# Technologies Used
Python: The core programming language.

Streamlit: For building the interactive web application GUI.

Langchain: Framework for developing applications powered by language models.

Google Gemini API: The large language model used for generating MCQs.

PyPDFLoader: For extracting text from PDF documents.

python-docx: (Optional) For extracting text from DOCX documents.

# Setup and Installation
To get this project up and running on your local machine, follow these steps:

Clone the Repository (if applicable):

# If this project is in a repository, clone it first
# git clone <repository_url>
# cd <project_directory>

Create a Virtual Environment (Recommended):

python -m venv venv
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

#Install Dependencies:

pip install streamlit langchain langchain-google-genai pypdf python-docx

Set Your Google API Key:
The application requires a Google API Key to access the Gemini model. It is highly recommended to set this as an environment variable for security.

Get your API Key: Obtain your API key from Google AI Studio.

Set Environment Variable:

Linux/macOS:

export GOOGLE_API_KEY="YOUR_API_KEY"

Windows (Command Prompt):

set GOOGLE_API_KEY="YOUR_API_KEY"

Windows (PowerShell):

$env:GOOGLE_API_KEY="YOUR_API_KEY"

Alternatively (Not Recommended for Production): You can directly hardcode the API key in the mcq_generator.py file by uncommenting and modifying this line:

# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

Replace "YOUR_API_KEY" with your actual key.

How to Run
Once you have completed the setup, you can run the Streamlit application:

streamlit run mcq_generator.py

This command will open the application in your default web browser.

Usage
Upload Document: On the left sidebar, use the file uploader to select your .txt, .pdf, or .docx file.

Specify Number of MCQs: Enter the desired number of multiple-choice questions (between 1 and 20) in the input field.

Generate MCQs: Click the "Generate MCQs" button. The application will extract text, generate questions, and transition to the test interface.

Take the Test: Select an option for each generated MCQ.

Submit Test: Once all questions are answered, click the "Submit Test" button.

View Results: The application will display a summary of your performance and all the correct answers.

Start New Test: Click the "Start New Test" button in the sidebar to clear the current test and begin a new one.

Contact
For any inquiries or feedback, please contact:

Name: Hafiz Hassan Abdullah

Email: rajahassan38201@gmail.com

Phone: +92 302 3536363
