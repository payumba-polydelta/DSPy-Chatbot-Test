import streamlit as st
import dspy
import time
from DSPyPineconeRM import PineconeRM

from helper import load_gemini_model

# Load models and configure DSPy settings
gemini_flash = load_gemini_model()
pinecone_retriever = PineconeRM
dspy.settings.configure(lm=gemini_flash, rm=pinecone_retriever)


class GenerateAnswerWithContext(dspy.Signature):
    """Generate an answer based on the provided context and question."""

    context = dspy.InputField(desc="Helpful information for answering the question.")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="Answer derived from the context")


class RAG(dspy.Module):
    """Retrieval-Augmented Generation (RAG) module for question answering."""

    def __init__(self, num_passages = 5):
        super().__init__()
        self.retrieve = pinecone_retriever(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerWithContext)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        context = [passage.long_text for passage in context]
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(
            context=context, 
            answer=prediction.answer
        )
        
        