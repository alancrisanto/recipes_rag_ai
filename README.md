# Recipes Generator

This project is a **Recipe Recommendation Application** that leverages **Gradio**, **LangChain**, and **Pinecone** for advanced recipe search and recommendation capabilities. The application allows users to ask for recipe suggestions based on ingredients, dietary preferences, or specific instructions, and uses a conversational agent backed by a knowledge base for enhanced results.

## Features

- **Recipe Recommendations**: Users can ask for recipes based on specific ingredients, dietary needs, or preferences (e.g., "Low-calorie chicken recipe").
- **Interactive Chat Interface**: Users interact with the application via a chatbot-like interface powered by **Gradio**.
- **Knowledge-Based Retrieval**: Integrates with **LangChain** and **Pinecone** to enable context-based recipe retrieval.
- **Custom Recipe Suggestions**: The core logic is handled by a `get_recipe` function, which processes user inputs and retrieves relevant recipes from the database.

