# pycon-colombia-2024
Repo for my PyCon Colombia 2024 talk called Embeddings is all you need: Artificial Intelligence for text, audio and images.
See all the info about the event, [clic here](https://2024.pycon.co/en) 

## Embeddings is all you need: Artificial Intelligence for text, audio and images.

Currently working on it! 
🎯⚒️⚙️💡

At their core, embeddings are a way to represent words, phrases, or even entire documents as vectors of real numbers in a continuous vector space. This representation captures the semantic meaning of the text, allowing similar words to have similar representations. Popular techniques to create embeddings include Word2Vec, GloVe (Global Vectors for Word Representation), and more recently, contextual embeddings like those from deep neural networks.

In this repo and talk we are going to explore how to use them to develop applications based on AI with the following 3 uses cases.

### 1. Chat with your own data. The PyCon Colombia 2024 ChatBot 🤖

This implementation is based on RAG (Retrieval Augmented Generation) architechture which uses LLMs, Embeddings, Vector Databases and Prompt Engineering to build a Chatbot for the PyCon Colombia 2024 as next:

<img src="assets/gifs/chatbot_app.gif" alt="Chatbot app" width="768" height="432">


### 2. Web Store that allows to find similar products based on a image 🔎

This implementation is a version on Reverse Image Search. Using ViT (Vision Transformer) models, Embeddings and Vector Databases to can search product into a web fashion store based on a image of a product as input and retrieve the most similar products.

<img src="assets/gifs/products_app.gif" alt="Products app" width="768" height="432">


### 3. Music Recommender Web App: Find songs similar to your favorite songs 🎵
This implementation is based on a Music recommender. Using PANNs (Pretrained Audio Neural Networks) models, Embeddings and Vector Databases to can search and recommend songs similar to your favorites songs. You enter a YouTube URL of a song and it fetch recommendations.

<img src="assets/gifs/songs_app.gif" alt="Songs app" width="768" height="432">