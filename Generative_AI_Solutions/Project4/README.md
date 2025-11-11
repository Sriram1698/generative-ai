# Personalized Real Estate Agent - A HomeMatcher

## Project Overview
<b>HomeMatch</b> is an AI-driven real estate recommendation system that uses <b>Large Language Models (LLms), Langchain, OpenAI Embeddings </b> and <b>LanceDB</b> to intelligently match home buyers's preferences with the given property listings.
It combines <b>semantic understanding</b> and <b>personalized response</b> to enhance user experience, ensuring factual accuracy while tailoring descriptions to individual buyer preferences.

## Key Features
- `Synthetic Real Estate listing generation` using LLM models.
- `Vector Embedding Generation` using OpenAI Embeddings.
- `Storing Listings in DB` using LanceDB vector database.
- `Efficient Semantic Search` on LanceDB to fetch the semantically similar data from vector database.
- `Personalized Description Augmentation` using LLM to enhance the user experience and personalization.
- `Chat history tracking` to avoid repetitive or duplicate data generation.
- `Buyer Preference` via both structured and natual lanuages. 

## Components
1. <b>Real Estate Listing Generation</b>:
  - Uses `Langchain` with `ChatOpenAI` and a `PromptTemplate`.
  - Generates listings with realistic fields:
    ```json
    {
        "Neighborhood": "Willow Creek",
        "Price": 650000.0,
        "Bedrooms": 4,
        "Bathrooms": 3,
        "House Size": 2400.0,
        "Description": "",
        "Neighborhood Description": ""
    }
    ```
  - Ensures valid `JSON` format with schema representations for the response
  - Added chat history to prevent duplicate or repetitive data generation.
  
2. <b>Embedding Creation</b>:
  - Converts each listing into a numerical vector embedding using `OpenAIEmbeddings` from `Langchain`
  - The embeddings capture the semantic meaning of each listing.
  
3. <b>Vector Storage (LanceDB)</b>:
  - Uses `LanceDB` for fast and persistent vector storage and search
  - The Schema used in the project is as follows:
    ```python
    class RealEstateData(LanceModel):
        Neighborhood: str
        Price: float
        Bedrooms: int
        Bathrooms: int
        House_Size: float
        Description: str
        Neighborhood_Description: str
        Embedding: vector(embedding_dim)
    ```
  - LanceDB enables cosine similarity search to match the buyer's preferences.

4. <b>Buyer Preference Interface</b>:
  - Buyer prefernce can be:
    - Natural Language like the following
    ```python
    query = "A cozy three-bedroom house with spacious kitchen and friendly neighborhood"
    ```
    - Sturctured form answers like (number of bedrooms, neighborhood, etc)
  - The user preferences are converting into query embeddings and searched semantially against the listings in the DB.'

5. <b>Semantic Search</b>:
  - Retrieves top-k listings based on cosine similarity:
  ```python
  results = table.search(query_vector).metric("cosine").limit(k).to_pydantic(RealEstateData)
  ```

6. <b>Personalized Listing Augmentation</b>:
  - Augments retrieved listings using LLM based on buyer's preferences.
  - Maintains factual integrity like the number of rooms, price and neighborhood remain unchanged.


## How to Run

1. Install Dependencies:
  ```bash
  pip install langchain langchain-openai openai lancedb pydantic
  ```

2. Set OpenAI API Key:
  In notebook add the API Key in the following line
  ```python
  API_KEY = "" # API KEY
  ```

3. Run the Notebook:
  Run by clicking the `Run All` button at the top of the Notebook

4. Listings Generation:
  Modify the variable `n` in the Notebook to control how many listings to generate.

5. Modify Query:
  Modify the user/buyer preference query in the following line
  ```python
  query = "A cozy three-bedroom house with spacious kitchen and friendly neighborhood"
  ```
