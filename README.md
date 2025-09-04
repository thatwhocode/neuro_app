**Custom Cascading NLP Pipeline for Data Verification**

**Goal of the project:**
*The ultimate goal of this project is to create, an application that will automate the process of validation and extraction of named entities throughout special text fields in a database also called "fabulas."*

**As for now, the application consists of two microservices that enable rapid filtration and detailed recognition, working locally to ensure the consistent safety of personal data.**

**ML Architecture**

*The system is built on a cascaded model architecture to optimize for both speed and accuracy.*

    *Two models are used here:*

        A Fine-tuned XLM-RoBERTa model for Named Entity Recognition (NER).

        A Classification model that identifies if a text contains any mentions of specific weapon names or models.

**Metrics and Key Values**

To ensure the model's performance on real-world data, all key metrics were calculated on a portion of the original labeled dataset that was never shown to the model during training.

    Pre-training: The first step involved pre-training a base model to predict the next word in Ukrainian Wikipedia sentences about weapons, giving it a strong domain understanding.

    NER Accuracy: The F1-score for NER accuracy is 88%, based on a dataset of 1500 manually labeled texts.

    Classifier Score: The classifier achieved an accuracy of 97%. It  worth noting that while this number is high it is very questionable as the model was trained on a labeled well balanced dataset of  texts mixed with an equal amount of random text data that did not contain any weapon mentions.

**Confidentiality**

*Due to the intended use within a law enforcement structure, the hardware is a critical consideration. All processing must be done locally to maintain data confidentiality.*

    GPU RTX 3060 TI

    CPU: Intel Core i5 11400

    RAM: 16 GB

**Technological Stack**

    Python with FastAPI for API development and deployment.

    Hugging Face Transformers, scikit-learn, and SpaCy for model development.

    Docker Compose for application deployment.

    Nginx as a reverse proxy and SSL certificate handler.

    Build Instructions:

    Clone the Repository: Clone the project repository from GitHub.

    Generate SSL Certificates: Navigate to the nginx/ssl directory and run the openssl command to generate your self-signed certificates for secure local communication.

    Build and Run: From the project's root directory, use docker compose to build and run all the services.

docker compose up --build -d

    Access the API: The application will be accessible at https://localhost:443/docs once the containers are running.