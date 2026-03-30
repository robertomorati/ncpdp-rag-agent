# ncpdp-rag-agent

A simple Python RAG (Retrieval-Augmented Generation) assistant that answers questions about the NCPDP using a PDF and audio transcription.

### 1. Clone the repository

```bash
git clone git@github.com:robertomorati/ncpdp-rag-agent.git
cd ncpdp-rag-agent
```

## Setupp

### 2. Create virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install system dependency (Mac)

```bash
brew install ffmpeg
```

### 5. Configure environment variables

Create a `.env` file:

```env
GEMINI_API_KEY=<YOUR_API_KEY>
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_COLLECTION=ncpdp_guide
CHROMA_PATH=./chroma_db
TOP_K=3
```

### 6. Data used

```bash
data/raw/pdf/
data/raw/ncpdp_audio.mp3
```


## Run

### 1. Extract text from PDF

```bash
python -m app.ingest_pdf
```

### 2. Transcribe audio

```bash
python -m app.transcribe_audio
```

### 3. Build vector database

```bash
python -m app.vectordb
```


### 4. Run the assistant

```bash
python -m app.main
```


## Usage

```
NCPDP RAG Agent
Type your question or 'exit' to quit.

Question: What is field 134-UK?

Answer:
Not found in provided database.

------------------------------------------------------------

Question: What is field 460-ET?

Answer:
Field 460-ET is:
*   **NCPDP Field Name:** QUANTITY PRESCRIBED
*   **Usage:** R/W (Required when)
*   **Source:** C (Submitted Claim or the Processor’s response to the Submitted Claim)
*   **Situation:** Required when received as part of the original claim from the provider or as part of the Processor’s response to the Submitted Claim.

------------------------------------------------------------

Question: What is the difference between 505-F5 and 518-FI?

Answer:
The difference between 505-F5 and 518-FI is as follows:

*   **505-F5** represents the "PATIENT PAY AMOUNT."
*   **518-FI** represents the "AMOUNT OF COPAY."

**Relationship:**
518-FI (AMOUNT OF COPAY) is required if the Patient Pay Amount (505-F5) includes copay as part of the patient's financial responsibility.

------------------------------------------------------------
```

---

## Improvements

* Needs improve chunk quality and retrieval precision
* Add docker/tiltfile
* Add https://streamlit.io/
* Add ELABORATE_QUERY primpt