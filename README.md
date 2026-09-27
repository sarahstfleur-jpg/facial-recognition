# Cloud-Based Facial Recognition Pipeline (Google Colab Edition)

A complete facial recognition system built inside Google Colab that detects human faces, calculates high-dimensional face embeddings via deep learning models, and matches them using pgvector stored remotely on an Aiven cloud PostgreSQL database.

## Architectural Workflow
1. **Dependency Alignment**: Overwrites default cloud packages to prevent library compilation mismatches.
2. **Profile Registration**: Loops through Colab's cloud filesystem directory (stored-faces/), processes known images into 512-dimensional vectors, and registers them to the cloud.
3. **Target Identification**: Evaluates a query image, uses a Euclidean distance database search (<->), and pulls the single closest match.

---

## Google Colab File Setup

Before running your code cells, look at the left sidebar of Google Colab, click the Folder icon, and organize your files to look exactly like this:

```text
sample_data/               # Default Colab sample folder (Ignore this)
stored-faces/              # Right-click -> "New Folder" and name it exactly this
│   ├── sarah.jpg          # Upload your known profile photos here
│   └── john.png
└── image.jpg              # Upload the mystery target image you want to recognize here
```

---

## Step-by-Step Cell Execution Guide

### Step 1: Install Compatible Notebook Dependencies
Google Colab uses a modern python core environment by default. Because the imgbeddings library relies on an older Hugging Face cache architecture, you must run this installation cell to prevent an ImportError:

```bash
!pip install huggingface-hub==0.25.2 psycopg2-binary imgbeddings pillow numpy
```
*Note: Once this installation cell completes, go to the top menu of Google Colab and click Runtime -> Restart session (or use the keyboard shortcut Ctrl + M .). This wipes the old cache and registers your downgraded packages.*

---

### Step 2: Database Table Setup (Run Once)
Before uploading data, ensure your Aiven cloud database possesses the correct storage columns. Run a script or use an interface tool to create this SQL layout:

```sql
CREATE TABLE pictures (
    file_name TEXT PRIMARY KEY,
    embedding FLOAT[]
);
```

---

### Step 3: Populate Known Profiles (embedding.py logic)
Paste your database upload code into a notebook cell, update your SERVICE_URI connection string (ensuring no trailing > bracket exists at the end of the text string), and click play.

* **What it does**: It translates the photos in your stored-faces/ folder into numerical footprints and uploads them.
* **Expected Output**:  
  `Connecting to your cloud database...`  
  `Connected to database successfully!`  
  `Successfully registered: sarah.jpg`  
  `Storage complete! All reference fingerprints are safely saved on the Aiven cloud.`

---

### Step 4: Scan and Identify the Target Face (recognize_face.py logic)
Upload a query photograph named image.jpg directly into the base file repository of Google Colab. Execute your lookup script cell to search the database.

* **What it does**: It computes a temporary vector for your target image, asks the Aiven database for the closest statistical match, and displays the result using notebook-friendly graphics libraries.
* **Expected Output**:  
  `Opening image and calculating face fingerprint...`  
  `Querying database for the closest facial match...`  
  `Match found! Closest profile image is: sarah.jpg`  
* **Visual Result**: A clean inline cell output displaying the image matching the person identified.
