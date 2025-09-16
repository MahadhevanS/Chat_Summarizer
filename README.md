# Chat_Summarizer

AI-powered chat summarizer that condenses messy, multi-turn conversations into clean summaries.  

## Overview

This project fine-tunes a Seq2Seq model (BART) to take raw user chat logs—often noisy, with irrelevant chatter and follow-ups—and generate structured, concise summaries focused on civic issues, complaints, hazards, etc.

---

## Features

- Handles noise: irrelevant messages, off-topic chatter.  
- Preserves multi-turn context and follow-up details.  
- Generates readable summaries focused on the problem, not the distractions.  
- Easy-to-use scripts for training and testing.

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/MahadhevanS/Chat_Summarizer.git
cd Chat_Summarizer

# 2. (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # on Linux/macOS
venv\Scripts\activate      # on Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Use train_summarizer.py to train/fine-tune the summarization model.
python train_summarizer.py

# Use test_summarizer.py to load the saved model and generate summaries for new chat logs.
python test_summarizer.py

