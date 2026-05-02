# Socratic-OT — Deployment Guide

**Team:** Vidhyadhari Bandaru · Richie Ilavarapu

---

## Option A — Google Colab (for demos, grading, and development)

### One-time setup
1. Go to [drive.google.com](https://drive.google.com)
2. Upload the entire `Socratic_OT` folder to **My Drive** (top level)
3. Verify the structure:
   ```
   My Drive/
   └── Socratic_OT/
       ├── main.py
       ├── run_in_colab.ipynb
       ├── src/
       │   ├── knowledge_base.py
       │   ├── tutor.py
       │   ├── memory.py
       │   ├── vlm.py
       │   └── evaluation.py
       └── Data/
           ├── text_chunks/
           │   └── text_chunks_full 2.csv
           └── image_metadata/
               └── image_metadata.json
   ```

### Get a free Groq API key (required)
1. Go to [console.groq.com](https://console.groq.com) → sign up (free, no credit card)
2. Click **API Keys** → **Create API Key** → copy it

### Run the tutor in Colab
1. Open [colab.google.com](https://colab.google.com)
2. **File → Open notebook → Google Drive** → navigate to `Socratic_OT/run_in_colab.ipynb`
3. **Runtime → Change runtime type → T4 GPU** (needed for LLaVA image analysis; optional if you only need text)
4. Click the **🔑 Secrets** icon in the left sidebar
   - Add secret: `GROQ_API_KEY` = your key from above
   - Toggle **Notebook access** ON
5. Run **Cell 1** — installs packages (~2 min, once per session)
6. Run **Cell 2** — mounts Drive, loads API key
7. Run **Cell 3** — starts the tutor
   - First run: builds ChromaDB (~3 min, embeds all 997 chunks)
   - Subsequent runs: loads from cache (instant)
8. Click the **public Gradio link** that appears (e.g. `https://abc123.gradio.live`)

### After the first run
- ChromaDB is saved to Google Drive so subsequent runs skip the embed step
- Session memory is saved to `Socratic_OT/Data/session_memory/`
- To force a full rebuild: run **Cell 5** (`%run main.py --rebuild`)

### Run RAGAS evaluation (optional)
After using the tutor, run **Cell 4** (`%run main.py --eval`) to measure:
- Faithfulness (target ≥ 0.90)
- Answer Relevance (target ≥ 0.85)
- Context Recall (target ≥ 0.80)
- Context Precision (target ≥ 0.80)
- Socratic Purity (target: 5/5 transcripts leak-free)

---

## Option B — HuggingFace Spaces (professional, permanent, public URL)

### Step 1 — Create a HuggingFace account
Go to [huggingface.co](https://huggingface.co) → Sign up (free)

### Step 2 — Create a new Space
1. Click your profile icon → **New Space**
2. Fill in:
   - **Space name:** `socratic-ot` (or any name you like)
   - **License:** MIT
   - **SDK:** Gradio
   - **SDK version:** 5.20.1
   - **Hardware:** CPU Basic (free) — or GPU T4 Small ($0.05/hr) for LLaVA image analysis
3. Click **Create Space**

### Step 3 — Add your Groq API key as a Secret
1. In your Space, go to **Settings → Variables and secrets**
2. Click **New secret**
   - Name: `GROQ_API_KEY`
   - Value: your key from [console.groq.com](https://console.groq.com)
3. (Optional) Add `OPENAI_API_KEY` for GPT-4o vision fallback

### Step 4 — Upload your project files
You need to upload these files to your Space repository:

**Option 4a — via the HuggingFace web UI (easiest)**
1. In your Space, click the **Files** tab
2. Click **Add file → Upload files**
3. Upload ALL of these:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `src/__init__.py`
   - `src/knowledge_base.py`
   - `src/tutor.py`
   - `src/memory.py`
   - `src/vlm.py`
   - `src/evaluation.py`
   - `Data/text_chunks/text_chunks_full 2.csv`
   - `Data/image_metadata/image_metadata.json`
   - `Data/images/` (all 6 anatomy image files)
4. Click **Commit changes to main**

**Option 4b — via Git (for developers)**
```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/socratic-ot
cd socratic-ot

# Copy project files into the cloned repo
cp -r /path/to/Socratic_OT/* .

# Push to HuggingFace
git add .
git commit -m "Initial Socratic-OT deployment"
git push
```

### Step 5 — Wait for the build
- HuggingFace will install packages from `requirements.txt` (~3-5 min)
- Then run `app.py` which builds ChromaDB (~3 min on first launch)
- Your Space will show **Building...** → **Running**
- Click the **App** tab to see your live tutor!

### Your permanent URL will be:
```
https://huggingface.co/spaces/YOUR_USERNAME/socratic-ot
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not set` | Add it in Colab Secrets (🔑) or HF Spaces Secrets |
| `FileNotFoundError: text_chunks_full 2.csv` | Make sure the CSV file (with the space in the name) is uploaded |
| `ModuleNotFoundError: chromadb` | Re-run Cell 1 in Colab; or check HF Spaces build logs |
| `CUDA out of memory` | Go to Runtime → Change runtime type → T4 GPU (Colab) |
| Gradio link expired | Re-run Cell 3 in Colab |
| HF Space stuck on "Building" | Check the Build logs tab for errors |
| LLaVA not loading on HF Spaces CPU | Normal — LLaVA needs GPU. CPU runtime uses GPT-4o fallback (needs `OPENAI_API_KEY`) or text-only mode |
| ChromaDB rebuild needed | Colab: run Cell 5. Local: `python main.py --rebuild`. HF Spaces: delete the `/tmp/chroma_db` folder via the Files tab and restart |

---

## Folder structure expected

```
Socratic_OT/
├── app.py                    ← HuggingFace Spaces entry point
├── main.py                   ← Colab / local entry point
├── requirements.txt          ← Python dependencies
├── README.md                 ← HF Spaces config + project description
├── DEPLOYMENT_GUIDE.md       ← This file
├── run_in_colab.ipynb        ← Colab notebook
├── src/
│   ├── __init__.py
│   ├── knowledge_base.py     ← ChromaDB + all-MiniLM-L6-v2
│   ├── tutor.py              ← LangGraph Socratic engine
│   ├── memory.py             ← Session memory + weak topic tracking
│   ├── vlm.py                ← LLaVA / GPT-4o vision module
│   └── evaluation.py         ← RAGAS + purity audit
└── Data/
    ├── text_chunks/
    │   └── text_chunks_full 2.csv   ← 997 chunks, 28 chapters
    ├── image_metadata/
    │   └── image_metadata.json
    └── images/
        └── (anatomy diagram files)
```
