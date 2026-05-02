Below is the project we are building:

Project 3 - Socratic-OT Multimodal AI Tutor (Tutoring/Learning Assistant -
Rehabilitation Science)
Background:
Occupational Therapy (OT) is a healthcare profession focused on helping people achieve independence in
daily life. To succeed, students must master Gross Anatomy and Neuroscience. These are "bottleneck"
courses due to their high content density and heavy reliance on complex 3D visual relationships.
The Problem: Students lack 1:1 support outside of class. Standard AI (like ChatGPT) often gives
answers away, preventing students from building the "clinical reasoning" needed for the NBCOT
certification exam. The Solution: You will build a Socratic AI Tutor that uses Retrieval-Augmented
Generation (RAG) to guide students using only faculty-approved textbooks and diagrams.
What is "Socratic" Tutoring?
In this project,
"Socratic" means the bot follows a "Tutor-not-Teller" philosophy.
●
The Goal: Lead the student to the answer by asking questions based on the textbook context.
The Constraint: The bot is strictly forbidden from providing a direct definition or answer in the
first two turns. It must use the retrieved data to ask a "leading" question that triggers the student's
memory.
●
Dataset: Input Sources
Gather and structure data into a "Knowledge Base.
"
●
●
●
Textual Data (PDF/HTML): Use open-source medical texts.
○
Source: OpenStax Anatomy & Physiology 2e
Visual Data (Images): Curate 20–30 anatomical diagrams with labels.
○
Source: AnatomyTOOL or MedPix®
Format: Text should be stored in a Vector Database (e.g., ChromaDB, FAISS). Images must have
associated metadata (JSON) describing the structures shown.

Please note that the model/architecture details given here are only for guidance, you are free
to use any model/architecture of your choice to complete the tasks.
Task 1: Content Retrieval & Knowledge Masking
●
●
●
Mechanism: Develop a RAG pipeline to retrieve sections from OT textbooks (e.g.,
neuroscience/anatomy).
The "Mask": The agent identifies the target answer but "masks" it from the student.
Logic: The LLM generates progressive hints based on the hidden data without revealing the
specific term, forcing student discovery.
Task 1: Adaptive Conversation Strategy
●
●
●
Phase 1 (Rapport): Open with context-aware chitchat (e.g.,
"Ready for the lab exam?") to build
engagement.
Phase 2 (Tutoring): Pivot to the masked content using a Socratic loop.
Agentic Control: A Manager Agent monitors the dialogue state, ensuring the transition from
social to pedagogical is seamless.
Task 3: Synthesis & Assessment
●
●
Evaluation: Move beyond multiple-choice. The agent prompts the student to apply the concept to
a clinical scenario (e.g.,
"Explain how this nerve lesion impacts the body").
Outcome: The agent compares the student's reasoning against the gold-standard textbook data
and provides a mastery summary.
Task 4: Multimodal Diagram Tutoring:
●
Use a Vision-Language Model (VLM) like GPT-4o or LLaVA. When a student uploads an image
and asks,
"What is this muscle?"The bot must identify it and ask the student about its function or
insertion point. Use a combination of high-reasoning API calls (OpenAI/Claude) and local Open
Source LLMs (Llama 3/Mistral) to manage costs and data privacy.
Task 5: Interactive Memory:
●
Implement a session-based memory where the bot remembers past student mistakes and
proactively revisits those topics. (tutoring modules- about 15mins per session)
Evaluation Metrics:
●
Groundedness (Quantitative): Use the RAGAS framework to measure Faithfulness. If the bot
provides info not found in the provided textbook, it fails the "hallucination" test.
●
Socratic Purity (Qualitative): You must provide 5 conversation transcripts. The bot must not
"leak" the final answer before the 3rd turn.
●
●
Multimodal Performance: The bot must correctly identify anatomical structures in a "blind test"
set of diagrams.
Generalizability: Demonstrate that your code can work for a different subject (e.g., Physics)
simply by swapping the Vector Database.
Bonus:
You can implement one or more of the below features for bonus points.
●
Personalization: A dashboard showing the student's "weak spots" based on chat history.
●
Accessibility: Integrating "Text-to-Speech" or "Speech-to-Text" to support diverse learning needs.
References:
●
●
●
●
●
●
●
MS Articulate 360: Linear, static branching. Why is your AI more flexible?
Boodlebox: A general-purpose tool. How does your OT-specific grounding make it safer for
healthcare?
Harvard Learning Lab (Tutor-not-Teller): Research their "System 2 Thinking" AI experiments
which force models to reason before responding.
Khanmigo (Khan Academy): The industry benchmark for AI tutoring. Study their technical
blogs on using "System Prompts" to prevent direct answers.
RAGAS Framework: The standard for evaluating Retrieval-Augmented Generation. Use this to
measure Faithfulness (no hallucinations) and Answer Relevance.
○
Resource: RAGAS Documentation
SocraticLM Paper: Review "SocraticLM: Exploring Socratic Personalized Teaching with
LLMs" (NeurIPS) for strategies on fine-tuning LLMs for guided questioning.
Multimodal VLM Research: Explore documentation for GPT-4o (Vision) or LLaV A to
understand how to prompt models for spatial anatomical reasoning.