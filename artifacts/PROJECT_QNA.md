# multimodal-finrag — Project Q&A
### Explained like you're 10 years old 🎒

---

## 1. What is this project?

Imagine you have a giant book — like a company's yearly report with 369 pages. You want to ask it questions like "How much money did they make?" but you don't want to read all 369 pages yourself.

This project is like a **super smart robot librarian**. You give it the book, it reads everything, remembers it, and then you can ask it any question. It will find the right page and give you the exact answer.

It works for financial documents like:
- Company annual reports (like Infosys)
- Quarterly earnings updates (like Tesla Q1 2026)

---

## 2. What is RAG and why use it here?

**RAG = Retrieval-Augmented Generation**

Think of it like an open-book exam vs a closed-book exam.

- **Closed-book (bad):** The AI answers from memory. But it might remember wrong things or old numbers. Like asking your friend "what was Tesla's revenue last quarter?" — they might guess wrong.

- **Open-book (RAG, good):** The AI first finds the relevant pages from your document, then reads those pages to answer. It's always correct because it's reading the actual book.

**Why important for finance?** Revenue, profits, employee count — these change every quarter. RAG always reads the latest uploaded document, so it never gives outdated or made-up numbers.

---

## 3. Walk me through the ingestion pipeline.

"Ingestion" means the robot librarian reading and memorizing your book. Here's what happens step by step:

```
You upload a PDF
      ↓
Save it to Amazon's cloud storage (S3) — like Google Drive
      ↓
Read all the text from the PDF (PyMuPDF)
— like scanning every page
      ↓
Cut it into small pieces called "chunks"
— like cutting a book into 800 sticky notes
      ↓
Convert each sticky note into a list of numbers (embeddings)
— numbers represent the "meaning" of the text
      ↓
Save all the numbers to a file on disk (index)
— like making a very organized index at the back of a book
```

**Smart trick:** At first we made 11,327 sticky notes (one per paragraph). Way too many! Changed to one sticky note per page → only 800 notes → 10x faster.

---

## 4. What is the retrieval strategy?

When you ask a question, the system needs to find the right sticky notes. It uses **two methods** and combines them:

**Method 1 — BM25 (keyword search):**
Like using Ctrl+F on your computer. If you search "22,387" it finds the exact page with that number. Great for exact numbers and names.

**Method 2 — Vector search (meaning search):**
Like finding a page even if it uses different words. If you ask "how profitable is Tesla?" it finds pages about "operating margin" and "net income" even if you didn't use those exact words.

**Then combine them:**
Both methods give you a list of "probably relevant" pages. We merge both lists using a formula called **RRF** (see next question).

**Then double-check:**
A smarter (but slower) AI reads your question and the top candidates together and gives a final score. This is called **reranking**.

---

## 5. What is RRF (Reciprocal Rank Fusion)?

Imagine two judges each ranking 10 contestants. Judge 1 says contestant A is #1. Judge 2 also says contestant A is #1. Clearly contestant A is the best!

RRF is a formula that combines two "best of" lists into one final list:

```
Score = 1/(60 + rank from list 1) + 1/(60 + rank from list 2)
```

A page ranked #1 by both keyword search AND meaning search gets the highest combined score. The number 60 is a "safety buffer" so that being ranked #1 vs #2 doesn't make a huge difference — overall ranking matters more than tiny differences.

---

## 6. What is the cross-encoder reranker and why use it?

Think of it like a talent show with two rounds:

**Round 1 (fast, rough):** 1000 people audition quickly. Judges pick top 20. (BM25 + vector search)

**Round 2 (slow, careful):** Those 20 perform properly in front of judges who pay full attention. Judges pick the best 4. (Cross-encoder reranker)

The cross-encoder reads your question AND the page together at the same time — like a judge who really listens. It's much more accurate but too slow to use on all 800 pages. So we only use it on the top 20 candidates from Round 1.

**Score fix:** The reranker gives raw numbers that can be negative (-728). We run them through a "sigmoid" formula that squishes any number into a nice 0–100% range. So -728 becomes something like 6%, which makes sense to display.

---

## 7. What LLM and embeddings are used?

| What | Which one | Why |
|------|-----------|-----|
| The brain that answers questions | Amazon Nova Lite | Free to use on AWS, works immediately |
| The "meaning converter" (embeddings) | Amazon Titan | Converts text → numbers |
| The double-checker (reranker) | MiniLM cross-encoder | Small, fast, runs on your laptop |

**Why not use ChatGPT?** This project uses AWS Bedrock — Amazon's AI service. We tried Claude (Anthropic's AI) but it was blocked because of payment setup issues. Amazon Nova Lite worked right away for free.

**Nova vs Claude:** They speak slightly different "languages". Like how British English and American English are similar but different ("lift" vs "elevator"). We had to write translation code to support both.

---

## 8. How is async indexing implemented?

"Async" means doing things in the background while you keep doing other things.

Imagine ordering pizza:
- **Without async:** You stand at the door waiting until the pizza arrives. Can't do anything else.
- **With async:** You order pizza, go watch TV. Pizza guy rings the bell when done.

Our system works the same way:

1. You upload a PDF → server says "Got it! Here's your order number: `abc123`" (instant, 2 seconds)
2. Server starts reading and indexing the PDF in the background (takes ~4 minutes)
3. Your browser checks every 5 seconds: "Is order `abc123` ready?"
4. When done, it shows "914 chunks indexed ✓"

This way the website doesn't freeze while waiting.

---

## 9. What was the biggest performance challenge and how did you fix it?

**Problem:** Uploading a 369-page PDF took over 10 minutes. Sometimes it never finished.

**Investigation — found 4 bugs one by one:**

🐛 **Bug 1:** Creating 11,327 sticky notes (one per paragraph). Way too many.
✅ Fix: One sticky note per page → 800 notes total.

🐛 **Bug 2:** Sending sticky notes to AWS one at a time for "meaning conversion". Each call takes 1.4 seconds. 11,327 × 1.4s = **4+ hours**!
✅ Fix: Send 5 sticky notes to AWS at the same time. 5x faster.

🐛 **Bug 3:** Converting every page into a photo even though we weren't using photos.
✅ Fix: Skip photo extraction entirely.

🐛 **Bug 4:** Sending 20 requests to AWS at the same time → AWS said "Too many requests! Slow down!"
✅ Fix: Max 5 at a time with a tiny pause between groups.

**Final result:** 369 pages indexed in ~4 minutes. Still slow because AWS free tier has limits — like a road with a speed limit.

---

## 10. What is LoRA finetuning used for?

Imagine you have a really smart student who knows everything about general English. But you need them to become an expert at reading financial documents and spotting important words like company names, money amounts, and dates.

**Full training** = Make the student forget everything and re-learn from scratch. Very expensive, takes a long time.

**LoRA** = Give the student a small "finance cheat sheet" (just 1% extra knowledge). They keep all their general knowledge but add new finance-specific skills. Much cheaper and faster!

We use this to teach the AI to find:
- **Company names** → "Infosys", "Tesla"
- **Money amounts** → "₹1,36,592 crore", "$22,387 million"
- **Dates** → "FY2025", "Q1 2026"
- **Percentages** → "21.1%", "+51%"

**Current status:** The cheat sheet hasn't been written yet (model not trained). This feature shows "unavailable" in the UI.

---

## 11. What does the Entities tab in the UI do?

It's like a highlighter pen that automatically highlights important financial words in any text you paste.

You paste: *"Infosys reported revenue of ₹1,36,592 crore in FY2025"*

It highlights:
- 🔵 **Infosys** → Company name
- 🟢 **₹1,36,592 crore** → Money amount
- 🟡 **FY2025** → Date

This is called **Named Entity Recognition (NER)**. Useful for quickly scanning documents without reading everything.

Currently not working — needs the LoRA model to be trained first.

---

## 12. How is AWS integrated?

AWS (Amazon Web Services) is like a giant set of tools you rent from Amazon. We use 3 tools:

| AWS Tool | What it does | Real-world analogy |
|----------|-------------|-------------------|
| **S3** | Stores your uploaded PDFs | Like Google Drive |
| **Bedrock** | Runs the AI brain (Nova Lite) and meaning converter (Titan) | Like renting a supercomputer |
| **Lambda** | Runs indexing code without a server | Like a vending machine — only works when you press a button |

To use AWS, you need a "key" (like a password). The key is passed as environment variables — like secret codes typed into the terminal before starting the server.

---

## 13. What are the API endpoints?

An API endpoint is like a button on a vending machine. Each button does one thing.

| Button | What it does |
|--------|-------------|
| `GET /health` | "Is the machine on?" → Yes/No |
| `POST /ingest` | "Here's a PDF, please read it" → Returns a job ticket |
| `GET /ingest/status/{id}` | "Is my job done yet?" → Indexing / Done / Failed |
| `POST /query` | "Answer this question" → Answer + sources |
| `POST /entities` | "Highlight the important words" → List of entities |

The UI (website) talks to these buttons behind the scenes when you click things.

---

## 14. What would you improve with more time?

1. **Turn charts back on** — Right now charts in PDFs are ignored. Want to re-enable AI reading charts and describing what they show (e.g. "Revenue grew 50% from Q1 to Q3").

2. **Train the highlighter** — The entity highlighter (LoRA model) needs to be trained on financial text before it works.

3. **Ask AWS for a faster lane** — AWS limits how many requests we can make per second (like a speed limit). Can request a higher limit.

4. **Show answers word by word** — Right now the full answer appears at once after 7 seconds. Could show it word-by-word as it's generated (like ChatGPT does).

5. **Search one document at a time** — Right now all uploaded documents are mixed together. Should be able to say "only search the Tesla PDF".

6. **Remember jobs after restart** — If the server restarts, all job statuses are forgotten. Should save them to a database.

7. **Add passwords** — Right now anyone who knows the URL can upload and query. Should add login/authentication for real use.
