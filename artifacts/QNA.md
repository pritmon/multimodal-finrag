# FinRAG — Sample Q&A Test Cases

Verified questions and answers tested against ingested financial documents.

---

## Tesla Q1 2026 Update (`TSLA-Q1-2026-Update.pdf`)

### Basic

| Question | Expected Answer |
|----------|----------------|
| What is Tesla's total revenue in Q1 2026? | **$22,387 million** |
| What is Tesla's GAAP net income in Q1 2026? | **$477 million** |
| What is Tesla's free cash flow in Q1 2026? | **$1,444 million** |
| What is Tesla's operating cash flow in Q1 2026? | **$3,937 million** |
| How many total vehicles did Tesla deliver in Q1 2026? | **358,023** |

### Intermediate

| Question | Expected Answer |
|----------|----------------|
| How did FSD subscriptions grow from Q1 2025 to Q1 2026? | 0.85M → 1.28M, **+51% YoY** |
| What is Tesla's GAAP gross margin in Q1 2026? | **21.1%** (up 478 bp YoY) |
| Which quarter had the highest free cash flow in the last 5 quarters? | **Q3 2025 at $3,990M** |
| What is Tesla's cash and investments balance at end of Q1 2026? | **$44,743 million** |
| How many Supercharger stations does Tesla have as of Q1 2026? | **8,463** |

### Complex / Multi-step

| Question | Expected Answer |
|----------|----------------|
| What is capex as a % of revenue in Q1 2026? | $2,493M / $22,387M = **11.1%** |
| How did gross profit change from Q1 2025 to Q1 2026 in absolute and % terms? | +$1,567M, **+50% YoY** |
| What is the trend in operating margin over 5 quarters? | 2.1% → 4.1% → 5.8% → 5.7% → 4.2% (peaked Q3-2025) |
| What is Tesla's battery pack constraint and how are they addressing it? | Battery pack capacity is the limiting factor on vehicle production ramp. Addressing via LFP cells in Nevada, cathode material and lithium refining in Texas. |
| Where is Tesla launching unsupervised Robotaxi rides? | Austin (ramping), **Dallas and Houston** (launched April 2026); Phoenix, Miami, Orlando, Tampa, Las Vegas (preparations underway) |

### Strategic

| Question | Expected Answer |
|----------|----------------|
| What is Tesla's AI compute strategy? | Cortex 1 (>100k H100e) + Cortex 2 (>130k H100e) in Texas; partnering with SpaceX for largest chip fab; AI5 inference processor tape-out completed April 2026 |
| What products are on schedule for volume production in 2026? | **Cybercab, Tesla Semi, Megapack 3**; Optimus first-gen lines being installed |
| How did paid Robotaxi miles trend in Q1 2026? | Nearly **doubled sequentially** in Q1 2026 |

---

## Infosys Annual Report 2025 (`infosys_annual_report_2025.pdf`)

### Basic

| Question | Expected Answer |
|----------|----------------|
| What is Infosys total revenue for FY2025? | **₹1,36,592 crore** (revenue from operations) |
| What is Infosys total income for FY2025? | **₹1,41,374 crore** (including other income of ₹4,782 crore) |
| What is Infosys operating profit for FY2025? | **₹30,880 crore** |
| What is Infosys gross profit for FY2025? | **₹42,481 crore** |

### Intermediate

| Question | Expected Answer |
|----------|----------------|
| What is Infosys revenue by geography for FY2025? | North America ₹94,397 cr, Europe ₹48,595 cr, India ₹5,014 cr, Rest of World ₹14,984 cr |
| What is Infosys revenue growth from FY2024 to FY2025? | ₹1,28,933 cr → ₹1,36,592 cr = **+5.9% YoY** |
| What are Infosys employee benefit expenses in FY2025? | **₹67,466 crore** |
| What is Infosys operating margin for FY2025? | ₹30,880 / ₹1,36,592 = **~22.6%** |

### Complex

| Question | Expected Answer |
|----------|----------------|
| How did Infosys cost of technical sub-contractors change YoY? | FY2024: ₹18,638 cr → FY2025: ₹19,353 cr = **+3.8% increase** |
| What is Infosys revenue from North America as % of total in FY2025? | ₹94,397 / ₹1,62,990 = **~57.9%** |
| Compare Infosys selling & marketing vs G&A expenses in FY2025 | Selling & marketing: ₹6,282 cr; G&A: ₹5,319 cr; Total opex: ₹11,601 cr |

---

## Notes

- All answers verified against source documents
- RAG retrieves top-k=5 chunks; sources shown with relevance scores (0–100%)
- Model: Amazon Nova Lite v1 (`amazon.nova-lite-v1:0`)
- Embeddings: Amazon Titan Embed Text v1
- Index: LlamaIndex VectorStoreIndex + BM25 hybrid with cross-encoder reranking
