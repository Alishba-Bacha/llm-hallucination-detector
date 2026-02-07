# Agentic Research Paper Hallucination Detector
## Product Requirements Document

### Problem Statement

**Specific Bottleneck:** Researchers using AI writing assistants (ChatGPT, Claude, Copilot) unintentionally introduce factual errors into academic papers. The current verification process is **manually intensive, expertise-dependent, and doesn't scale**, causing:

1. **Time Waste:** 3-4 hours per paper spent manually checking citations and claims
2. **Error Propagation:** 30-40% of subtle hallucinations missed during peer review
3. **Retraction Risk:** 15% of paper retractions are due to AI-generated factual errors
4. **Reviewer Burden:** 50% of peer review time consumed by basic fact-checking instead of scientific evaluation
5. **Publication Delays:** 6-9 months to discovery and correction of errors through traditional peer review

This problem **cannot be solved with a single LLM response** because it requires:
- Extracting specific claims from complex academic text
- Querying multiple academic databases (arXiv, PubMed, Semantic Scholar)
- Comparing claims against evidence with nuanced understanding
- Checking internal consistency across the entire paper
- Generating actionable recommendations with confidence scores

### User Personas

#### Primary User: Research Assistant

#### Secondary User: Journal Editor

#### Tertiary User: Research Lab Manager 

### Success Metrics

#### Phase 1:  
 - **Accuracy:** 70% recall, 70% precision on hallucination detection
 - **Speed:** <60 seconds processing time for research abstracts
 - **Usability:** 10 satisfied beta users from target research lab
 - **Coverage:** Processes arXiv CS abstracts successfully

#### Phase 2:
- **Accuracy:** 85% recall, 80% precision on full papers
- **Speed:** <5 minutes for 10-page PDFs
- **Coverage:** Supports computer science and biology domains

#### Phase 3: 
- **Accuracy:** 90% recall, 85% precision across domains
- **Speed:** <3 minutes average processing time
- **Usability:** 100+ papers processed with positive feedback
- **Coverage:** Integrates with Overleaf/arXiv submission systems

#### Tool & Data Inventory
### 1. **Academic Databases & APIs:**
## Primary Sources for Evidence Retrieval
**ARXIV_API** = "https://api.arxiv.org/"  
- Usage: Search papers by keyword, author, title
- Rate Limit: 1 request/second, 5000 requests/day
- Data: Abstracts, full texts (when available), metadata
- Cost: Free

**SEMANTIC_SCHOLAR_API** = "https://api.semanticscholar.org/"  
- Usage: Get citation graphs, paper embeddings, TLDRs  
- Rate Limit: 100 requests/minute (free tier)
- Data: Citations, references, field of study tags
- Cost: Free tier available

**PUBMED_API** = "https://eutils.ncbi.nlm.nih.gov/"  
- Usage: Retrieve biomedical and life sciences papers
- Rate Limit: 10 requests/second
- Data: Full-text articles, MeSH terms, clinical data
- Cost: Free

**CROSSREF_API** = "https://api.crossref.org/"  
- Usage: Resolve DOIs, retrieve publication metadata
- Rate Limit: 50 requests/second (polite pool)
- Data: Publication dates, journals, authors, citations
- Cost: Free

**IEEE_XPLORE_API** = "https://ieeexploreapi.ieee.org/"  
- Usage: Access engineering and electronics papers
- Cost: Open access papers

#### Quantitative Metrics:
1. **Detection Performance:**
   - Hallucination Recall: >85%
   - Precision: >85%
   - False Positive Rate: <20%
   - F1 Score: >85%

2. **Processing Efficiency:**
   - End-to-End Time: <10 minutes per paper

3. **User Impact:**
   - Time Saved: 85% reduction in manual verification
   - Error Reduction: 80% decrease in published errors

#### Qualitative Metrics:
1. **Academic Impact:**
   - Reduced paper retractions
   - Higher reviewer satisfaction
   - Faster publication cycles
   - Improved research credibility
