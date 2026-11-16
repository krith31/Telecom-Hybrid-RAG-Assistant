"""Reference-based retrieval metrics; missing labels are not zero scores."""
import math
import re

def retrieval_metrics(retrieved_ids, relevant_ids, k=3):
    if k < 1:
        raise ValueError("k must be positive")
    if relevant_ids is None:
        return {"status":"unlabeled"}
    if not isinstance(relevant_ids, list) or any(not isinstance(x, str) for x in relevant_ids):
        raise ValueError("relevant_doc_ids must be a list of strings or null")
    relevant = set(relevant_ids)
    if not relevant:
        return {"status":"no_relevant_documents"}
    ranked = list(dict.fromkeys(str(x) for x in retrieved_ids))[:k]
    hits = [int(x in relevant) for x in ranked]
    total = sum(hits)
    dcg = sum(v/math.log2(i+2) for i,v in enumerate(hits))
    ideal = sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
    ap = sum(sum(hits[:i+1])/(i+1) for i,v in enumerate(hits) if v)/min(k,len(relevant))
    return {"status":"ok", f"precision@{k}":total/k, f"recall@{k}":total/len(relevant),
            f"hit@{k}":float(total>0), f"mrr@{k}":next((1/(i+1) for i,v in enumerate(hits) if v),0),
            f"ndcg@{k}":dcg/ideal, f"ap@{k}":ap}

def citation_validity(answer, chunks):
    """Checks source identity only; semantic support is judged separately."""
    citations = re.findall(r"\[Spec:\s*([^|\]]+)\|\s*Clause\s+([^\]]+)\]", answer, re.I)
    known = {(str(c.get("metadata",{}).get("spec","")).strip(),
              str(c.get("metadata",{}).get("section_id","")).strip()) for c in chunks}
    valid = sum((spec.strip(),clause.strip()) in known for spec,clause in citations)
    return {"citation_count":len(citations), "valid_source_count":valid,
            "source_validity":valid/len(citations) if citations else None}
