"""Portable source ingestion for an isolated evaluation collection."""
import hashlib
import json
from pathlib import Path
import re

from .capture import write_json


def document_sections(path):
    path=Path(path)
    if path.suffix.lower() in (".txt",".md"):
        return [{"section_id":"document","title":path.stem,"text":path.read_text(encoding="utf-8-sig")}]
    if path.suffix.lower()!=".docx":
        raise ValueError("Supported source documents are .docx, .txt and .md")
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    document=Document(path)
    sections=[];current={"section_id":"preamble","title":"Preamble","blocks":[]}
    for element in document.element.body.iterchildren():
        if element.tag==qn("w:p"):
            paragraph=Paragraph(element,document)
            text=paragraph.text.strip()
            if not text or paragraph.style.name.lower().startswith("toc"):
                continue
            match=re.match(r"^(\d+(?:\.\d+)*|Annex\s+[A-Z](?:\.\d+)*)\s+(.+)$",text)
            if match:
                if current["blocks"]:
                    sections.append({**current,"text":"\n".join(current.pop("blocks"))})
                current={"section_id":match[1],"title":match[2],"blocks":[]}
            else:
                current["blocks"].append(text)
        elif element.tag==qn("w:tbl"):
            table=Table(element,document)
            current["blocks"].append("\n".join(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows))
    if current["blocks"]:
        sections.append({**current,"text":"\n".join(current.pop("blocks"))})
    return sections


def source_chunks(directory, tokenizer, max_tokens=384, overlap=64):
    if max_tokens<32 or overlap<0 or overlap>=max_tokens:
        raise ValueError("Invalid token window or overlap")
    directory=Path(directory)
    paths=sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in (".docx",".txt",".md"))
    if not paths:
        raise ValueError(f"No .docx, .txt or .md source documents found in {directory}")
    chunks=[]
    for path in paths:
        relative=path.relative_to(directory).as_posix()
        source_hash=hashlib.sha256(relative.encode()+b"\0"+path.read_bytes()).hexdigest()[:16]
        for section in document_sections(path):
            text=section["text"].strip()
            if not text:
                continue
            prefix=f"[{path.stem} | Clause {section['section_id']}]\n"
            budget=max_tokens-len(tokenizer.encode(prefix,add_special_tokens=False))-2
            if budget<=overlap:
                raise ValueError("Source breadcrumb leaves insufficient chunk token budget")
            ids=tokenizer.encode(text,add_special_tokens=False)
            for i,start in enumerate(range(0,len(ids),budget-overlap)):
                content=prefix+tokenizer.decode(ids[start:start+budget],skip_special_tokens=True)
                chunks.append({"id":f"{source_hash}::{section['section_id']}::{i}","content":content,
                               "metadata":{"spec":path.stem,"section_id":section["section_id"],
                                           "title":section["title"],"source_path":relative,"source_hash":source_hash,
                                           "chunk_index":i}})
                if start+budget>=len(ids):
                    break
    validate_chunks(chunks)
    return chunks


def validate_chunks(chunks):
    if not isinstance(chunks,list) or not chunks:
        raise ValueError("Need a nonempty chunk list")
    seen=set()
    for chunk in chunks:
        if not isinstance(chunk,dict) or not isinstance(chunk.get("id"),str) or not chunk["id"]:
            raise ValueError("Every chunk requires a nonempty string ID")
        if chunk["id"] in seen:
            raise ValueError("Duplicate chunk IDs; fix source IDs before evaluation")
        seen.add(chunk["id"])
        if not isinstance(chunk.get("content"),str) or not chunk["content"].strip():
            raise ValueError("Every chunk requires nonempty content")
    return chunks


def read_collection(store, name):
    collection=store.get_or_create_collection(name)
    chunks=[]
    offset=0
    while True:
        data=collection.get(limit=500,offset=offset,include=["documents","metadatas"])
        if not data["ids"]:
            break
        for id_,text,metadata in zip(data["ids"],data["documents"],data["metadatas"]):
            chunks.append({"id":id_,"content":text,"metadata":metadata or {}})
        offset+=len(data["ids"])
    return validate_chunks(chunks)


def prepare_corpus(pipeline,output,specs=None,chunks_file=None,collection="telecom_specs"):
    output=Path(output)
    store=pipeline.vector_store
    if specs:
        model=store.embedding_manager._model
        max_tokens=min(384,int(model.max_seq_length))
        chunks=source_chunks(specs,model.tokenizer,max_tokens=max_tokens,overlap=min(64,max_tokens//4))
    elif chunks_file:
        chunks=validate_chunks(json.loads(Path(chunks_file).read_text(encoding="utf-8")))
    else:
        chunks=read_collection(store,collection)
    digest=hashlib.sha256(json.dumps(chunks,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    snapshot=output/"corpus_chunks.json"
    write_json(snapshot,chunks)
    if specs or chunks_file:
        collection="telecom_eval_"+digest[:16]
        store.ingest_vector_chunks(str(snapshot),collection_name=collection)
    return chunks,collection,digest
