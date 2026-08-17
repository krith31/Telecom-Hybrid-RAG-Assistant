import os
import json
import re
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

class TelecomChunker:
    def __init__(self, input_dir, output_dir, chunk_size=600, chunk_overlap=120):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base") # Standard for GPT-4/3.5
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def format_table(self, table_data):
        """Convert table data to a markdown-like string."""
        if not table_data: return ""
        rows = []
        for row in table_data:
            rows.append(" | ".join(row))
        return "\n".join(rows)

    def expand_acronyms_in_text(self, text, abbreviations):
        """Optionally expand acronyms or just return metadata. 
        Here we prepend the relevant ones as metadata context to the chunk."""
        found = {}
        # Simple regex for uppercase acronyms
        words = re.findall(r'\b[A-Z]{2,}\b', text)
        for word in words:
            if word in abbreviations:
                found[word] = abbreviations[word]
        return found

    def chunk_section(self, section, abbreviations):
        """Chunks a single section into multiple pieces with breadcrumbs."""
        spec_id = section.get("spec", "Unknown Spec")
        
        # Clean paths and titles (remove \t and extra spaces)
        clean_path = [re.sub(r'[ \t]+', ' ', p).strip() for p in section["path"]]
        clean_title = re.sub(r'[ \t]+', ' ', section["title"]).strip()
        
        path_str = " > ".join(clean_path)
        breadcrumb = f"[{spec_id} | {path_str}]\n\n"
        
        full_text = ""
        chunks = []
        
        # Combine text and tables
        current_text_block = ""
        for item in section["content"]:
            if isinstance(item, str):
                current_text_block += item + "\n"
            elif isinstance(item, dict) and item.get("type") == "table":
                # Process table
                table_str = self.format_table(item["data"])
                # If table is small, add to block
                if self.count_tokens(table_str) < self.chunk_size * 0.5:
                    current_text_block += f"\n[Table Content]:\n{table_str}\n"
                else:
                    # If table is large, it might need its own chunking logic
                    # For now, we'll let the splitter handle it, or could do row-wise
                    current_text_block += f"\n[Large Table Content]:\n{table_str}\n"

        if not current_text_block.strip():
            return []

        # Use LangChain's splitter for the prose
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        text_chunks = splitter.split_text(current_text_block)
        
        for i, text in enumerate(text_chunks):
            # Clean up tabs and excessive whitespace inside the chunk text
            # This replaces consecutive tabs/spaces (but not newlines) with a single space
            text = re.sub(r'[ \t]+', ' ', text)
            
            # Inject breadcrumb
            chunk_content = breadcrumb + text.strip()
            
            # Extract acronyms for this specific chunk
            relevant_abbr = self.expand_acronyms_in_text(text, abbreviations)
            
            chunks.append({
                "id": f"{section['id']}_{i}",
                "content": chunk_content,
                "metadata": {
                    "spec": spec_id,
                    "section_id": section["id"],
                    "title": clean_title,
                    "path": clean_path,
                    "acronyms": relevant_abbr,
                    "chunk_index": i
                }
            })
            
        return chunks

    def process_all(self):
        all_chunks = []
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.input_dir, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"Chunking {data['spec']}...")
                abbreviations = data.get("abbreviations", {})
                
                for section in tqdm(data["sections"]):
                    section_chunks = self.chunk_section(section, abbreviations)
                    all_chunks.extend(section_chunks)

        # Save all chunks to a single file for the RAG pipeline
        output_file = os.path.join(self.output_dir, "vector_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    chunker = TelecomChunker(
        input_dir=r"d:\Antigravity\TeleComRAG\Data\processed",
        output_dir=r"d:\Antigravity\TeleComRAG\Data\chunks"
    )
    chunker.process_all()
