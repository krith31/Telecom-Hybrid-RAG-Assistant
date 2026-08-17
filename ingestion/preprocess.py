import os
import re
import json
from docx import Document
from tqdm import tqdm
import pandas as pd

class TelecomPreprocessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.abbreviations = {}
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_abbreviations(self, doc):
        """Extract abbreviations from section 3.2 of the 3GPP document."""
        print("Extracting abbreviations...")
        in_abbreviation_section = False
        for para in doc.paragraphs:
            if para.style.name.lower().startswith('toc'):
                continue
                
            text = para.text.strip()
            
            # Detect section 3.2
            if re.match(r"^3\.2[\s\xa0]+Abbreviations", text, re.IGNORECASE):
                in_abbreviation_section = True
                continue
            
            if in_abbreviation_section:
                # Stop if we hit section 3.3 or 4
                if re.match(r"^(3\.3|4)[\s\xa0]+", text):
                    break
                
                # Typical format: AMF    Access and Mobility Management Function
                # Or sometimes tables are used. If it's a paragraph:
                match = re.match(r"^([A-Z0-9\-\/]+)[\s\xa0\t]+(.+)$", text)
                if match:
                    abbr, full = match.groups()
                    if abbr != text and len(abbr) > 1:
                        self.abbreviations[abbr] = full.strip()

        # Also check tables in section 3.2 if paragraphs didn't yield much
        if not self.abbreviations:
            for table in doc.tables:
                # Check if the table is near the abbreviations section
                # This is a bit tricky, but 3GPP often uses tables for abbr
                first_cell = table.cell(0, 0).text.strip()
                if "abbreviation" in first_cell.lower() or "term" in first_cell.lower():
                    for row in table.rows[1:]:
                        if len(row.cells) >= 2:
                            abbr = row.cells[0].text.strip()
                            full = row.cells[1].text.strip()
                            if abbr: self.abbreviations[abbr] = full

        print(f"Found {len(self.abbreviations)} abbreviations.")

    def clean_text(self, text):
        """Remove figure placeholders, admin notes, and void clauses."""
        # Remove Figure placeholders
        text = re.sub(r"Figure\s+\d+.*?:.*", "", text)
        # Remove Void clauses
        if "void" in text.lower() and len(text) < 50:
            return ""
        # Remove editor notes (simple heuristic)
        if text.startswith("Editor's note:"):
            return ""
        # Clean non-breaking spaces
        text = text.replace('\xa0', ' ')
        return text.strip()

    def parse_docx(self, file_path):
        doc = Document(file_path)
        spec_name = os.path.basename(file_path).replace(".docx", "")
        
        # Extract abbreviations first
        self.extract_abbreviations(doc)
        
        sections = []
        current_section = {"id": "", "title": "", "content": [], "path": []}
        breadcrumb_stack = []

        print(f"Parsing {spec_name}...")
        
        # We iterate through paragraphs to build hierarchy
        for para in tqdm(doc.paragraphs):
            if para.style.name.lower().startswith('toc'):
                continue
                
            text = para.text.strip()
            if not text:
                continue

            # Clean non-breaking spaces before header check
            text = text.replace('\xa0', ' ')

            # Detect Heading (Regex for 3GPP numbering like 5.34.6.2)
            header_match = re.match(r"^(\d+(\.\d+)*)[\s]+(.*)$", text)
            
            # Check if it's an Annex
            annex_match = re.match(r"^Annex\s+([A-Z])[\s]+(.*)$", text)

            if header_match or annex_match:
                # Save previous section
                if current_section["id"]:
                    sections.append(current_section)

                if header_match:
                    section_id, _, section_title = header_match.groups()
                else:
                    section_id, section_title = annex_match.groups()
                    section_id = f"Annex {section_id}"

                # Skip Change History
                if "change history" in section_title.lower():
                    current_section = {"id": "", "title": "", "content": [], "path": []}
                    continue

                # Update breadcrumbs
                level = section_id.count(".")
                if "Annex" in section_id: level = 0
                
                breadcrumb_stack = breadcrumb_stack[:level]
                breadcrumb_stack.append(f"{section_id} {section_title}")

                current_section = {
                    "id": section_id,
                    "title": section_title,
                    "content": [],
                    "path": list(breadcrumb_stack),
                    "spec": spec_name
                }
            else:
                cleaned = self.clean_text(text)
                if cleaned and current_section["id"]:
                    current_section["content"].append(cleaned)

        # Tables handling
        print("Processing tables...")
        for table in doc.tables:
            # Skip abbreviations table
            if "abbreviation" in table.cell(0, 0).text.lower():
                continue
                
            table_data = []
            for row in table.rows:
                table_data.append([cell.text.strip() for cell in row.cells])
            
            # Convert to a readable string or keep as list
            if current_section["id"]:
                current_section["content"].append({"type": "table", "data": table_data})

        sections.append(current_section)
        return sections

    def process_all(self):
        all_specs_data = {}
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".docx"):
                path = os.path.join(self.input_dir, filename)
                sections = self.parse_docx(path)
                
                output_file = os.path.join(self.output_dir, filename.replace(".docx", ".json"))
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "spec": filename,
                        "abbreviations": self.abbreviations,
                        "sections": sections
                    }, f, indent=2)
                
                print(f"Saved {output_file}")

if __name__ == "__main__":
    preprocessor = TelecomPreprocessor(
        input_dir=r"d:\Antigravity\TeleComRAG\Data\specs",
        output_dir=r"d:\Antigravity\TeleComRAG\Data\processed"
    )
    preprocessor.process_all()
