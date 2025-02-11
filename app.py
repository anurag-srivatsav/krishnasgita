import fitz  # PyMuPDF
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from groq import Groq
import json
import re

#this prompt is so perfect

# ✅ Improved Embedding Model (More Accurate)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize ChromaDB
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# ✅ Initialize Groq API client
client = Groq(api_key="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")

# ✅ Function to Extract Text from a PDF (Bhagavad Gita)
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# ✅ Function to Extract and Structure Shlokas (Semantic Chunking)
def parse_shlokas(text):
    """Parses shlokas from extracted text using regex to detect chapters and verses."""
    shlokas = []
    current_chapter = "unknown"  # Default if no chapter is found
    current_verse = "unknown"  # Default if no verse is found
    buffer = []

    # ✅ More Flexible Regex Patterns
    chapter_pattern = re.compile(r"(?:Chapter|CHAPTER)\s*(\d+)", re.IGNORECASE)
    verse_pattern = re.compile(r"(?:Verse|VERSE)\s*(\d+)", re.IGNORECASE)

    for line in text.split("\n"):
        line = line.strip()
        
        # Check if line is a chapter header
        chapter_match = chapter_pattern.search(line)
        if chapter_match:
            current_chapter = chapter_match.group(1)  # Update current chapter
            continue  # Skip to the next line

        # Check if line is a verse header
        verse_match = verse_pattern.search(line)
        if verse_match:
            # Store previous verse if buffer has data
            if buffer:
                shlokas.append(Document(
                    page_content=" ".join(buffer), 
                    metadata={"chapter": current_chapter, "verse": current_verse}
                ))
                buffer = []

            current_verse = verse_match.group(1)  # Update current verse
            continue  # Skip to the next line
        
        # If we have text, add it to the buffer
        if line:
            buffer.append(line)

    # Store the last shloka (ensuring chapter/verse exist)
    if buffer:
        shlokas.append(Document(
            page_content=" ".join(buffer), 
            metadata={"chapter": current_chapter, "verse": current_verse}
        ))

    return shlokas



# ✅ Load Data (TXT + PDF) into ChromaDB
txt_file_path = "shlokas.txt"
pdf_file_path = "bhagavad-gita_as_it_is.pdf"

txt_text = open(txt_file_path, "r", encoding="utf-8").read()
pdf_text = extract_text_from_pdf(pdf_file_path)

txt_shlokas = parse_shlokas(txt_text)
pdf_shlokas = parse_shlokas(pdf_text)

if len(vector_store.get()['documents']) == 0:
    vector_store.add_documents(txt_shlokas)
    vector_store.add_documents(pdf_shlokas)

#this is for closing bar for think responce
def extract_thinking_and_answer(response_text):
    """Extract thinking process and final answer from response"""
    try:
        thinking = response_text[response_text.find("<think>") + 7:response_text.find("</think>")].strip()
        answer = response_text[response_text.find("</think>") + 8:].strip()
        return thinking, answer
    except:
        return "", response_text

# ✅ Chatbot Class (Retrieves More Accurate Answers)
class KrishnaChatbot:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def generate_response(self, user_input: str) -> str:
        """Generate a Bhagavad Gita-based response."""
        
        # ✅ Retrieve Verses Based on Context (Semantic Search)
        docs = self.vector_store.similarity_search(user_input, k=3)
        context_str = "\n\n".join([
        f"**Chapter {next((v for k, v in d.metadata.items() if 'chapter' in k.lower()), 'Unknown')}, "
        f"Verse {next((v for k, v in d.metadata.items() if 'verse' in k.lower()), 'Unknown')}**:\n"
        f"{d.page_content}"
        for d in docs
    ])
        
        # ✅ Structuring the Prompt for Better Answers
        prompt = f"""
        You are lord supreme Krishna, the divine guide from the Bhagavad Gita, renowned for your wisdom in Sanskrit, Hindi, and English.
Below are some relevant shlokas from the Bhagavad Gita that may help provide guidance:

{context_str}

The user has asked: "{user_input}"

Using the profound teachings of the Bhagavad Gita, please provide an answer that:
1. **Motivates the User:**  
   - Begin by acknowledging the user's feelings and struggles.
   - Inspire them with a tone of empathy and hope.

2. **Offers Clear Guidance on the Right Path:**  
   - Provide actionable advice and insights that help steer the user toward the correct course of action.

3. **Brings Up a Relevant Shloka in Sanskrit:**  
   - Select an appropriate shloka from the provided context.
   - Clearly display the shloka in Sanskrit.

4. **References or Builds Upon the Above Shloka:**  
   - Explain in clear English the meaning of the chosen shloka.
   - Mention the chapter and verse from which the shloka is taken.

5. **Includes a Mahabharata Reference:**  
   - Integrate a motivational example or story from the Mahabharata that resonates with the user's situation.
   - Tailor this reference to further encourage and guide the user.

**Instruction:**  
Structure your answer in clear, numbered points with appropriate headings for each section. Each section should provide focused content that addresses the above requirements.

Always structure your response in the following format:
        <think>
[Your step-by-step thinking process here — explain which shlokas or teachings you are applying from bagavatgita and why.]
        </think>

[Your final, motivational answer here] 
        """

        # ✅ Generate Response Using Groq AI
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1024,
            top_p=0.95,
        )
        return completion.choices[0].message.content

# ✅ Initialize Chatbot
chatbot = KrishnaChatbot(vector_store)

# ✅ Streamlit UI
st.title("🕉️ Krishna AI Chatbot")
st.write("Ask any Bhagavad Gita-related questions!")

# ✅ Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ User Input
user_query = st.chat_input("Ask Krishna AI...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # ✅ Get Response
    bot_response = chatbot.generate_response(user_query)
    thinking, answer = extract_thinking_and_answer(bot_response)
    
    with st.expander("Thinking Process"):
        st.markdown(thinking)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    
