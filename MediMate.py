

import streamlit as st 
import google.generativeai as genai 
import google.ai.generativelanguage as glm 
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from dotenv import load_dotenv
from PIL import Image
# from streamlit_tags import st_aggrid
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os 
import io 

load_dotenv()

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr


# API_KEY = os.environ.get("GOOGLE_API_KEY")
API_KEY=genai.configure(api_key="AIzaSyCCMu8d6y4K_B-YLNTJbfAY2ktG7SEEB5c")
class GeminiProLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-pro"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        gemini_pro_model = genai.GenerativeModel("gemini-pro")

        
        model_response = gemini_pro_model.generate_content(
            prompt, 
            generation_config={"temperature": 0.1}
        )
        text_content = model_response.candidates[0].content.parts[0].text
        return text_content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_id": "gemini-pro", "temperature": 0.1}

generation_config = {
  "temperature": 1,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
def load_chain():
    llm = GeminiProLLM()
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()
st.image("./logo.png", width=200)
st.write("")
def generate_pdf(messages):
    doc = SimpleDocTemplate("conversation.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            elements.append(Paragraph(f"User: {content}", styles["BodyText"]))
        else:
            elements.append(Paragraph(f"Assistant: {content}", styles["BodyText"]))
        
    doc.build(elements)

def load_text_file(messages: List[Mapping[str, str]]) -> str:
    text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"
    return text
def generate_text(messages: List[Mapping[str, str]], file_name: str = "conversation.txt"):
    text = load_text_file(messages)
    with open(file_name, "w") as f:
        f.write(text)
gemini_pro, gemini_vision = st.tabs(["MediMate", "Medimate Pro"])

def main():
    prompt = st.chat_input("You:")
    model = genai.GenerativeModel("gemini-pro")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = 0
    
    with gemini_pro:
        
        st.header("I'm Your MadiMate")
        st.write("")
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        # Display previous messages
        for message in st.session_state['messages']:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.markdown(content)

        
        

        # if st.button("SEND",use_container_width=True):
        #     response = model.generate_content(prompt)

        #     st.write("")
        #     st.header(":blue[Response]")
        #     st.write("")

        #     st.markdown(response.text)
        
        
        if prompt:
            st.session_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = chatchain(prompt)["response"]
            st.session_state['messages'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        
        if st.button("Download Conversation as PDF"):
            
            generate_pdf(st.session_state['messages'])
                
            st.download_button(
                label="Download conversation.pdf",
                data=open("conversation.pdf", "rb").read(),
                file_name="conversation.pdf",
                mime="application/pdf"
                )
        if st.button("Download Conversation as Text"):
            generate_text(st.session_state['messages'], "conversation.txt")
            st.download_button(
                label="Download conversation.txt",
                data=open("conversation.txt", "rb").read(),
                file_name="conversation.txt",
                mime="text/plain"
            )
        

    with gemini_vision:
        st.header("Interact with MediMate Pro")
        st.write("")

        image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
        uploaded_file = st.file_uploader("Choose and Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), use_column_width=True)

            st.markdown("""
                <style>
                        img {
                            border-radius: 10px;
                        }
                </style>
                """, unsafe_allow_html=True)
            
        if st.button("GET RESPONSE", use_container_width=True):
            model = genai.GenerativeModel("gemini-pro-vision")

            if uploaded_file is not None:
                if image_prompt != "":
                    image = Image.open(uploaded_file)

                    response = model.generate_content(
                        glm.Content(
                            parts = [
                                glm.Part(text=image_prompt),
                                glm.Part(
                                    inline_data=glm.Blob(
                                        mime_type="image/jpeg",
                                        data=image_to_byte_array(image)
                                    )
                                )
                            ]
                        )
                    )

                    response.resolve()

                    st.write("")
                    st.write(":blue[Response]")
                    st.write("")

                    st.markdown(response.text)

                else:
                    st.write("")
                    st.header(":red[Please Provide a prompt]")

            else:
                st.write("")
                st.header(":red[Please Provide an image]")

if __name__ == "__main__":
    main()
    
