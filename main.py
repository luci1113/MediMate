import streamlit as st 
import google.generativeai as genai 
import google.ai.generativelanguage as glm 
from dotenv import load_dotenv
from PIL import Image
import os 
import io
from langchain.chains.conversation.memory import ConversationSummaryMemory

# from gene import GenerativeModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


def api():
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=API_KEY)
    
    return API_KEY
def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr=imgByteArr.getvalue()
    return imgByteArr
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


def load_css():
    with open('static/style.css',"r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)
# def load_chain():
#     llm = GeminiProLLM()
#     memory = ConversationBufferMemory()
#     chain = ConversationChain(llm=llm, memory=memory)
#     return chain

# chatchain = load_chain()



st.image("./Google-Gemini-AI-Logo.png", width=200)
st.write("")

gemini_pro, gemini_vision = st.tabs(["Gemini Pro", "Gemini Pro Vision"])

def main():
    prompt = st.chat_input("You:")
    # button=st.button("send"":male-doctor:")
    model = genai.GenerativeModel("gemini-pro")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = 0
    with gemini_pro:
        
        st.header("Interact with Gemini Pro")
        st.write("")
        
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        # Display previous messages
        for message in st.session_state['messages']:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.markdown(content)
            

        if prompt:
            st.session_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = model.generate_content(prompt)["response"]
            st.session_state['messages'].append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    
    with gemini_vision:
        st.header("Interact with Gemini Pro Vision")
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






































# os.environ['GOOGLE_API_KEY'] = "AIzaSyDGgLvrWWfisiOjdMAkuej31EE3hC_OHKI"
# genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

# model = genai.GenerativeModel('gemini-pro')
# response = model.generate_content("How to make guns")

# print(response.prompt_feedback)