import streamlit as st
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_data(path):
    return os.path.join(_ROOT, 'data', path)

def app():
    
    
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center;  \
                      color: black;'> Climate Policy Understanding App</h2>", 
                      unsafe_allow_html=True)

    
    st.markdown("<div style='text-align: center; \
                    color: grey;'>Climate Policy Understanding App is an open-source\
                         digital tool which aims to assist policy analysts and \
                          other users in extracting and filtering relevant \
                            information from public documents.</div>",
                        unsafe_allow_html=True)
    footer = """
           <div class="footer-custom">
               Guidance & Feedback - <a>Nadja Taeger</a> |<a>Marie Hertel</a> | <a>Cecile Schneider</a> |
               Developer - <a href="https://www.linkedin.com/in/erik-lehmann-giz/" target="_blank">Erik Lehmann</a>  |   
               <a href="https://www.linkedin.com/in/prashantpsingh/" target="_blank">Prashant Singh</a> |
               
           </div>
       """
    st.markdown(footer, unsafe_allow_html=True)

    c1, c2, c3 =  st.columns([8,1,12])
    with c1:
        st.image(get_data("ndc.png"))
    with c3:
        st.markdown('<div style="text-align: justify;">The manual extraction \
        of relevant information from text documents is a \
    time-consuming task for any policy analysts. As the amount and length of \
    public policy documents in relation to sustainable development (such as \
    National Development Plans and Nationally Determined Contributions) \
    continuously increases, a major challenge for policy action tracking – the \
    evaluation of stated goals and targets and their actual implementation on \
    the ground – arises. Luckily, Artificial Intelligence (AI) and Natural \
    Language Processing (NLP) methods can help in shortening and easing this \
    task for policy analysts.</div><br>',
    unsafe_allow_html=True)

    intro = """
    <div style="text-align: justify;">

    For this purpose, IKI Tracs, SV KLIMA, SPA and Data Service Center (Deutsche Gesellschaft für Internationale \
    Zusammenarbeit (GIZ) GmbH) are collaborating since 2022 in the development \
    of an AI-powered open-source web application that helps find and extract \
    relevant information from public policy documents faster to facilitate \
    evidence-based decision-making processes in sustainable development and beyond.  


    </div>
    <br>
    """
    st.markdown(intro, unsafe_allow_html=True)
    st.image(get_data("paris.png"))