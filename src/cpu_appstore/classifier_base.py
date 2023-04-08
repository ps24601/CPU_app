from haystack.nodes import TransformersDocumentClassifier
from haystack.schema import Document
from typing import List, Tuple, Dict
from typing_extensions import Literal
import logging
import pandas as pd
from pandas import DataFrame, Series
from utils.checkconfig import getconfig
import utils
from utils.streamlitcheck import check_streamlit
from utils.preprocessing import processingpipeline

try:
    import streamlit as st
except ImportError:
    logging.info("Streamlit not installed")

# Declare all the necessary variables
def get_classifier_params(model_name):
    config = getconfig('paramconfig.cfg')
    params = {}
    params['model_name'] = config.get(model_name,'MODEL')
    params['split_by'] = config.get(model_name,'SPLIT_BY')
    params['split_length'] = int(config.get(model_name,'SPLIT_LENGTH'))
    params['split_overlap'] = int(config.get(model_name,'SPLIT_OVERLAP'))
    params['remove_punc'] = bool(int(config.get(model_name,'REMOVE_PUNC')))
    params['split_respect_sentence_boundary'] = bool(int(config.get(model_name,'RESPECT_SENTENCE_BOUNDARY')))
    params['threshold'] = float(config.get(model_name,'THRESHOLD'))
    params['top_n'] = int(config.get(model_name,'TOP_KEY'))

    return params

@st.cache(allow_output_mutation=True)
def para_classification(haystack_doc:List[Document],_lab_dict:Dict,
                        threshold:float = 0.8,param_val = None, 
                        classifier_model:TransformersDocumentClassifier= None
                        )->Tuple[DataFrame,Series]:
    """
    Text-Classification on the list of texts provided. Classifier provides the 
    most appropriate label for each text. these labels are in terms of if text 
    belongs to which particular Sustainable Devleopment Goal (SDG).

    Params
    ---------
    haystack_doc: List of haystack Documents. The output of Preprocessing Pipeline 
    contains the list of paragraphs in different format,here the list of 
    Haystack Documents is used.
    threshold: threshold value for the model to keep the results from classifier
    classifiermodel: you can pass the classifier model directly,which takes priority
    however if not then looks for model in streamlit session.
    In case of streamlit avoid passing the model directly.


    Returns
    ----------
    df: Dataframe with two columns['SDG:int', 'text']
    x: Series object with the unique SDG covered in the document uploaded and 
    the number of times it is covered/discussed/count_of_paragraphs. 

    """
    logging.info("Working on Economy Target Classification")
    if not classifier_model:
        if check_streamlit():
            classifier_model = st.session_state['{}_classifier'.format(param_val)]
        else:
            logging.warning("""No streamlit envinornment found or pass correct 
                            param_val to look in session_state or Pass the classifier object""")
            return
    
    results = classifier_model.predict(haystack_doc)


    labels_= [(l.meta['classification']['label'],
            l.meta['classification']['score'],l.content,) for l in results]

    df = DataFrame(labels_, columns=["Target Label","Relevancy","text"])
    
    df = df.sort_values(by="Relevancy", ascending=False).reset_index(drop=True)  
    df.index += 1
    df =df[df['Relevancy']>threshold]

    # creating the dataframe for value counts of SDG, along with 'title' of SDGs
    x = df['Target Label'].value_counts()
    x = x.rename('count')
    x = x.rename_axis('Target Label').reset_index()
    x["Target Label"] = pd.to_numeric(x["Target Label"])
    x = x.sort_values(by=['count'], ascending=False)
    x['TARGET_name'] = x['Target Label'].apply(lambda x: _lab_dict[x])
    x['TARGET_Num'] = x['Target Label'].apply(lambda x: "TARGET LABEL "+str(x))

    df['TARGET LABEL'] = pd.to_numeric(df['TARGET LABEL'])
    df = df.sort_values('TARGET LABEL')

    return df, x

@st.cache(allow_output_mutation=True)
def load_Classifier(config_file:str = None,param_val = None, 
                    classifier_name:str = None):
    """
    loads the document classifier using haystack, where the name/path of model
    in HF-hub as string is used to fetch the model object.Either configfile or 
    model should be passed.
    1. https://docs.haystack.deepset.ai/reference/document-classifier-api
    2. https://docs.haystack.deepset.ai/docs/document_classifier

    Params
    --------
    config_file: config file path from which to read the model name
    classifier_name: if modelname is passed, it takes a priority if not \
    found then will look for configfile, else raise error.


    Return: document classifier model
    """
    if not classifier_name:
        if not config_file:
            logging.warning("Pass either model name or config file")
            return
        else:
            config = getconfig(config_file)
            classifier_name = config.get(param_val,'MODEL')
    
    logging.info("Loading classifier")    
    doc_classifier = TransformersDocumentClassifier(
                        model_name_or_path=classifier_name,
                        task="text-classification")

    return doc_classifier


def runPreprocessingPipeline(file_name:str, file_path:str, 
            split_by: Literal["sentence", "word"] = 'word',
            split_length:int = 60, split_respect_sentence_boundary:bool = False,
            split_overlap:int = 10,remove_punc:bool = False)->List[Document]:
    """
    creates the pipeline and runs the preprocessing pipeline, 
    the params for pipeline are fetched from paramconfig

    Params
    ------------

    file_name: filename, in case of streamlit application use 
    st.session_state['filename']
    file_path: filepath, in case of streamlit application use st.session_state['filepath']
    split_by: document splitting strategy either as word or sentence
    split_length: when synthetically creating the paragrpahs from document,
                    it defines the length of paragraph.
    split_respect_sentence_boundary: Used when using 'word' strategy for 
    splititng of text.
    split_overlap: Number of words or sentences that overlap when creating
        the paragraphs. This is done as one sentence or 'some words' make sense
        when  read in together with others. Therefore the overlap is used.
    remove_punc: to remove all Punctuation including ',' and '.' or not


    Return
    --------------
    List[Document]: When preprocessing pipeline is run, the output dictionary 
    has four objects. For the Haysatck implementation of SDG classification we, 
    need to use the List of Haystack Document, which can be fetched by 
    key = 'documents' on output.

    """

    classifier_processing_pipeline = processingpipeline()

    output_sdg_pre = classifier_processing_pipeline.run(file_paths = file_path, 
                            params= {"FileConverter": {"file_path": file_path, \
                                        "file_name": file_name}, 
                                     "UdfPreProcessor": {"remove_punc": remove_punc, \
                                            "split_by": split_by, \
                                            "split_length":split_length,\
                                            "split_overlap": split_overlap, \
        "split_respect_sentence_boundary":split_respect_sentence_boundary}})
    
    return output_sdg_pre

    