import streamlit as st
import pandas as pd
import os
import base64
import shutil

from model_setup import Model_SetUp

st.title('Welcome to ChemFate!')

chem_type = st.selectbox(
     'Please select a chemical type:',
     ('NonionizableOrganic', 'IonizableOrganic', 'Metal', 'Nanomaterial'))

st.write('You selected:', chem_type)

st.subheader("Input Files")
chem_file = st.file_uploader(label="Chemical File", type=["xlsx"])
region_file = st.file_uploader(label="Region File", type=["xlsx"])
release_file = st.file_uploader(label="Chemical Release File", type=["xlsx"])

run_option = 1 # can be 1 or 2:
bgPercOption2 = 10 # can be anywhere between 0-100

st.subheader("Output Files")
file_name = st.text_input('Output File Name', '') # pyrimethanil, cyprodinil, copper, nanoCopper

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(CUR_PATH, 'Output', file_name)

if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)

# year month day
start_date = st.text_input('Start Date', '2005 1 1')
end_date = st.text_input('Start Date', '2005 1 5')

def create_download_zip(zip_directory, zip_path, filename):
    """
        zip_directory (str): path to directory  you want to zip
        zip_path (str): where you want to save zip file
        filename (str): download filename for user who download this
    """
    shutil.make_archive(zip_path, 'zip', zip_directory)
    with open(zip_path + '.zip', 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{filename}\'>\
            **Download Results (.zip)** \
        </a>'
        st.markdown(href, unsafe_allow_html=True)

if st.button(label="Click to Run ChemFate"):
    st.write("ChemFate Model Started to Run ......")
    model = Model_SetUp(start_date, end_date, run_option, bgPercOption2,
                        chem_type, chem_file, region_file, release_file, output_file_path, file_name)
    model.run_model()
    create_download_zip(zip_directory=output_file_path,
                        zip_path=output_file_path,
                        filename=file_name + '.zip')


