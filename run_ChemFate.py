import streamlit as st
from datetime import datetime
import pandas as pd
import os
import base64
import shutil

from model_setup import Model_SetUp

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

def get_date(region_file):
    df = pd.read_excel(region_file, sheet_name="Climate")
    start_month = df.iat[0, 0]
    start_day = df.iat[0, 1]
    start_year = df.iat[0, 2]
    end_month = df.iat[-1, 0]
    end_day = df.iat[-1, 1]
    end_year = df.iat[-1, 2]
    start_date = str(int(start_year)) + ' ' + str(int(start_month)) + ' ' + str(int(start_day))
    end_date = str(int(end_year)) + ' ' + str(int(end_month)) + ' ' + str(int(end_day))
    return start_date, end_date

st.title('Welcome to ChemFate!')
# selection for chemical type
step1_txt = "ChemFate predicts daily chemical environmental concentrations for four classes of chemicals. " \
            "ChemFate comprises four different models:\n" \
            "1) organoFate, for non-ionizable organic chemicals, \n" \
            "2) ionOFate, for ionizable organic chemicals, \n" \
            "3) metalFate, for metals, \n" \
            "4) nanoFate, for nanomaterials. "
st.write(step1_txt)
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

start_date_from_file = "2005 1 1"
end_date_from_file = "2014 12 31"
if region_file:
    start_date_from_file, end_date_from_file = get_date(region_file)

st.markdown("Please enter the start date and end date for your model simulation time: ")
st.markdown("Note: from your input region file, your date range is from " + "**" + start_date_from_file + "**" + " to "
            + "**"+ end_date_from_file + "**" + \
            ", but you can change the start date and end date to any date in between. "
            "Please check the dates in the Release file to make sure their start and end dates are within the range. )")
    # year month day
start_date = st.text_input('Start Date', start_date_from_file)
end_date = st.text_input('End Date', end_date_from_file)

start_day_from_file = datetime.strptime(start_date_from_file, "%Y %m %d")
end_day_from_file = datetime.strptime(end_date_from_file, "%Y %m %d")
start_day_from_user = datetime.strptime(start_date, "%Y %m %d")
end_day_from_user = datetime.strptime(end_date, "%Y %m %d")

if (start_day_from_user < start_day_from_file):
    st.markdown("**Error:** your Start Date is outside the range.")
if (end_day_from_user > end_day_from_file):
    st.markdown("**Error:** your End Date is outside the range.")
    
if st.button(label="Click to Run ChemFate"):
    st.write("ChemFate Model Started to Run ......")
    model = Model_SetUp(start_date, end_date, run_option, bgPercOption2,
                        chem_type, chem_file, region_file, release_file, output_file_path, file_name)
    model.run_model()
    create_download_zip(zip_directory=output_file_path,
                        zip_path=output_file_path,
                        filename=file_name + '.zip')

credit_txt1 = "_nanoFate was developed by Dr. Kendra Garner and Dr. Arturo Keller_"
credit_txt2 = "_organoFate, ionOFate, and metalFate were developed by Dr. Mengya Tao and Dr. Arturo Keller_"
credit_txt3 = "_Questions: email arturokeller@ucsb.edu; Keller Lab, Bren School, UC Santa Barbara, USA_"
st.write("\n")
st.write("\n")
st.write("\n")
st.markdown(credit_txt1)
st.write(credit_txt2)
st.write(credit_txt3)

