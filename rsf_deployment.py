# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 17:21:42 2023

@author: mitul
"""
import pandas as pd
import numpy as np
import pickle 
import streamlit as st


pickle_in = open("rsf1.pkl","rb")
rsf1=pickle.load(pickle_in)

# Creating a function for prediction 
def graft_surv_pred(input_data):    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = rsf1.predict(input_data_reshaped)
    
    surv_test_rsf  = rsf1.predict_survival_function(input_data_reshaped, return_array=False)
    surv = rsf1.predict_survival_function(input_data_reshaped, return_array=True)

    event_times = rsf1.event_times_

    lower, upper = event_times[0], event_times[-1]
    y_times = np.arange(lower, upper)

    T1, T2 = surv_test_rsf [0].x.min(),surv_test_rsf [0].x.max()
    mask = np.logical_or(y_times >= T2, y_times < T1) # mask outer interval
    times = y_times[~mask]

    rsf_surv_prob_test  = np.row_stack([fn(times) for fn in surv_test_rsf])
    rsf_surv_prob_test1 = pd.DataFrame(rsf_surv_prob_test)

    After_1_year = rsf_surv_prob_test1.iloc[ :,2]*100 
    
    return(('Probability of survival at After 1 Year :', After_1_year.to_string(index=False)))   

def main():   
# Every good app has a title, so let's add one
    st.title('Risk modeling for Renal Post-Transplant complications')
    st.text('Domain: Life Sciences  & Health Care')
    st.text('Project type: Research Project')
    
    st.text('\nThe parameters are as follows: \n')    
    # Getting input data from user
    Systolicbloodpressure = st.text_input("Systolicbloodpressure", "Type Here")
    Diastolicbloodpressure = st.text_input("Diastolicbloodpressure", "Type Here")
    Bodyweight = st.text_input("Bodyweight", "Type Here")
    Whitebloodcellcount = st.text_input("Whitebloodcellcount", "Type Here")
    Hemoglobinlevel = st.text_input("Hemoglobinlevel", "Type Here")
    Platelets = st.text_input("Platelets", "Type Here")
    Serumcreatininelevel = st.text_input("Serumcreatininelevel", "Type Here")
    BloodUreaNitrogenlevel = st.text_input("BloodUreaNitrogenlevel", "Type Here")
    Glucoselevel = st.text_input("Glucoselevel", "Type Here")
    Potassiumlevel = st.text_input("Potassiumlevel", "Type Here")
    Sodiumlevel = st.text_input("Sodiumlevel", "Type Here")
    Calciumlevel = st.text_input("Calciumlevel", "Type Here")
    Phosphoruslevel = st.text_input("Phosphoruslevel", "Type Here")
    Tac_MR = st.text_input("Tac_MR", "Type Here")
    Recipientsage = st.text_input("Recipientsage", "Type Here")
    Donorage = st.text_input("Donorage", "Type Here")
    Bodymassindex = st.text_input("Bodymassindex", "Type Here")
    Numberofposttransplantadmission = st.text_input("Numberofposttransplantadmission", "Type Here")
    Durationsincetransplant = st.text_input("Durationsincetransplant", "Type Here")
    eGFR = st.text_input("eGFR", "Type Here")
    Recipientssex = st.text_input("Recipientssex", "Type Here")
    RecipientsReligion = st.text_input("RecipientsReligion", "Type Here")
    Recipientslevelofeducation = st.text_input("Recipientslevelofeducation", "Type Here")
    Recipientsemploymentstatus= st.text_input("Recipientsemploymentstatus", "Type Here")
    Recipientsresidence = st.text_input("Recipientsresidence", "Type Here")
    Donorsex = st.text_input("Donorsex", "Type Here")
    Donortorecipientrelationship = st.text_input("Donortorecipientrelationship", "Type Here")
    Placecenterofallograft = st.text_input("Placecenterofallograft", "Type Here")
    Posttransplantregularphysicale = st.text_input("Posttransplantregularphysicale", "Type Here")
    Pretransplanthistoryofsubstanc = st.text_input("Pretransplanthistoryofsubstanc", "Type Here")
    Posttransplantnonadherence= st.text_input("Posttransplantnonadherence", "Type Here")
    CausesofEndStageRenalDisea = st.text_input("CausesofEndStageRenalDisea", "Type Here")
    Historyofpretransplantcomorbid = st.text_input("Historyofpretransplantcomorbid", "Type Here")
    Historyofdialysisbeforetranspl = st.text_input("Historyofdialysisbeforetranspl", "Type Here")
    Historyofbloodtransfusi= st.text_input("Historyofbloodtransfusi", "Type Here")
    Historyofabdominalsurge = st.text_input("Historyofabdominalsurge", "Type Here")
    Familyhistoryofkidneydisea = st.text_input("Familyhistoryofkidneydisea", "Type Here")
    Posttransplantmalignan = st.text_input("Posttransplantmalignan", "Type Here")
    PosttransplantUrologicalcompli = st.text_input("PosttransplantUrologicalcompli", "Type Here")
    PosttransplantVascularcomplica = st.text_input("PosttransplantVascularcomplica", "Type Here")
    PosttransplantCardiovascularco= st.text_input("PosttransplantCardiovascularco", "Type Here")
    PosttransplantInfection = st.text_input("PosttransplantInfection", "Type Here")
    Posttransplantdiabetes = st.text_input("Posttransplantdiabetes", "Type Here")
    Posttransplanthypertension = st.text_input("Posttransplanthypertension", "Type Here")
    Anepisodeofacuterejection = st.text_input("Anepisodeofacuterejection", "Type Here")
    Anepisodeofchronicrejection = st.text_input("Anepisodeofchronicrejection", "Type Here")
    PosttransplantGastrointestin = st.text_input("PosttransplantGastrointestin", "Type Here")
    Anepisodeofhyperacuterejection = st.text_input("Anepisodeofhyperacuterejection", "Type Here")
    Posttransplantglomerulonephrit = st.text_input("Posttransplantglomerulonephrit", "Type Here")
    Posttransplantdelayedgraftfunc = st.text_input("Posttransplantdelayedgraftfunc", "Type Here")
    Posttransplantfluidoverloa = st.text_input("Posttransplantfluidoverloa", "Type Here")
    PosttransplantCovid19 = st.text_input("PosttransplantCovid19", "Type Here")
    marital_statuss = st.text_input("marital_statuss","Type Here")
    postwaterintakee = st.text_input("postwaterintakee", "Type Here")
    
    # Code for prediction
    diagnosis = ''
             
    # Creating a button for prediction
    if st.button('Predict graft survival'):
        diagnosis = graft_surv_pred([Systolicbloodpressure, Diastolicbloodpressure, Bodyweight, Whitebloodcellcount,
                                     Hemoglobinlevel,Platelets,Serumcreatininelevel, BloodUreaNitrogenlevel,
                                     Glucoselevel,Potassiumlevel, Sodiumlevel, Calciumlevel, Phosphoruslevel,
                                     Tac_MR,Recipientsage, Donorage, Bodymassindex, Numberofposttransplantadmission,
                                     Durationsincetransplant,eGFR, Recipientssex,RecipientsReligion, Recipientslevelofeducation, Recipientsemploymentstatus, Recipientsresidence, Donorsex, Donortorecipientrelationship,Placecenterofallograft ,
                                     Posttransplantregularphysicale,  Pretransplanthistoryofsubstanc, Posttransplantnonadherence, CausesofEndStageRenalDisea,
                                     Historyofpretransplantcomorbid, Historyofdialysisbeforetranspl, Historyofbloodtransfusi,
                                     Historyofabdominalsurge, Familyhistoryofkidneydisea, Posttransplantmalignan, PosttransplantUrologicalcompli,PosttransplantVascularcomplica,
                                     PosttransplantCardiovascularco, PosttransplantInfection, Posttransplantdiabetes,
                                     Posttransplanthypertension, Anepisodeofacuterejection, Anepisodeofchronicrejection,
                                     PosttransplantGastrointestin, Anepisodeofhyperacuterejection,Posttransplantglomerulonephrit, Posttransplantdelayedgraftfunc, Posttransplantfluidoverloa, PosttransplantCovid19,
                                     marital_statuss, postwaterintakee])
        
    st.info(diagnosis)    
    
if __name__ == '__main__':
    main()
    
    
