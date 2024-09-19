import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from fuzzywuzzy import process
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


df = pd.read_csv('supercar_dataset.csv')


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
car_names = df['name'].tolist()
car_name_embeddings = model.encode(car_names, convert_to_tensor=True)


generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')


def search_car_hp_torque(query):
    corrected_query, score = process.extractOne(query, car_names)
    query_embedding = model.encode(corrected_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, car_name_embeddings)[0]
    top_result_index = scores.argmax().item()
    return df.iloc[top_result_index]['name'], df.iloc[top_result_index]['hp'], df.iloc[top_result_index]['Torque']

def process_complex_query(query):
    result = generator(query, max_length=50, num_return_sequences=1)[0]['generated_text']
    return result


def generate_car_description(car_name):
    prompt = f"Tell me about the car {car_name}. What makes it special?"
    description = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return description

st.title('Supercar Information & Query Processing')
st.write('Type a query to get information about a supercar, like horsepower, torque, or a car description.')


query = st.text_input('Enter your query:').lower()

if query:
    if "horsepower" in query.lower():
        car, hp, torque = search_car_hp_torque(query)
        st.write(f"**Car:** {car}")
        st.write(f"**HP:** {hp}")
         
    elif "torque" in query.lower():
        car,hp,torque=search_car_hp_torque(query)
        st.write(f"**Car:** {car}")
        st.write(f"**Torque:** {torque}")
    else:
       
        complex_result = process_complex_query(query)
        st.write(f"**Answer:** {complex_result}")


    car_name = query.split()[-1]  
    description = generate_car_description(car_name)
    st.write(f"**Car Description:** {description}")
