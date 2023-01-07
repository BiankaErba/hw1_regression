from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List

from io import StringIO
import pandas as pd
import pickle

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Csv(BaseModel):
    content: str

EXPECTED_COLUMNS = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
INCLUDE = [*[str(i) for i in range(0, 10)], '.']
MODEL = pickle.load(open('lasso_model.sav', 'rb'))

def predict_by_df(df_test):
	df_test = df_test[df_test.columns[df_test.columns.isin(EXPECTED_COLUMNS)]]

	def float_converter(value):
		if type(value) != str:
			return value
		if pd.isna(value):
			return None
		return ''.join([i for i in value if i in INCLUDE])

	for col in EXPECTED_COLUMNS:
		df_test[col] = df_test[col].apply(float_converter).astype('float64')
		
	return MODEL.predict(df_test.dropna())

app = FastAPI()

@app.post("/predict_item")
def predict_item(item: Item) -> float:
	predicted = predict_items([item])
	return predicted[0] if len(predicted) else None

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
	return [float(i) for i in predict_by_df(pd.DataFrame(jsonable_encoder(items)))]

@app.post("/csv_content")
def csv_content(content: Csv) -> Csv:
	df = pd.read_csv(StringIO(content.content))
	df['predict'] = predict_by_df(df)
	csv_buffer = StringIO()
	df.to_csv(csv_buffer, index=False)
	content.content = csv_buffer.getvalue()
	return content
